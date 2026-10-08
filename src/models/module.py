"""MammoCycleModule learns correspondence between exams of one breast from the cycle-consistency of tracking.

Batches are {'frames': [B, T, 1, H, W]} in [0, 1]: the exams of one breast and view, oldest first, with T >= 2 constant
within a batch. The last frame is the query frame. H and W are multiples of the encoder's stride, and the background
is below tissue_threshold.
"""
import math
from dataclasses import dataclass

import lightning as L
import torch
import torch.nn.functional as F
from torch import Tensor

from src.data.augmentations import synthetic_prior

from .encoder import ResNetEncoder
from .patch_sampler import sample_patches
from .propagation import box_coverage, locate, propagate_labels
from .tracker import Hop, Tracker, cycle_loss, patch_grid, place_layout, run_cycles, sample_features, similarity_loss


SYNTHETIC_BOX_SIZE = 64  # pixels; the boxes of the synthetic validation pairs, about the size of a lesion


def full_precision(tensor: Tensor) -> torch.autocast:
    """Switch mixed precision off for the tracker.

    The affinity and the pixel arithmetic need float32: half precision resolves a coordinate near 1000 only to half a
    pixel. Converting inputs with .float() is not enough, because autocast runs matrix products in half precision.
    """
    return torch.autocast(tensor.device.type, enabled=False)


@dataclass
class StepOutput:
    loss: Tensor  # scalar
    cycle_loss: Tensor  # scalar, mean over cycles
    similarity_loss: Tensor  # scalar, mean over priors
    cycle_losses: dict[str, Tensor]  # 'skip_1', 'skip_2', 'long_2', ... -> [B * P]
    cycle_errors: dict[str, Tensor]  # the same keys; mean cell distance in pixels, detached
    similarity_losses: dict[str, Tensor]  # 'prior_1', 'prior_2', ... -> [B * P]
    first_hop: Hop  # from the query into the most recent prior, with diagnostics
    query: Tensor  # [B * P, C, hq, wq]
    start_grid: Tensor  # [B * P, hq * wq, 2]
    frame_features: list[Tensor]  # T maps [B * P, C, hf, wf], oldest first, each repeated per patch


def background_landing(points: Tensor, tissue: Tensor) -> Tensor:
    """Share of matched positions that fall outside tissue: certain failures. points [B, N, 2], tissue [B, 1, H, W].

    Positions outside the frame count as background. Returns [B].
    """
    size = points.new_tensor([tissue.shape[-1], tissue.shape[-2]])
    grid = (2 * points / size - 1)[:, :, None]  # [B, N, 1, 2]
    on_tissue = F.grid_sample(tissue.float(), grid, mode='nearest', padding_mode='zeros', align_corners=False)

    return 1 - on_tissue.flatten(1).mean(dim=1)


def box_metrics(center: Tensor, box: Tensor, target_box: Tensor) -> dict[str, Tensor]:
    """Per-sample location metrics. center [B, 2], box and target_box [B, 4] -> three tensors of shape [B]."""
    target_center = (target_box[:, :2] + target_box[:, 2:]) / 2
    inside = (center >= target_box[:, :2]) & (center <= target_box[:, 2:])

    overlap = (torch.minimum(box[:, 2:], target_box[:, 2:]) - torch.maximum(box[:, :2], target_box[:, :2])).clamp_min(0)
    area = lambda boxes: (boxes[:, 2:] - boxes[:, :2]).clamp_min(0).prod(dim=1)
    intersection = overlap.prod(dim=1)

    return {'hit_rate': inside.all(dim=1).float(),
            'center_distance_px': (center - target_center).norm(dim=1),
            'iou': intersection / (area(box) + area(target_box) - intersection).clamp_min(1e-6)}


class MammoCycleModule(L.LightningModule):
    """Tracks patches of the newest exam back through the older ones and forward again, and trains the encoder so the
    patches return to where they started. Only the encoder has parameters: the tracker and the patch sampling are fixed
    formulas. Its features are the product, matched directly at inference (locate_by_propagation, locate_by_window).
    """

    def __init__(self, encoder: ResNetEncoder, patch_size: int = 160, num_patches: int = 4,
                 min_tissue_fraction: float = 0.75, tissue_threshold: float = 0.01, temperature: float = 0.03,
                 max_rotation_degrees: float = 20.0, cycle_weight: float = 1.0, similarity_weight: float = 1.0,
                 huber_delta: float | None = None, top_k: int = 5, propagation_radius: float | None = None,
                 label_threshold: float = 0.5, cycle_tolerance_px: float = 16.0, learning_rate: float = 1e-4,
                 weight_decay: float = 1e-4, warmup_steps: int = 1000):
        super().__init__()
        if patch_size % encoder.stride:
            raise ValueError(f'{patch_size = } must be a multiple of {encoder.stride = }')

        self.save_hyperparameters(ignore=['encoder'])
        self.encoder = encoder
        self.tracker = Tracker(encoder.stride, temperature, max_rotation_degrees)
        self._epoch_errors: list[Tensor] = []

    def encode(self, images: Tensor) -> Tensor:
        """[N, 1, H, W] -> L2-normalized features [N, C, H / stride, W / stride] in float32."""
        return F.normalize(self.encoder(images).float(), dim=1)

    def _sample_positions(self, frames: Tensor, count: int, size: int, generator: torch.Generator | None) -> Tensor:
        hp = self.hparams
        return sample_patches(frames, count, size, self.encoder.stride, hp.min_tissue_fraction, hp.tissue_threshold,
                              generator)

    # ------------------------------------------------------------------ training

    def cycle_step(self, frames: Tensor, generator: torch.Generator | None = None) -> StepOutput:
        """Track patches of the last frame through every cycle and compute the loss. frames [B, T, 1, H, W]."""
        hp, stride = self.hparams, self.encoder.stride
        cells = hp.patch_size // stride
        top_left = self._sample_positions(frames[:, -1], hp.num_patches, hp.patch_size, generator)  # [B, P, 2]

        features = self.encode(frames.flatten(0, 1)).unflatten(0, frames.shape[:2])  # [B, T, C, hf, wf]
        # The tracker works on B * P patches: sample b's patches are rows b * P ... b * P + P - 1.
        frame_features = [feature.repeat_interleave(hp.num_patches, dim=0) for feature in features.unbind(dim=1)]

        with full_precision(frames):
            # Patches lie on the stride grid, so reading the query from the frame's own features is exact.
            start_grid = patch_grid(top_left.flatten(0, 1), cells, cells, stride)
            query = sample_features(frame_features[-1], start_grid, stride, (cells, cells))
            cycles, first_hops = run_cycles(self.tracker, query, frame_features)

            cycle_losses = {name: cycle_loss(start_grid, end_grid, hp.patch_size, hp.huber_delta)
                            for name, end_grid in cycles.items()}
            cycle_errors = {name: (end_grid - start_grid).norm(dim=-1).mean(dim=1).detach()
                            for name, end_grid in cycles.items()}
            similarity_losses = {f'prior_{i}': similarity_loss(query, hop.features)
                                 for i, hop in enumerate(first_hops, start=1)}

            # Means over cycles, not sums, so the loss keeps its scale for any number of exams.
            mean_cycle = torch.stack(list(cycle_losses.values())).mean()
            mean_similarity = torch.stack(list(similarity_losses.values())).mean()
            loss = hp.cycle_weight * mean_cycle + hp.similarity_weight * mean_similarity

        return StepOutput(loss, mean_cycle, mean_similarity, cycle_losses, cycle_errors, similarity_losses,
                          first_hops[0], query, start_grid, frame_features)

    def on_fit_start(self):
        if not self.encoder.calibrated:
            raise RuntimeError('The encoder is not calibrated: run scripts/calibrate_encoder.py first. '
                               'Without it the affinity is almost flat and nothing is learned.')

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        output = self.cycle_step(batch['frames'])
        batch_size = len(batch['frames'])
        self.log('train/loss', output.loss, batch_size=batch_size, prog_bar=True)
        self.log_dict({f'train/{name}': value for name, value in self._step_metrics(output).items()},
                      batch_size=batch_size)

        return output.loss

    @staticmethod
    def _step_metrics(output: StepOutput) -> dict[str, Tensor]:
        return {
            'cycle_loss': output.cycle_loss,
            'similarity_loss': output.similarity_loss,
            **{f'cycle_loss/{name}': value.mean() for name, value in output.cycle_losses.items()},
            **{f'cycle_error_px/{name}': value.mean() for name, value in output.cycle_errors.items()},
            **{f'similarity_loss/{name}': value.mean() for name, value in output.similarity_losses.items()},
            # How sharp and how consistent the matches into the most recent prior are.
            **{f'localizer/{name}': value.mean() for name, value in output.first_hop.diagnostics.items()},
        }

    # ------------------------------------------------------------------ validation and test

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int):
        self._evaluate('val', batch['frames'], batch_idx)

    def test_step(self, batch: dict[str, Tensor], batch_idx: int):
        self._evaluate('test', batch['frames'], batch_idx)

    def on_validation_epoch_end(self):
        self._log_epoch_errors('val')

    def on_test_epoch_end(self):
        self._log_epoch_errors('test')

    def _evaluate(self, stage: str, frames: Tensor, batch_idx: int):
        hp = self.hparams
        # The same patches and synthetic warps every epoch, so the numbers are comparable between epochs.
        generator = torch.Generator().manual_seed(batch_idx)
        output = self.cycle_step(frames, generator)
        self._epoch_errors.append(output.cycle_errors['skip_1'])
        metrics = {'loss': output.loss, **self._step_metrics(output)}

        with full_precision(frames):
            if len(frames) > 1:
                other = self.different_patient_error(output).mean()
                metrics['different_patient_error_px'] = other
                metrics['different_patient_ratio'] = other / output.cycle_errors['skip_1'].mean().clamp_min(1e-6)

            tissue = (frames[:, -2] > hp.tissue_threshold).repeat_interleave(hp.num_patches, dim=0)
            metrics['background_landing'] = background_landing(output.first_hop.points, tissue).mean()
            metrics |= self._synthetic_box_metrics(frames[:, -1], generator)

        self.log_dict({f'{stage}/{name}': value for name, value in metrics.items()}, batch_size=len(frames),
                      sync_dist=True)

    @torch.no_grad()
    def different_patient_error(self, output: StepOutput) -> Tensor:
        """Error in pixels of the length-1 skip cycle when the prior comes from the next sample in the batch. [B * P].

        A model that matches tissue closes same-patient cycles far better than these. If the two errors are close, it
        matches by outline and position. Needs a batch of at least two.
        """
        other_prior = output.frame_features[-2].roll(self.hparams.num_patches, dims=0)
        there = self.tracker(output.query, other_prior)
        back = self.tracker(there.features, output.frame_features[-1])

        return (back.grid - output.start_grid).norm(dim=-1).mean(dim=1)

    def _synthetic_box_metrics(self, current: Tensor, generator: torch.Generator) -> dict[str, Tensor]:
        """Box metrics on synthetic pairs: a box at a random tissue position, and a fake prior from a known warp.

        Both inference methods should beat the identity baseline, the same coordinates on the prior.

        TODO(annotated evaluation): the same metrics on real cancer pairs, once they exist.
        """
        top_left = self._sample_positions(current, 1, SYNTHETIC_BOX_SIZE, generator)[:, 0].float()
        boxes = torch.cat([top_left, top_left + SYNTHETIC_BOX_SIZE], dim=1)
        prior, target_boxes = synthetic_prior(current, boxes, generator=generator)

        results = {'propagate': self.locate_by_propagation(current, prior, boxes),
                   'window': self.locate_by_window(current, prior, boxes),
                   'identity': {'center': (boxes[:, :2] + boxes[:, 2:]) / 2, 'box': boxes}}

        return {f'box/{method}/{name}': value.mean()
                for method, result in results.items()
                for name, value in box_metrics(result['center'], result['box'], target_boxes).items()}

    def _log_epoch_errors(self, stage: str):
        if self._epoch_errors:
            # Per process under multi-GPU training; sync_dist averages them over processes, an approximation.
            errors = torch.cat(self._epoch_errors).float()
            self.log_dict({f'{stage}/cycle_error_px/median': errors.median(),
                           f'{stage}/cycle_error_px/p90': errors.quantile(0.9),
                           f'{stage}/cycle_closed_share': (errors < self.hparams.cycle_tolerance_px).float().mean()},
                          sync_dist=True)
        self._epoch_errors = []

    # ------------------------------------------------------------------ inference

    @torch.no_grad()
    def locate_by_propagation(self, source: Tensor, target: Tensor, boxes: Tensor) -> dict[str, Tensor]:
        """Find on target the region that boxes [B, 4] (x1, y1, x2, y2 pixels) mark on source; both [B, 1, H, W].

        Every target cell takes the label of its top_k most similar source cells, the paper's test-time method.
        Returns center [B, 2] and box [B, 4] of the target cells above label_threshold, score [B] (the map's peak),
        found [B] (whether any cell passed; if not, center and box are the peak cell) and the lesion map [B, h, w].

        TODO(mask inference): propagate a segmentation itself, not its bounding box.
        """
        hp, stride = self.hparams, self.encoder.stride
        source_features, target_features = self.encode(source), self.encode(target)
        with full_precision(source):
            labels = box_coverage(boxes.float(), *source_features.shape[-2:], stride)[:, None]
            lesion = propagate_labels(source_features, target_features, labels, self.tracker.temperature, stride,
                                      hp.top_k, hp.propagation_radius)[:, 0]

            return {**locate(lesion, stride, hp.label_threshold), 'map': lesion}

    @torch.no_grad()
    def locate_by_window(self, source: Tensor, target: Tensor, boxes: Tensor,
                         context: float = 1.5) -> dict[str, Tensor]:
        """Find on target the region that boxes mark on source by tracking a window around each box with one hop.

        The window is patch_size wide, or context times the longer box side when that is larger, and is centred on the
        box as far as the frame allows. It matches on the surrounding tissue, which is more robust than propagation
        when the lesion itself looks different on the two exams. Returns center [B, 2], box [B, 4] (the query box moved
        by the fitted shift and rotation, so it keeps its size), round_trip_error [B] (pixels between the window and
        where it lands when tracked back) and similarity [B] (mean cosine similarity of the window and its match).
        """
        stride = self.encoder.stride
        height, width = source.shape[-2:]
        source_features, target_features = self.encode(source), self.encode(target)
        results = []

        with full_precision(source):
            for box, source_map, target_map in zip(boxes.float(), source_features, target_features):
                # One box at a time, because windows differ in size.
                longest = (box[2:] - box[:2]).max().item()
                cells = min(max(self.hparams.patch_size // stride, math.ceil(context * longest / stride)),
                            min(height, width) // stride)
                size = cells * stride
                top_left = ((box[:2] + box[2:] - size) / 2 / stride).round() * stride  # on the stride grid
                top_left = top_left.clamp(box.new_zeros(2), box.new_tensor([width - size, height - size]))

                start_grid = patch_grid(top_left[None], cells, cells, stride)
                query = sample_features(source_map[None], start_grid, stride, (cells, cells))
                hop = self.tracker(query, target_map[None])
                back = self.tracker(hop.features, source_map[None])

                # Rotate the box's corners about the window centre and move them to the matched centre.
                corners = torch.stack([box[[0, 1]], box[[2, 1]], box[[2, 3]], box[[0, 3]]])
                moved = place_layout((corners - (top_left + size / 2))[None], hop.center, hop.angle)[0]
                results.append({'center': moved.mean(dim=0),
                                'box': torch.cat([moved.amin(dim=0), moved.amax(dim=0)]),
                                'round_trip_error': (back.grid - start_grid).norm(dim=-1).mean(),
                                'similarity': 1 - similarity_loss(query, hop.features)[0]})

        return {key: torch.stack([result[key] for result in results]) for key in results[0]}

    # ------------------------------------------------------------------ optimization

    def configure_optimizers(self):
        hp = self.hparams
        optimizer = torch.optim.AdamW(self.parameters(), lr=hp.learning_rate, weight_decay=hp.weight_decay)
        total_steps = self.trainer.estimated_stepping_batches

        def schedule(step: int) -> float:  # linear warm-up, then cosine decay
            if step < hp.warmup_steps:
                return (step + 1) / hp.warmup_steps
            progress = (step - hp.warmup_steps) / max(total_steps - hp.warmup_steps, 1)
            return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)

        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}
