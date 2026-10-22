"""
Step D: a Siamese matcher that outputs a probability map over candidate
lesion positions in a search window on the prior, plus the same anatomical
prior fusion as B5 (the network replaces NCC as the appearance score).

Pipeline for one (current, prior) pair
--------------------------------------
  template  : current image, T x T px centered on the current lesion,
              + a 2nd channel = box mask (where the lesion is)          (2,T,T)
  search    : prior image, S x S px centered on the anatomical prediction (1,S,S)
  encoder   : shared ResNet-18 (stride 8)   -> (C,T/8,T/8), (C,S/8,S/8)
  kernel    : central K x K cells of the template features              (C,K,K)
  xcorr     : depthwise cross-correlation of kernel over search          (C,Ho,Wo)
  head      : 1x1 convs -> 1 logit per cell                              (Ho,Wo)
  output    : softmax over cells = p(lesion center at cell | appearance)
  inference : log_softmax(logits)/tau + log p_anat(cell) -> argmax -> image px

Default sizes (0.2 mm/px): T=256 (51 mm), S=512 (102 mm), stride 8 (1.6 mm
cells), K=9 -> 56 x 56 output cells covering +-44.8 mm around the window center.

Geometry: with a stride-8 encoder, feature index a sits at input pixel 8a + d
(the same d for both branches, since they share weights). If the kernel is
centered at template feature c_t, output cell j means "lesion center at
search pixel  8 * (j + (K-1)/2 - c_t) + (T-1)/2"; d cancels. See
`grid_affine`, checked numerically in test_stepD.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from b5_prior_fusion import ScoreMap
from geometry import apply_affine, compose, crop_affine, invert_affine, warp_crop


# --------------------------------------------------------------------------- #
# Configuration and output-grid geometry
# --------------------------------------------------------------------------- #
@dataclass
class MatcherConfig:
    spacing_mm: float = 0.2        # working resolution (images pre-resampled)
    template_px: int = 256         # 51.2 mm of context around the current lesion
    search_px: int = 512           # 102.4 mm window on the prior
    stride: int = 8                # encoder output stride
    kernel_cells: int = 9          # 9 x 1.6 mm = 14.4 mm correlation kernel
    feat_ch: int = 128
    use_layer3: bool = True        # dilated layer3: bigger receptive field, ~3x compute
    pretrained: bool = True        # ImageNet init (grayscale-adapted)
    target_sigma_mm: float = 2.0   # width of the soft target (≈ annotation noise)
    # training-time sampling of the search window
    p_uniform_center: float = 0.7  # fraction of windows with GT uniformly placed
    anat_jitter_mm: float = 2.0    # jitter when centering on the anatomical point
    rot_deg: float = 10.0          # search-crop rotation augmentation
    scale_aug: float = 0.10        # search-crop scale augmentation (+-10 %)

    @property
    def c_t(self) -> int:          # template feature index used as kernel center
        return (self.template_px // self.stride) // 2

    @property
    def out_cells(self) -> int:
        return self.search_px // self.stride - self.kernel_cells + 1

    @property
    def cell_mm(self) -> float:
        return self.stride * self.spacing_mm

    @property
    def max_offset_mm(self) -> float:
        """Largest displacement from the window center the grid can represent."""
        return (self.out_cells - 1) / 2 * self.cell_mm


def grid_affine(cfg: MatcherConfig) -> np.ndarray:
    """2x3: output cell (j, i) -> search-crop pixel of the lesion center."""
    off = cfg.stride * ((cfg.kernel_cells - 1) / 2 - cfg.c_t) + (cfg.template_px - 1) / 2
    return np.array([[cfg.stride, 0, off], [0, cfg.stride, off]], float)


def search_px_to_grid(cfg, xy) -> np.ndarray:
    return apply_affine(invert_affine(grid_affine(cfg)), xy)


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #
def _dilate_layer(layer: nn.Sequential, dilation: int):
    """Turn a stride-2 ResNet stage into stride 1 with dilated 3x3 convs, so the
    output stride stays 8 while the receptive field grows (SiamRPN++ trick)."""
    for m in layer.modules():
        if isinstance(m, nn.Conv2d):
            if m.stride == (2, 2):
                m.stride = (1, 1)
            if m.kernel_size == (3, 3):
                m.dilation = (dilation, dilation)
                m.padding = (dilation, dilation)


class Encoder(nn.Module):
    def __init__(self, cfg: MatcherConfig, in_ch: int = 2):
        super().__init__()
        weights = "IMAGENET1K_V1" if cfg.pretrained else None
        r = torchvision.models.resnet18(weights=weights)
        conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
        with torch.no_grad():
            conv1.weight.zero_()
            conv1.weight[:, 0] = r.conv1.weight.sum(1)   # gray image = sum of RGB filters
            # channel 1 (box mask) starts at 0: pretrained features unchanged at init
        self.stem = nn.Sequential(conv1, r.bn1, r.relu, r.maxpool)   # /4
        self.layer1, self.layer2 = r.layer1, r.layer2                # /4, /8 (128 ch)
        self.use_layer3 = cfg.use_layer3
        if cfg.use_layer3:
            _dilate_layer(r.layer3, 2)
            self.layer3 = r.layer3                                   # /8 (256 ch)
        self.out_ch = 128 + (256 if cfg.use_layer3 else 0)

    def forward(self, x):
        x = self.layer1(self.stem(x))
        f2 = self.layer2(x)
        if not self.use_layer3:
            return f2
        return torch.cat([f2, self.layer3(f2)], 1)


def dw_xcorr(search: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Depthwise cross-correlation: each channel of each sample's kernel slides
    over the same channel of that sample's search features. (B,C,H,W) x
    (B,C,k,k) -> (B,C,H-k+1,W-k+1), done as one grouped conv."""
    B, C, H, W = search.shape
    k = kernel.shape[-1]
    out = F.conv2d(search.reshape(1, B * C, H, W), kernel.reshape(B * C, 1, k, k), groups=B * C)
    return out.reshape(B, C, H - k + 1, W - k + 1)


def conv_bn_relu(cin, cout, k):
    return nn.Sequential(nn.Conv2d(cin, cout, k, padding=k // 2, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class SiameseMatcher(nn.Module):
    def __init__(self, cfg: MatcherConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        C = cfg.feat_ch
        self.neck = nn.Sequential(nn.Conv2d(self.encoder.out_ch, C, 1, bias=False),
                                  nn.BatchNorm2d(C))                       # shared
        self.adj_t = conv_bn_relu(C, C, 3)                                  # template-only
        self.adj_s = conv_bn_relu(C, C, 3)                                  # search-only
        self.head = nn.Sequential(conv_bn_relu(C, C, 1), nn.Conv2d(C, 1, 1))

    def kernel(self, template):
        z = self.adj_t(self.neck(self.encoder(template)))
        c, h = self.cfg.c_t, (self.cfg.kernel_cells - 1) // 2
        return z[:, :, c - h:c + h + 1, c - h:c + h + 1]                    # (B,C,K,K)

    def forward(self, template, search):
        """template (B,2,T,T) [image, box mask]; search (B,1,S,S) -> logits (B,Ho,Wo)."""
        search = torch.cat([search, torch.zeros_like(search)], 1)          # empty mask channel
        x = self.adj_s(self.neck(self.encoder(search)))
        r = dw_xcorr(x, self.kernel(template))
        return self.head(r)[:, 0]


def soft_target_ce(logits, gt_grid, sigma_cells, valid=None):
    """Cross-entropy between the spatial softmax and a small Gaussian around the
    GT cell. A softmax over cells is a distribution over the lesion location,
    so its log can be added to the log prior at inference. valid [B, H, W], if
    given, marks the cells the logits can reach; the target keeps off the rest."""
    B, H, W = logits.shape
    jj = torch.arange(W, device=logits.device).view(1, 1, W)
    ii = torch.arange(H, device=logits.device).view(1, H, 1)
    d2 = (jj - gt_grid[:, 0].view(B, 1, 1)) ** 2 + (ii - gt_grid[:, 1].view(B, 1, 1)) ** 2
    q = torch.exp(-d2 / (2 * sigma_cells ** 2))
    if valid is not None:
        q = q * valid  # masked logits are -1e20, so any target mass there swamps the loss
    q = q / q.sum((1, 2), keepdim=True).clamp_min(1e-30)
    logp = F.log_softmax(logits.reshape(B, -1), -1).reshape(B, H, W)
    return -(q * logp).sum((1, 2)).mean()


# --------------------------------------------------------------------------- #
# Data: crops, jitter, augmentation
# --------------------------------------------------------------------------- #
def box_mask(T, box_wh):
    m = np.zeros((T, T), np.float32)
    c = (T - 1) / 2
    hw, hh = box_wh[0] / 2, box_wh[1] / 2
    x0, x1 = int(np.floor(c - hw)), int(np.ceil(c + hw))
    y0, y1 = int(np.floor(c - hh)), int(np.ceil(c + hh))
    m[max(0, y0):min(T, y1 + 1), max(0, x0):min(T, x1 + 1)] = 1.0
    return m


def normalize_crop(x, rng=None):
    """Per-crop standardization (robust to exposure/vendor change between years),
    with optional gamma/contrast/noise augmentation."""
    x = x.astype(np.float32)
    if rng is not None:
        lo, hi = np.percentile(x, [1, 99])
        x = np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)
        x = x ** rng.uniform(0.7, 1.4)
        x = x * rng.uniform(0.8, 1.2) + rng.normal(0, 0.02, x.shape).astype(np.float32)
    return (x - x.mean()) / (x.std() + 1e-6)


class PairDataset(torch.utils.data.Dataset):
    """Each case: dict with img_cur, img_prior (working spacing), p_cur, box_wh,
    p_gt (prior), p_anat (prior); optional box_wh_prior to also train the
    reverse direction (prior -> current).

    train=True : search windows centered either on p_gt + uniform offset
                 (prob p_uniform_center: GT anywhere on the grid, so the network
                 cannot learn 'answer is near the center' and stays a pure
                 appearance matcher) or on p_anat + small jitter (realistic
                 test-time context); plus rotation/scale/intensity augmentation.
                 Every epoch draws new windows, so a pair is never seen twice
                 at the same offset -> no offset memorization.
    train=False: window centered exactly on p_anat, no augmentation.
    """

    def __init__(self, cases, cfg: MatcherConfig, train: bool, epoch_len: int | None = None):
        self.cases, self.cfg, self.train = cases, cfg, train
        self.epoch_len = epoch_len or len(cases)

    def __len__(self):
        return self.epoch_len if self.train else len(self.cases)

    def __getitem__(self, idx):
        cfg = self.cfg
        rng = np.random.default_rng() if self.train else None
        c = self.cases[idx % len(self.cases)]
        T, S = cfg.template_px, cfg.search_px

        src, dst, p_src, p_dst, box = c["img_cur"], c["img_prior"], c["p_cur"], c["p_gt"], c["box_wh"]
        reverse = self.train and "box_wh_prior" in c and rng.random() < 0.5
        if reverse:
            src, dst, p_src, p_dst, box = c["img_prior"], c["img_cur"], c["p_gt"], c["p_cur"], c["box_wh_prior"]

        M_t = crop_affine(p_src, (T, T))
        tmpl = normalize_crop(warp_crop(src, M_t, (T, T)), rng)
        tmpl = np.stack([tmpl, box_mask(T, box)])

        lim = 0.9 * cfg.max_offset_mm / cfg.spacing_mm
        for _ in range(20):
            if not self.train:
                center, ang, sc = np.asarray(c["p_anat"], float), 0.0, 1.0
            else:
                ang = rng.uniform(-cfg.rot_deg, cfg.rot_deg)
                sc = 1 + rng.uniform(-cfg.scale_aug, cfg.scale_aug)
                if reverse or rng.random() < cfg.p_uniform_center:
                    center = np.asarray(p_dst, float) + rng.uniform(-lim, lim, 2)
                else:
                    center = np.asarray(c["p_anat"], float) + \
                        rng.normal(0, cfg.anat_jitter_mm / cfg.spacing_mm, 2)
            M_s = crop_affine(center, (S, S), scale=sc, angle_deg=ang)
            gt_grid = search_px_to_grid(cfg, apply_affine(invert_affine(M_s), p_dst))[0]
            inside = np.all(gt_grid >= 0) and np.all(gt_grid <= cfg.out_cells - 1)
            if inside or not self.train:
                break
        search = normalize_crop(warp_crop(dst, M_s, (S, S)), rng)[None]

        return {"template": torch.from_numpy(tmpl).float(),
                "search": torch.from_numpy(search).float(),
                "gt_grid": torch.from_numpy(gt_grid).float(),
                "M_grid": torch.from_numpy(compose(M_s, grid_affine(cfg))),
                "idx": idx % len(self.cases)}


# --------------------------------------------------------------------------- #
# Training / evaluation
# --------------------------------------------------------------------------- #
def make_optimizer(model, lr=3e-4, backbone_lr_mult=0.1, wd=1e-4):
    bb = list(model.encoder.parameters())
    bb_ids = {id(p) for p in bb}
    rest = [p for p in model.parameters() if id(p) not in bb_ids]
    return torch.optim.AdamW([{"params": bb, "lr": lr * backbone_lr_mult},
                              {"params": rest, "lr": lr}], weight_decay=wd)


def train_matcher(model, train_ds, val_ds, epochs=30, batch_size=16, lr=3e-4,
                  device="cuda", num_workers=8, log_every=50, val_fn=None):
    cfg = model.cfg
    model.to(device)
    dl = torch.utils.data.DataLoader(train_ds, batch_size, shuffle=True, drop_last=True,
                                     num_workers=num_workers, persistent_workers=num_workers > 0)
    opt = make_optimizer(model, lr)
    total = epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr * 0.1, lr], total_steps=total,
                                                pct_start=0.05)
    use_amp = device.startswith("cuda")
    scaler = torch.amp.GradScaler(enabled=use_amp)
    sigma = cfg.target_sigma_mm / cfg.cell_mm
    step, history = 0, []
    for ep in range(epochs):
        model.train()
        for b in dl:
            with torch.autocast(device_type=device.split(":")[0], enabled=use_amp):
                logits = model(b["template"].to(device), b["search"].to(device))
            loss = soft_target_ce(logits.float(), b["gt_grid"].to(device), sigma)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            history.append(loss.item())
            if step % log_every == 0:
                print(f"ep {ep} step {step}/{total} loss {np.mean(history[-log_every:]):.3f}")
            step += 1
        if val_fn is not None:
            val_fn(model, ep)
    return history


@torch.no_grad()
def score_maps(model, ds, device="cuda", batch_size=8):
    """Run the matcher on every case of `ds` (train=False: window on p_anat).
    Returns ScoreMaps whose score is log p(cell | appearance), ready for
    b5_prior_fusion.sweep_tau / predict (same fusion code as B5)."""
    model.eval().to(device)
    dl = torch.utils.data.DataLoader(ds, batch_size, shuffle=False)
    out = []
    for b in dl:
        logits = model(b["template"].to(device), b["search"].to(device)).float()
        lp = F.log_softmax(logits.flatten(1), -1).view_as(logits).cpu().numpy()
        for k in range(len(lp)):
            out.append(ScoreMap(score=lp[k].astype(np.float64), M_grid=b["M_grid"][k].numpy()))
    return out


# --------------------------------------------------------------------------- #
# Zero-shot variant of B4/B5 with pretrained features (no training)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def zero_shot_score_map(encoder: Encoder, cfg: MatcherConfig, img_cur, p_cur, box_wh,
                        img_prior, p_center, device="cpu") -> ScoreMap:
    """Cosine similarity between pretrained template features (central K x K)
    and every K x K window of the search features. Plug into B5 fusion."""
    T, S = cfg.template_px, cfg.search_px
    M_t, M_s = crop_affine(p_cur, (T, T)), crop_affine(p_center, (S, S))
    t = normalize_crop(warp_crop(img_cur, M_t, (T, T)))
    s = normalize_crop(warp_crop(img_prior, M_s, (S, S)))
    t = torch.from_numpy(np.stack([t, np.zeros_like(t)]))[None].float().to(device)
    s = torch.from_numpy(np.stack([s, np.zeros_like(s)]))[None].float().to(device)
    encoder.eval().to(device)
    zt, zs = encoder(t), encoder(s)
    c, h = cfg.c_t, (cfg.kernel_cells - 1) // 2
    k = zt[:, :, c - h:c + h + 1, c - h:c + h + 1]
    k = k - k.mean(1, keepdim=True)                          # center channels
    zs = zs - zs.mean(1, keepdim=True)
    num = F.conv2d(zs, k)                                     # (1,1,Ho,Wo)
    den = torch.sqrt(F.conv2d(zs ** 2, torch.ones_like(k)) * (k ** 2).sum() + 1e-8)
    cos = (num / den)[0, 0].cpu().numpy().astype(np.float64)
    return ScoreMap(score=cos, M_grid=compose(M_s, grid_affine(cfg)))
