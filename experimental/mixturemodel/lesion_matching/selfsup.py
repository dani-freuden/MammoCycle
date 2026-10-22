"""
Self-supervised pretraining for the Step D matcher from SINGLE images.

Idea: take one mammogram, pick a point p (a real annotated lesion or a random
point in the breast), cut the template around p from a mildly deformed copy
(p kept at its centre), and cut the search window from a randomly deformed and
re-rendered copy of the same image. Both sides are re-rendered independently,
so neither is the image as it is and they share no exact pixels. Because we
generated the deformation, we know exactly where p lands in the search window
-> a training pair with exact ground truth, for free, from every image you
have (no prior annotation needed).

The output dict has the same keys/shapes as stepD_matcher.PairDataset, so the
same model, loss and training loop are used unchanged.

What the deformation simulates (all magnitudes in DeformConfig):
  geometry   - rotation, anisotropic scale and shear (positioning/compression),
               smooth elastic field (local tissue rearrangement),
               local growth/shrinkage around p (lesion change)
  appearance - smooth multiplicative bias field (exposure / density change),
               superposition of other tissue (3D -> 2D projection changes),
               blur / sharpening (vendor processing), gamma/contrast,
               resolution changes, fresh independent noise, optional
               lesion fading
The network learns to be invariant to exactly these changes - nothing more.

Ground truth: the search crop is rendered by sampling the source image at
    x = L( A(u + d(u)) )          u = search-crop pixel
where d is the elastic field (crop px), A the affine crop map and L the local
radial scaling centered at p (L(p) = p). So p appears at the u* solving
u* + d(u*) = A^-1(p), found by fixed-point iteration (converges because the
field is smooth: |grad d| << 1).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

from .geometry import apply_affine, invert_affine, warp_crop, crop_affine
from .stepD_matcher import MatcherConfig, box_mask, normalize_crop, search_px_to_grid, grid_affine


@dataclass
class DeformConfig:
    # Calibrated on 2026-10-18 (calibrate_deform.py): NCC finds the point within 5 mm in 26 / 44 / 48% of these
    # pairs with a 14.4 mm / 1.5 x box / 51 mm template, against 24 / 40 / 42% on the real annotated pairs.
    # --- geometry ---
    rot_deg: float = 15.0
    scale_range: float = 0.20          # each axis independently in [1-s, 1+s] -> anisotropic
    shear: float = 0.05
    elastic_mm: float = 6.0            # sd of control-point displacement
    elastic_spacing_mm: float = 25.0   # control-point spacing (smoothness)
    max_strain: float = 0.2            # cap on local stretch |grad d|: keeps the map
                                       # invertible (no folding) and the warp realistic
    p_lesion_scale: float = 0.7
    lesion_scale: tuple = (0.6, 1.6)   # local size change at p (log-uniform)
    # --- appearance ---
    bias_field: float = 0.5            # max amplitude of log multiplicative field
    superpos: tuple = (0.2, 0.5)       # weight of blended-in other tissue (its breast only)
    p_superpos: float = 0.9
    p_texture_swap: float = 1.0        # replace this share of the fine detail (below texture_swap_mm) with
    texture_swap: tuple = (0.8, 1.0)   # another breast region's, keeping the coarse structure: overlap re-mixed
    texture_swap_mm: float = 4.0       # by another compression. What makes a small patch unreliable, as in real pairs
    p_blur: float = 0.5
    blur_px: tuple = (0.5, 1.5)
    p_sharpen: float = 0.3
    noise_rel: tuple = (0.02, 0.15)    # fresh noise, relative to crop std (both sides)
    p_resample: float = 0.5            # resolution jitter, each side on its own: down by a factor
    resample: tuple = (1.0, 2.0)       # in this range, then back up
    p_fade: float = 0.3
    fade: tuple = (0.2, 0.7)           # fraction of lesion contrast removed
    # --- the template side, so that neither crop is the image as it is ---
    template_deform: float = 0.5       # the template gets this fraction of the search's geometric
                                       # deformation, with the lesion kept at its centre
    # --- where to put the "lesion" ---
    p_real_box: float = 0.5            # use a real annotated box when the image has one
    texture_bias: float = 0.5          # prob. of preferring a textured point over a uniform one
    edge_margin_mm: float = 5.0        # keep sampled points away from the skin line
    min_breast_frac: float = 0.25      # reject search windows mostly outside the breast


class SingleImageSSLDataset(torch.utils.data.Dataset):
    """
    load_fn(i) -> dict with
        "img":   float32 image at the working spacing (MatcherConfig.spacing_mm),
                 chest wall on the same side for all images
        "mask":  optional bool breast mask (default: whole image)
        "boxes": optional list of (cx, cy, w, h) real annotations, working px
    box_sizes_mm: optional array of real box sizes (max side, mm) to sample
                  box sizes for random points; default log-normal, median 12 mm.
    fixed_seed:   if set, sample i is deterministic (use for a frozen
                  synthetic validation set).
    """

    def __init__(self, n_images, load_fn, mcfg: MatcherConfig, dcfg: DeformConfig = None,
                 epoch_len: int = 50_000, box_sizes_mm=None, fixed_seed: int | None = None):
        self.n, self.load_fn, self.m, self.d = n_images, load_fn, mcfg, dcfg or DeformConfig()
        self.epoch_len, self.fixed_seed = epoch_len, fixed_seed
        self.box_sizes_mm = None if box_sizes_mm is None else np.asarray(box_sizes_mm, float)

    def __len__(self):
        return self.epoch_len

    def __getitem__(self, idx):
        rng = np.random.default_rng(None if self.fixed_seed is None else self.fixed_seed + idx)
        for _ in range(20):
            rec = self.load_fn(int(rng.integers(self.n)))
            out = self.make(rec, rng)
            if out is not None:
                return out
        raise RuntimeError("could not make a valid sample in 20 tries")

    # ------------------------------------------------------------------ #
    def sample_point(self, img, mask, rec, rng):
        sp = self.m.spacing_mm
        boxes = rec.get("boxes") or []
        if boxes and rng.random() < self.d.p_real_box:
            cx, cy, w, h = boxes[int(rng.integers(len(boxes)))]
            return np.array([cx, cy], float), np.array([w, h], float)
        ys, xs = np.nonzero(self._inner(mask))
        if len(xs) == 0:
            return None, None
        pick = rng.integers(len(xs), size=4)
        cands = np.stack([xs[pick], ys[pick]], 1).astype(float)
        if rng.random() < self.d.texture_bias:          # prefer structure over flat fat
            half = int(round(5.0 / sp))
            stds = [img[max(0, int(y) - half):int(y) + half, max(0, int(x) - half):int(x) + half].std()
                    for x, y in cands]
            p = cands[int(np.argmax(stds))]
        else:
            p = cands[0]
        if self.box_sizes_mm is not None and len(self.box_sizes_mm):
            side_mm = float(rng.choice(self.box_sizes_mm))
        else:
            side_mm = float(np.clip(np.exp(rng.normal(np.log(12.0), 0.5)), 5, 60))
        ar = np.exp(rng.normal(0, 0.2))
        return p, np.array([side_mm * ar, side_mm / ar]) / sp

    def make(self, rec, rng):
        m, d = self.m, self.d
        img = rec["img"].astype(np.float32)
        mask = rec.get("mask")
        if mask is None:
            mask = np.ones(img.shape, bool)
        p, box = self.sample_point(img, mask, rec, rng)
        if p is None:
            return None
        T, S = m.template_px, m.search_px

        # ---------------- template: deformed too, the lesion kept at the centre ----
        A_t, disp_t = self.random_warp(T, rng, d.template_deform)
        c_t = np.full(2, (T - 1) / 2)
        disp_t -= _bilinear(disp_t, c_t)                    # no displacement at the centre
        tmpl = _remap(img, _sampling_map(A_t, disp_t, p - A_t @ c_t))
        tmpl = self.render_template(tmpl, rng)

        # ---------------- search geometry ---------------------------------
        A, disp = self.random_warp(S, rng)                  # A: image px per crop px
        uc = (S - 1) / 2
        lim = 0.85 * m.max_offset_mm / m.spacing_mm         # GT anywhere on the grid
        target = uc + rng.uniform(-lim, lim, 2)             # where p should land (pre-elastic)
        t = p - A @ target
        M = np.hstack([A, t[:, None]])                      # crop px -> image px

        q = apply_affine(invert_affine(M), p)[0]
        u = q.copy()
        for _ in range(50):
            u = q - _bilinear(disp, u)
        if np.linalg.norm(u + _bilinear(disp, u) - q) > 0.05:
            return None                                      # did not converge
        gt_grid = search_px_to_grid(m, u)[0]
        if np.any(gt_grid < 0) or np.any(gt_grid > m.out_cells - 1):
            return None

        X = _sampling_map(A, disp, t)                                   # (S,S,2) image px
        if rng.random() < d.p_lesion_scale:                             # lesion grows/shrinks
            k = np.exp(rng.uniform(*np.log(d.lesion_scale)))
            rho = max(box) / 2
            Y = X - p.astype(np.float32)
            s = 1 + (k - 1) * np.exp(-(Y ** 2).sum(-1) / (2 * rho ** 2))
            X = p.astype(np.float32) + Y * s[..., None]

        search = _remap(img, X)
        inside = _remap(mask.astype(np.uint8), X, cv2.INTER_NEAREST)
        if inside.mean() < d.min_breast_frac:
            return None
        search = self.render_search(search, img, mask, inside, u, box, rng)

        return {"template": torch.from_numpy(np.stack([tmpl, box_mask(T, box)])).float(),
                "search": torch.from_numpy(search[None]).float(),
                "gt_grid": torch.from_numpy(gt_grid).float(),
                "M_grid": torch.zeros(2, 3, dtype=torch.float64),   # not an affine crop
                "idx": -1}

    def random_warp(self, S, rng, strength=1.0):
        """A random affine A (image px per crop px) and a smooth elastic field (S, S, 2) in crop px, with
        strength x the geometric magnitudes of DeformConfig."""
        d, m = self.d, self.m
        th = np.deg2rad(rng.uniform(-d.rot_deg, d.rot_deg) * strength)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        Sc = np.diag(1 + rng.uniform(-d.scale_range, d.scale_range, 2) * strength)
        Sh = np.array([[1, rng.uniform(-d.shear, d.shear) * strength], [0, 1]])

        disp = strength * d.elastic_mm / m.spacing_mm * self._smooth_field(S, d.elastic_spacing_mm, rng)
        J = np.stack(np.gradient(disp, axis=(0, 1)), -1)                   # (S,S,2,2) partials
        strain = np.sqrt((J ** 2).sum((-1, -2))).max()
        if strain > d.max_strain:
            disp *= d.max_strain / strain
        return R @ Sc @ Sh, disp

    def _smooth_field(self, S, spacing_mm, rng):
        """A smooth random displacement field (S, S, 2): standard normal control points spacing_mm apart,
        cubic-interpolated."""
        g = max(2, int(round(S * self.m.spacing_mm / spacing_mm))) + 1
        coarse = rng.normal(0, 1, (g, g, 2)).astype(np.float32)
        return cv2.resize(coarse, (S, S), interpolation=cv2.INTER_CUBIC)

    # ---------------- appearance ----------------------------------------
    def _blur_sharpen_noise(self, x, rng):
        d = self.d
        if rng.random() < d.p_resample:                     # coarser detector / processing
            h, w = x.shape
            f = rng.uniform(*d.resample)
            small = cv2.resize(x, (max(1, round(w / f)), max(1, round(h / f))), interpolation=cv2.INTER_AREA)
            x = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        if rng.random() < d.p_blur:
            x = cv2.GaussianBlur(x, (0, 0), rng.uniform(*d.blur_px))
        elif rng.random() < d.p_sharpen:
            b = cv2.GaussianBlur(x, (0, 0), 2.0)
            x = x + rng.uniform(0.3, 1.0) * (x - b)
        x = x + rng.uniform(*d.noise_rel) * (x.std() + 1e-6) * \
            rng.standard_normal(x.shape).astype(np.float32)
        return x

    def render_template(self, x, rng):
        # light, independent processing so neither side keeps the exact pixel noise
        return normalize_crop(self._blur_sharpen_noise(x, rng), rng)

    def render_search(self, x, img, mask, inside, u, box, rng):
        d, S = self.d, x.shape[0]
        if rng.random() < d.p_fade:                         # lesion less visible in the prior
            rho = max(2.0, max(box) / 2)
            yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
            w = np.exp(-((xx - u[0]) ** 2 + (yy - u[1]) ** 2) / (2 * rho ** 2))
            bg = cv2.GaussianBlur(x, (0, 0), rho)
            x = x - rng.uniform(*d.fade) * w * (x - bg)
        if d.bias_field > 0:                                # exposure / density drift
            c = rng.normal(0, 1, (4, 4)).astype(np.float32)
            f = cv2.resize(c, (S, S), interpolation=cv2.INTER_CUBIC)
            x = x * np.exp(rng.uniform(0, d.bias_field) * f / (np.abs(f).max() + 1e-6))
        if rng.random() < d.p_superpos:                     # different tissue overlap
            other, w = self._other_region(img, mask, inside, S, rng)
            w = w * rng.uniform(*d.superpos)
            x = (1 - w) * x + w * other
        if rng.random() < d.p_texture_swap:                 # fine detail re-mixed, coarse structure kept
            other, w = self._other_region(img, mask, inside, S, rng)
            fine = lambda z: z - cv2.GaussianBlur(z, (0, 0), d.texture_swap_mm / self.m.spacing_mm)
            x = x + w * rng.uniform(*d.texture_swap) * (fine(other) - fine(x))
        return normalize_crop(self._blur_sharpen_noise(x, rng), rng)

    def _other_region(self, img, mask, inside, S, rng):
        """An S x S crop of another breast region (a random point, at a random angle), and the weight (S, S) it may
        be blended in with: breast over breast only, so neither crop's skin line, background or image edge comes
        along. The window's own border is not an edge (the breast goes on past it), so it does not fade there."""
        ys, xs = np.nonzero(mask)
        j = int(rng.integers(len(xs)))
        M = crop_affine((xs[j], ys[j]), (S, S), angle_deg=rng.uniform(0, 360))
        weight = warp_crop(self._soft(mask, edge_outside=True), M, (S, S)) * self._soft(inside, edge_outside=False)
        return warp_crop(img, M, (S, S)), weight

    def _soft(self, mask, edge_outside):
        """_inner blurred by half of edge_margin_mm: about 0 at the skin line, 1 well inside the breast."""
        sigma = self.d.edge_margin_mm / self.m.spacing_mm / 2
        border = cv2.BORDER_CONSTANT if edge_outside else cv2.BORDER_REFLECT_101
        return cv2.GaussianBlur(self._inner(mask, edge_outside).astype(np.float32), (0, 0), sigma, borderType=border)

    def _inner(self, mask, edge_outside=False):
        """The breast mask eroded by edge_margin_mm: away from the skin line, and with edge_outside from the
        image edge too (the chest wall), which the erosion otherwise treats as breast."""
        r = max(1, int(round(self.d.edge_margin_mm / self.m.spacing_mm)))
        return cv2.erode(mask.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8),
                         borderType=cv2.BORDER_CONSTANT, borderValue=0 if edge_outside else 1) > 0


def _sampling_map(A, disp, t):
    """The image point (x, y) each crop pixel u samples, A (u + d(u)) + t: (S, S, 2)."""
    S = disp.shape[0]
    uu, vv = np.meshgrid(np.arange(S, dtype=np.float32), np.arange(S, dtype=np.float32))
    V = np.stack([uu + disp[..., 0], vv + disp[..., 1]], -1)
    return V @ A.T.astype(np.float32) + t.astype(np.float32)


def _remap(img, X, interpolation=cv2.INTER_LINEAR):
    """img sampled at the points X (S, S, 2), zero outside it."""
    return cv2.remap(img, np.ascontiguousarray(X[..., 0], dtype=np.float32),
                     np.ascontiguousarray(X[..., 1], dtype=np.float32), interpolation,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _bilinear(field, p):
    x, y = float(p[0]), float(p[1])
    h, w = field.shape[:2]
    x0 = int(np.clip(np.floor(x), 0, w - 2)); y0 = int(np.clip(np.floor(y), 0, h - 2))
    fx, fy = x - x0, y - y0
    return ((1 - fx) * (1 - fy) * field[y0, x0] + fx * (1 - fy) * field[y0, x0 + 1]
            + (1 - fx) * fy * field[y0 + 1, x0] + fx * fy * field[y0 + 1, x0 + 1])


# --------------------------------------------------------------------------- #
# Helpers: mixing for fine-tuning, frozen validation, error in grid space
# --------------------------------------------------------------------------- #
class MixedDataset(torch.utils.data.Dataset):
    """Fine-tuning: each item is a real labeled pair with prob 1 - p_syn,
    otherwise a synthetic one (keeps the pretrained invariances, regularizes
    when real labels are few)."""

    def __init__(self, real_ds, syn_ds, p_syn=0.25, epoch_len=None):
        self.real, self.syn, self.p = real_ds, syn_ds, p_syn
        self.epoch_len = epoch_len or len(real_ds)

    def __len__(self):
        return self.epoch_len

    def __getitem__(self, idx):
        if np.random.default_rng().random() < self.p:
            return self.syn[idx]
        return self.real[int(np.random.default_rng().integers(len(self.real)))]


def freeze(ds, n):
    """Materialize n samples (use with fixed_seed) -> list usable as a Dataset."""
    return [ds[i] for i in range(n)]


@torch.no_grad()
def grid_error_mm(model, samples, cfg: MatcherConfig, device="cpu", batch_size=16):
    """Argmax error in mm on samples that carry gt_grid (synthetic or real)."""
    model.eval().to(device)
    dl = torch.utils.data.DataLoader(samples, batch_size, shuffle=False)
    errs = []
    for b in dl:
        lg = model(b["template"].to(device), b["search"].to(device)).cpu()
        B, H, W = lg.shape
        k = lg.view(B, -1).argmax(1)
        pred = torch.stack([k % W, k // W], 1).float()
        errs.append(((pred - b["gt_grid"]).norm(dim=1) * cfg.cell_mm).numpy())
    return np.concatenate(errs)


def save_examples(ds, path, n=6, seed=0):
    """Contact sheet: template | box mask | search with the GT cross. Look at
    this before training: if you cannot find the lesion by eye, neither can
    the network; if it is trivially identical, the task is too easy."""
    m = ds.m
    tiles = []
    old = ds.fixed_seed
    ds.fixed_seed = seed
    for i in range(n):
        s = ds[i]
        t = _to_u8(s["template"][0].numpy()); mk = (s["template"][1].numpy() * 255).astype(np.uint8)
        sr = _to_u8(s["search"][0].numpy())
        gx, gy = apply_affine(grid_affine(m), s["gt_grid"].numpy()[None])[0]
        sr = cv2.cvtColor(sr, cv2.COLOR_GRAY2BGR)
        cv2.drawMarker(sr, (int(round(gx)), int(round(gy))), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
        S = sr.shape[0]
        left = np.vstack([cv2.cvtColor(t, cv2.COLOR_GRAY2BGR), cv2.cvtColor(mk, cv2.COLOR_GRAY2BGR)])
        left = cv2.resize(left, (S // 2, S)) if left.shape[0] != S else left
        tiles.append(np.hstack([left, np.full((S, 6, 3), 255, np.uint8), sr]))
    ds.fixed_seed = old
    cv2.imwrite(path, np.vstack([np.vstack([t, np.full((6, t.shape[1], 3), 255, np.uint8)]) for t in tiles]))


def _to_u8(x):
    lo, hi = np.percentile(x, [1, 99])
    return (np.clip((x - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8)
