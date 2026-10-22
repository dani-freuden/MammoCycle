"""Checks for Step D: grid geometry, shapes, and a short CPU training run on toy pairs."""
import time
import numpy as np
import torch
import torch.nn.functional as F
import cv2

from .geometry import crop_affine, warp_crop, compose
from .b5_prior_fusion import AnatomicalPrior, local_residuals_mm, decode, sweep_tau, print_table
from .stepD_matcher import (MatcherConfig, SiameseMatcher, PairDataset, grid_affine, dw_xcorr,
                           train_matcher, score_maps)
from .synthetic import make_case

torch.set_num_threads(2)
rng = np.random.default_rng(1)

# ---- 1. geometry: correlate stride-8 pooled features and map the peak back ----------------
cfg = MatcherConfig(pretrained=False)
img = cv2.GaussianBlur(rng.standard_normal((900, 900)).astype(np.float32), (0, 0), 6)
p = np.array([450.3, 449.7])
T, S, K = cfg.template_px, cfg.search_px, cfg.kernel_cells
errs = []
for _ in range(10):
    off = rng.uniform(-200, 200, 2)
    M_s = crop_affine(p + off, (S, S))
    t = torch.from_numpy(warp_crop(img, crop_affine(p, (T, T)), (T, T)))[None, None]
    s = torch.from_numpy(warp_crop(img, M_s, (S, S)))[None, None]
    ft, fs = F.avg_pool2d(t, 8), F.avg_pool2d(s, 8)          # a stand-in stride-8 "encoder"
    c, h = cfg.c_t, (K - 1) // 2
    k = ft[:, :, c - h:c + h + 1, c - h:c + h + 1]
    r = dw_xcorr(fs - fs.mean(), k - k.mean())[0, 0].numpy().astype(np.float64)
    assert r.shape == (cfg.out_cells, cfg.out_cells)
    pt, _ = decode(r / 1e-3, compose(M_s, grid_affine(cfg)))
    errs.append(np.linalg.norm(pt - p))
print(f"geometry: peak->image error over 10 random windows: max {max(errs):.1f} px "
      f"(cell = {cfg.stride} px)")
assert max(errs) < cfg.stride, "grid mapping is off"

# ---- 2. shapes with the real network ---------------------------------------------------------
net = SiameseMatcher(cfg)
with torch.no_grad():
    out = net(torch.randn(2, 2, T, T), torch.randn(2, 1, S, S))
print("full-size logits:", tuple(out.shape), f"| cells of {cfg.cell_mm} mm, range +-{cfg.max_offset_mm:.1f} mm")
print("params (M):", round(sum(p.numel() for p in net.parameters()) / 1e6, 2))

# ---- 3. short training run on toy pairs (small config so it fits on 2 CPU cores) ----------
small = MatcherConfig(spacing_mm=0.4, template_px=128, search_px=256, kernel_cells=7,
                      feat_ch=64, use_layer3=False, pretrained=False, target_sigma_mm=3.0)
print(f"\nsmall config: grid {small.out_cells}x{small.out_cells}, cell {small.cell_mm} mm, "
      f"range +-{small.max_offset_mm:.1f} mm")
mk = lambda n: [make_case(rng, spacing_mm=0.4, h=300, w=250, lesion_amp=2.0, superpos=0.5,
                          lesion_sd_mm=4.0) for _ in range(n)]
train_c, val_c = mk(120), mk(60)
for c in train_c:
    c["box_wh_prior"] = c["box_wh"]          # toy: same size; enables reverse-direction samples
prior = AnatomicalPrior.fit(local_residuals_mm(train_c, small.spacing_mm))

model = SiameseMatcher(small)
val_ds = PairDataset(val_c, small, train=False)

def report(tag):
    maps = score_maps(model, val_ds, device="cpu")
    for c, m in zip(val_c, maps):
        c["smap"] = m
    rows = sweep_tau(val_c, prior, [0.5, 1, 2, 4], small.spacing_mm)
    print(f"\n[{tag}] validation (window centered on anatomical point)")
    print_table(rows)

report("untrained")
train_ds = PairDataset(train_c, small, train=True, epoch_len=480)
t0 = time.time()
hist = train_matcher(model, train_ds, val_ds, epochs=4, batch_size=16, lr=1e-3,
                     device="cpu", num_workers=0, log_every=15)
print(f"train time {time.time() - t0:.0f}s | loss first 15 steps {np.mean(hist[:15]):.3f} "
      f"-> last 15 steps {np.mean(hist[-15:]):.3f} (uniform guess = {np.log(small.out_cells**2):.3f})")
report("after short training")
