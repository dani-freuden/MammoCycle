"""Checks for self-supervised pretraining: exact ground truth, example sheet,
and a short pretrain -> zero-shot / few-label fine-tune comparison on toy data."""
import time
from pathlib import Path
import numpy as np
import torch
import cv2

from .b5_prior_fusion import AnatomicalPrior, local_residuals_mm, sweep_tau, print_table
from .geometry import apply_affine
from .stepD_matcher import MatcherConfig, SiameseMatcher, PairDataset, grid_affine, train_matcher, score_maps
from .selfsup import SingleImageSSLDataset, DeformConfig, MixedDataset, freeze, grid_error_mm, save_examples
from .synthetic import make_texture, add_blob, make_case

torch.set_num_threads(2)
rng = np.random.default_rng(0)

# ---- 1. ground truth is exact: plant a bright dot, find it in the deformed search crop ----
# Intensity normalization is bypassed here (it clips the dot), so only geometry is tested:
# affine + elastic + lesion growth/shrinkage, all on.
from . import selfsup
_norm = selfsup.normalize_crop
selfsup.normalize_crop = lambda x, rng=None: x
cfg = MatcherConfig(pretrained=False)
dot_cfg = DeformConfig(p_superpos=0, p_texture_swap=0, p_fade=0, noise_rel=(0, 0), p_sharpen=0, p_blur=0,
                       bias_field=0, p_real_box=1.0, p_lesion_scale=1.0, elastic_mm=4.0)
def dot_image(i):
    r = np.random.default_rng(1000 + i)
    img = cv2.GaussianBlur(r.standard_normal((1200, 1000)).astype(np.float32), (0, 0), 3)
    p = r.uniform(300, 700, 2)
    img = add_blob(img, p, 3.0, 50.0)
    return {"img": img, "boxes": [(p[0], p[1], 40, 40)]}
ds = SingleImageSSLDataset(20, dot_image, cfg, dot_cfg, fixed_seed=0)
def dot_centre(x):
    w = np.clip(x - 0.5 * x.max(), 0, None)              # pixels above half the dot's peak
    yy, xx = np.mgrid[0:x.shape[0], 0:x.shape[1]]
    return np.array([(w * xx).sum() / w.sum(), (w * yy).sum() / w.sum()])
errs, t_errs = [], []
for i in range(60):
    s = ds[i]
    gt = apply_affine(grid_affine(cfg), s["gt_grid"].numpy()[None])[0]
    errs.append(np.linalg.norm(dot_centre(s["search"][0].numpy()) - gt))
    t_errs.append(np.linalg.norm(dot_centre(s["template"][0].numpy()) - (cfg.template_px - 1) / 2))  # deformed too
selfsup.normalize_crop = _norm
print(f"GT check over 60 deformed samples: mean {np.mean(errs):.2f} px, max {np.max(errs):.2f} px; "
      f"template centre: max {np.max(t_errs):.2f} px")
assert np.max(errs) < 1.0 and np.max(t_errs) < 1.0

# ---- 2. toy "unlabeled images" and toy labeled pairs -------------------------------------
small = MatcherConfig(spacing_mm=0.4, template_px=128, search_px=256, kernel_cells=7,
                      feat_ch=64, use_layer3=False, pretrained=False, target_sigma_mm=3.0)
SC = tuple(v * 0.2 / 0.4 for v in (2, 6, 15, 40))
def toy_image(i):
    r = np.random.default_rng(i)
    img = make_texture(400, 330, r, scales_px=SC)
    boxes = []
    for _ in range(3):
        p = r.uniform(60, [270, 340]); sd = r.uniform(5, 10)
        img = add_blob(img, p, sd, r.uniform(1.0, 2.0)); boxes.append((p[0], p[1], 4 * sd, 4 * sd))
    return {"img": img, "boxes": boxes}
N_IMG = 60
ssl_train = SingleImageSSLDataset(N_IMG, toy_image, small, epoch_len=2400)
ssl_val = freeze(SingleImageSSLDataset(N_IMG, toy_image, small, fixed_seed=12345), 64)
save_examples(SingleImageSSLDataset(N_IMG, toy_image, small), str(Path(__file__).with_name("ssl_examples.png")), n=4)
t0 = time.time(); [ssl_train[i] for i in range(32)]
print(f"sample generation: {(time.time() - t0) / 32 * 1000:.0f} ms/sample (small config)")

mk = lambda n: [make_case(rng, spacing_mm=0.4, h=300, w=250, lesion_amp=2.0, superpos=0.5,
                          lesion_sd_mm=4.0) for _ in range(n)]
labeled, val_c = mk(20), mk(60)
prior = AnatomicalPrior.fit(local_residuals_mm(mk(80), small.spacing_mm))
val_ds = PairDataset(val_c, small, train=False)

def real_val(model, tag):
    for c, m in zip(val_c, score_maps(model, val_ds, device="cpu")):
        c["smap"] = m
    rows = [r for r in sweep_tau(val_c, prior, [0.5, 1, 2], small.spacing_mm)
            if r[0] != "prior mode (bias-corrected)"]
    print(f"\n[{tag}] toy real pairs, window on anatomical point")
    print_table(rows)

# ---- 3. pretrain on synthetic pairs from single images -----------------------------------
torch.manual_seed(0)
model = SiameseMatcher(small)
print(f"\nuntrained: synthetic-val median error {np.median(grid_error_mm(model, ssl_val, small)):.1f} mm")
t0 = time.time()
train_matcher(model, ssl_train, None, epochs=1, batch_size=16, lr=1e-3, device="cpu",
              num_workers=0, log_every=30)
print(f"pretrain time {time.time() - t0:.0f}s | synthetic-val median error "
      f"{np.median(grid_error_mm(model, ssl_val, small)):.1f} mm")
real_val(model, "SSL only, zero labels")
ssl_state = {k: v.clone() for k, v in model.state_dict().items()}

# ---- 4. few labels: scratch vs SSL init (same budget) ------------------------------------
real_train = PairDataset(labeled, small, train=True)
for name, init in [("scratch", None), ("SSL init", ssl_state)]:
    torch.manual_seed(1)
    m = SiameseMatcher(small)
    if init is not None:
        m.load_state_dict(init)
        ds_ft = MixedDataset(real_train, ssl_train, p_syn=0.25, epoch_len=480)
    else:
        ds_ft = PairDataset(labeled, small, train=True, epoch_len=480)
    train_matcher(m, ds_ft, None, epochs=2, batch_size=16, lr=1e-3, device="cpu",
                  num_workers=0, log_every=1000)
    real_val(m, f"{name} + fine-tune on 20 labeled pairs")
