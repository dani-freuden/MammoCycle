# Lesion matching: B5 (zero-shot + anatomical prior) and Step D (Siamese matcher)

| File | What it is |
|---|---|
| `geometry.py` | Resampling to a fixed mm/px, affine crops (crop px -> image px), local radial/tangential frame |
| `b5_prior_fusion.py` | Anatomical prior (Gaussian + outlier mixture, fitted by EM), NCC score maps, fusion `s/tau + log prior`, decoding, tau sweep, metrics, bootstrap CI |
| `stepD_matcher.py` | Siamese matcher (ResNet-18, depthwise xcorr, heatmap head), soft-target CE loss, pair sampler with jitter/augmentation, training loop, score maps for fusion, zero-shot deep-feature score map |
| `selfsup.py` | Self-supervised pairs from single images: random deformation (affine, elastic, lesion growth) + appearance changes, exact ground truth; `MixedDataset` for fine-tuning; `save_examples` contact sheet; `grid_error_mm` |
| `synthetic.py` | Toy pairs with known correspondence (unit tests; also the template for self-supervised pretraining on real single images) |
| `test_b5.py`, `test_stepD.py`, `test_selfsup.py` | End-to-end checks on toy data (run each with `python`) |

## Plugging in your data

Resample every image once to the working spacing (`geometry.resample_to_spacing`, then
`rescale_points` for your points) and cache it. Then build one dict per pair, all points in
working-spacing pixels as (x, y):

```python
case = {
    "img_cur": ..., "img_prior": ...,      # float32 arrays, chest wall on the same side
    "p_cur": (x, y), "box_wh": (w, h),      # current lesion center and box size
    "p_anat": (x, y),                       # from_anatomical(to_anatomical(p_cur, cur), prior)
    "centroid": (x, y),                     # prior breast-mask centroid
    "mask_prior": bool_array,               # optional
    "p_gt": (x, y),                         # prior GT center (train/val/test only)
    "box_wh_prior": (w, h),                 # optional: enables prior->current training samples
}
```

Split by patient, fit `AnatomicalPrior` on train, choose `tau` on val with `choose_tau`
(truncated mean, not median), and report on test once.

The toy numbers printed by the tests only show the code works; they say nothing about real
mammograms.

## Self-supervised pretraining from single images

```python
from selfsup import SingleImageSSLDataset, DeformConfig, MixedDataset, freeze, save_examples

def load(i):                        # from your cache, at the working spacing
    return {"img": img_i, "mask": breast_mask_i, "boxes": [(cx, cy, w, h), ...]}  # boxes optional

ssl = SingleImageSSLDataset(n_images, load, cfg, DeformConfig(), epoch_len=200_000,
                            box_sizes_mm=real_box_sizes_mm)
save_examples(ssl, "ssl_check.png")                     # look at it before training
ssl_val = freeze(SingleImageSSLDataset(n_images, load, cfg, fixed_seed=123), 500)

model = SiameseMatcher(cfg)
train_matcher(model, ssl, None, epochs=1, ...)          # stage 1: pretrain
ft = MixedDataset(PairDataset(train_cases, cfg, train=True), ssl, p_syn=0.25, epoch_len=20_000)
train_matcher(model, ft, None, epochs=10, lr=1e-4, ...) # stage 2: fine-tune on real pairs
```

Use only images of training patients for pretraining, so validation/test patients stay unseen.
