"""Synthetic check of B5: fit prior on train, pick tau on val, report on test."""
import numpy as np
from .geometry import crop_affine, warp_crop, apply_affine, invert_affine, compose
from .b5_prior_fusion import (AnatomicalPrior, local_residuals_mm, dog_filter, ncc_score_map,
                             sweep_tau, choose_tau, print_table, predict, summarize, bootstrap_ci)
from .synthetic import make_case

SP = 0.2
rng = np.random.default_rng(0)

# --- geometry round trip -------------------------------------------------------
img = rng.standard_normal((300, 400)).astype(np.float32)
M = crop_affine((200.3, 150.7), (64, 48), scale=1.3, angle_deg=17)
pts = rng.uniform(0, 40, (5, 2))
back = apply_affine(compose(invert_affine(M), M), pts)
assert np.allclose(back, pts)
c = warp_crop(img, crop_affine((120, 80), (31, 31)), (31, 31))
assert np.isclose(c[15, 15], img[80, 120]), "crop center must sample the requested point"
print("geometry OK")

# --- cases ---------------------------------------------------------------------
def build(n):
    cs = []
    for _ in range(n):
        c = make_case(rng, spacing_mm=SP)
        c["img_cur_f"] = dog_filter(c["img_cur"], SP)
        c["img_prior_f"] = dog_filter(c["img_prior"], SP)
        cs.append(c)
    return cs

train, val, test = build(80), build(60), build(80)

prior = AnatomicalPrior.fit(local_residuals_mm(train, SP))
print("fitted prior:", prior.describe())
print("(simulated anatomical error (-3, 0) mm, sd (5, 9), 10% outliers -> residual bias should be ~(+3, 0))\n")

for cs in (val, test):
    for c in cs:
        c["smap"] = ncc_score_map(c["img_cur_f"], c["p_cur"], c["box_wh"],
                                  c["img_prior_f"], c["p_anat"], SP, radius_mm=40)

# sanity: score-map grid maps the gt to a high NCC cell
c = val[0]
g = apply_affine(invert_affine(c["smap"].M_grid), c["p_gt"])[0]
print("val[0] NCC at gt cell:", round(float(c["smap"].score[int(round(g[1])), int(round(g[0]))]), 3),
      "| map max:", round(float(c["smap"].score.max()), 3))

taus = np.logspace(-3, 0, 10)
rows = sweep_tau(val, prior, taus, SP)
print("\nVALIDATION")
print_table(rows)
best_tau = choose_tau(rows, "tmean30")
print("\nchosen tau (by truncated mean) =", best_tau,
      "| what selecting by median would pick:", choose_tau(rows, "median"))

def errs(cs, tau):
    return np.array([np.linalg.norm(predict(c["smap"], prior, c, SP, tau)[0] - c["p_gt"]) * SP for c in cs])

print("\nTEST (tau frozen from val)")
rows_t = []
for name, tau in [("prior only", np.inf), ("appearance only", None), (f"fused tau={best_tau:.3g}", best_tau)]:
    e = errs(test, tau)
    rows_t.append((name, summarize(e)))
e_anat = np.array([np.linalg.norm(c["p_anat"] - c["p_gt"]) * SP for c in test])
rows_t.insert(0, ("anatomical point", summarize(e_anat)))
print_table(rows_t)
e_f = errs(test, best_tau)
print("fused median 95% CI:", np.round(bootstrap_ci(e_f), 2), "| anatomical:", np.round(bootstrap_ci(e_anat), 2))
