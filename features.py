# features.py  — minimal strong baseline
import numpy as np
import cv2 as cv

# small helpers
def _mean_std(arr):
    arr = np.asarray(arr).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 0.0
    return float(np.mean(arr)), float(np.std(arr))

def _percentile(arr, p):
    arr = np.asarray(arr).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, p))

def _lbp_entropy_np(lbp, region=None):
    vals = lbp[region>0].ravel() if region is not None else lbp.ravel()
    if vals.size == 0:
        return 0.0
    hist, _ = np.histogram(vals, bins=256, range=(0,256), density=True)
    hist = hist + 1e-12
    return float(-np.sum(hist * np.log(hist)))

def _edge_overlap_fraction(edges, boundary_mask):
    e = (edges > 0)
    b = (boundary_mask > 0)
    return float((e & b).sum()/b.sum()) if b.sum() else 0.0

def _component_stats_min(mask):
    #Return only: num_shadow_components, comp_area_p90, comp_peri2_over_area_p50
    num, labels, stats, _ = cv.connectedComponentsWithStats(mask, connectivity=8)
    areas, peri2_over_area = [], []

    for i in range(1, num):
        a = stats[i, cv.CC_STAT_AREA]
        areas.append(a)
        comp = (labels == i).astype(np.uint8) * 255
        cnts, _ = cv.findContours(comp, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if cnts and a > 0:
            peri = cv.arcLength(max(cnts, key=cv.contourArea), True)
            peri2_over_area.append((peri * peri) / a)

    out = {"num_shadow_components": int(num - 1)}
    out["comp_area_p90"] = _percentile(areas, 90)
    out["comp_peri2_over_area_p50"] = _percentile(peri2_over_area, 50)
    return out

def _angular_variance_min(gx, gy, boundary_mask, eps=1e-6):
    #Return only: normal_mrl (alignment) and normal_angle_std (irregularity)
    by, bx = np.where(boundary_mask > 0)
    if by.size == 0:
        return {"normal_angle_std": 0.0, "normal_mrl": 0.0}
    gxs = gx[by, bx].astype(np.float32)
    gys = gy[by, bx].astype(np.float32)
    mag = np.sqrt(gxs * gxs + gys * gys) + eps
    nx, ny = gxs / mag, gys / mag
    ang = np.arctan2(ny, nx)
    C = float(np.mean(np.cos(ang)))
    S = float(np.mean(np.sin(ang)))
    R = float(np.sqrt(C * C + S * S))     # mean resultant length [0..1]
    ang_std = float(np.sqrt(-2 * np.log(R + eps)))
    return {"normal_angle_std": ang_std, "normal_mrl": R}

# main: minimal strong baseline
def features_from_products(mask, edges, lbp, L8, gx, gy,
                           d_list, c_list, s_arr, d_lo, d_hi):

    # build a compact, high-signal feature vector
    # expected inputs: intermediates from texture.py and per-boundary sample arrays
    
    # boundary and masks
    boundary = cv.morphologyEx(
        mask, cv.MORPH_GRADIENT, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    )
    in_m = mask > 0
    out_m = ~in_m

    feats = {}

    # coverage / edges 
    feats["mask_frac"] = float(in_m.mean())
    feats["edge_density_all"] = float((edges > 0).mean())
    feats["edge_density_shadow"] = float((edges[in_m] > 0).mean()) if in_m.any() else 0.0
    feats["edge_on_boundary_frac"] = _edge_overlap_fraction(edges, boundary)

    # luminance contrast (means/stds only)
    mu_in, sd_in = _mean_std(L8[in_m] if in_m.any() else np.array([0.0]))
    mu_out, sd_out = _mean_std(L8[out_m] if out_m.any() else np.array([0.0]))
    feats["L_in_mean"] = mu_in
    feats["L_in_std"]  = sd_in
    feats["L_out_mean"] = mu_out
    feats["L_out_std"]  = sd_out
    feats["contrast_shadow_vs_non"] = (mu_out - mu_in) / (mu_out + 1e-6)

    # texture entropy (LBP)
    feats["lbp_entropy_in"]  = _lbp_entropy_np(lbp, region=mask)
    feats["lbp_entropy_out"] = _lbp_entropy_np(lbp, region=(255 - mask))

    # shadow component shape (minimal)
    feats |= _component_stats_min(mask)

    # boundary geometry consistency
    feats |= _angular_variance_min(gx, gy, boundary)

    # boundary sampling aggregates (compact set)
    d_arr = np.asarray(d_list, dtype=np.float32)
    s_arr = np.asarray(s_arr, dtype=np.float32)

    feats["valid_boundary_samples"] = int(len(d_arr))

    # texture chi2 across boundary: mean + high percentile
    feats["tex_chi2_d_mean"] = float(d_arr.mean()) if d_arr.size else 0.0
    feats["tex_chi2_d_p85"]  = _percentile(d_arr, 85) if d_arr.size else 0.0
    feats["share_high_texture_jump"] = float(np.mean(d_arr > d_hi)) if d_arr.size else 0.0

    # tamper score per-sample: mean + high percentile
    feats["tamper_sample_s_mean"] = float(s_arr.mean()) if s_arr.size else 0.0
    feats["tamper_sample_s_p90"]  = _percentile(s_arr, 90) if s_arr.size else 0.0
    feats["share_high_tamper"]    = float(np.mean(s_arr > 0.7)) if s_arr.size else 0.0

    # keep these if you want to debug thresholds later; they won't hurt
    feats["d_lo"] = float(d_lo)
    feats["d_hi"] = float(d_hi)

    return feats
