# lighting_features.py for lighting feature extraction
import numpy as np
import cv2 as cv

# helpers (local only)
def srgb_to_linear(x8):
    x = x8.astype(np.float32) / 255.0
    a = 0.055
    return np.where(x <= 0.04045, x/12.92, ((x + a)/(1 + a))**2.4).astype(np.float32)

def get_brightness(img_bgr):
    # linear luminance (Y) from BGR
    lin = srgb_to_linear(img_bgr)
    B, G, R = cv.split(lin)
    return (0.114*B + 0.587*G + 0.299*R).astype(np.float32)

def get_rgb_direction(img_bgr):
    lin = srgb_to_linear(img_bgr)
    n = np.linalg.norm(lin, axis=2, keepdims=True) + 1e-6
    return lin / n

def calc_skew(values):
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return 0.0
    m = v.mean()
    s = v.std(ddof=1)
    if s < 1e-12:
        return 0.0
    z = (v - m) / s
    return float(np.mean(z**3))

def calc_kurtosis(values):
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return -3.0
    m = v.mean()
    s = v.std(ddof=1)
    if s < 1e-12:
        return -3.0
    z = (v - m) / s
    return float(np.mean(z**4)) - 3.0  # excess kurtosis

def sobel_xy(gray32):
    gx = cv.Sobel(gray32, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(gray32, cv.CV_32F, 0, 1, ksize=3)
    return gx, gy

def boundary_from_mask(mask):
    se = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
    return cv.morphologyEx(mask, cv.MORPH_GRADIENT, se)

# umbra (dark part) and ring (bright area) selection
def find_dark_part(shadow_mask, brightness, keep_fraction=0.30, shrink=2):
    mask = shadow_mask.astype(bool)
    if shrink > 0:
        kernel = np.ones((3,3), np.uint8)
        mask = cv.erode(shadow_mask, kernel, iterations=shrink).astype(bool)
    Ys = brightness[mask]
    if Ys.size == 0:
        return None
    thr = np.quantile(Ys, keep_fraction)
    umbra = mask & (brightness <= thr)
    if np.count_nonzero(umbra) < 50:
        return None
    return umbra

def find_bright_nearby(img_bgr, shadow_mask, brightness, expand=4, color_diff=0.08):
    kernel = np.ones((3,3), np.uint8)
    expanded = cv.dilate((shadow_mask>0).astype(np.uint8), kernel, iterations=expand).astype(bool)
    ring = expanded & (~(shadow_mask>0)) & (brightness > 0.001)
    if ring.sum() == 0:
        return None

    color_dirs = get_rgb_direction(img_bgr)
    edge = (shadow_mask>0) & (~cv.erode((shadow_mask>0).astype(np.uint8), kernel, 1).astype(bool))
    if edge.sum() == 0:
        return ring

    avg_color = color_dirs[edge].mean(axis=0, keepdims=True)
    ring_idx = np.flatnonzero(ring.ravel())
    ring_cols = color_dirs.reshape(-1,3)[ring_idx]

    # cosine distance
    sim = 1.0 - (ring_cols @ avg_color.T).ravel()
    keep = np.zeros(ring.size, bool)
    keep[ring_idx[sim <= color_diff]] = True
    res = keep.reshape(ring.shape)
    return res if res.sum() > 0 else ring

# call
def extract(
    img_bgr,
    *,
    mask,            
    L=None,                 
    gx=None, gy=None,       
    min_area=1200,
    expand=6,
    color_diff=0.08,
    darkest_frac=0.30,
    shrink=2
) -> dict:
    """
    lighting-consistency features;
    return a dict of global and aggregated per-shadow metrics
    """
    feats = {}

    # sanitize mask
    mask = (mask > 0).astype(np.uint8) * 255
    num_labels, labels = cv.connectedComponents(mask)
    feats["num_components_total"] = int(num_labels - 1)

    # brightness/gradients
    if L is None:
        L = get_brightness(img_bgr)  # float32, ~[0..1] linear Y
    if gx is None or gy is None:
        gx, gy = _sobel_xy(L)

    # scene-level context
    Lv = L[np.isfinite(L)]
    feats["image_median_brightness"] = float(np.median(Lv)) if Lv.size else 0.0
    feats["image_brightness_iqr"] = float(np.percentile(Lv,75)-np.percentile(Lv,25)) if Lv.size else 0.0

    # per-shadow containers
    per_sr4 = []          # [median, IQR, skew, kurt] of strength ratios
    Yn_list = []
    umbra_frac_list = []
    ring_to_area_list = []
    area_list = []
    mrl_list = []
    angstd_list = []
    usable = 0

    # iterate components
    for i in range(1, num_labels):
        comp = (labels == i)
        area = int(np.count_nonzero(comp))
        if area < min_area:
            continue

        # boundary alignment (Sobel on L)
        boundary = boundary_from_mask(comp.astype(np.uint8)*255)
        ys, xs = np.where(boundary > 0)
        if ys.size > 0:
            gxs, gys = gx[ys, xs].astype(np.float32), gy[ys, xs].astype(np.float32)
            mag = np.sqrt(gxs*gxs + gys*gys) + 1e-6
            nx, ny = gxs/mag, gys/mag
            ang = np.arctan2(ny, nx)
            C, S = float(np.mean(np.cos(ang))), float(np.mean(np.sin(ang)))
            R = float(np.sqrt(C*C + S*S))
            angle_std = float(np.sqrt(max(0.0, -2*np.log(R + 1e-6))))
        else:
            R, angle_std = 0.0, 0.0
        mrl_list.append(R)
        angstd_list.append(angle_std)
        area_list.append(area)

        # umbra (darkest fraction) and bright ring
        umbra = find_dark_part(comp.astype(np.uint8), L, keep_fraction=darkest_frac, shrink=shrink)
        if umbra is None:
            continue
        ring = find_bright_nearby(img_bgr, comp.astype(np.uint8), L, expand=expand, color_diff=color_diff)
        if ring is None or ring.sum() < 30:
            continue

        Ys = L[umbra]
        Ys = Ys[np.isfinite(Ys)]
        if Ys.size < 50:
            continue
        Yn = float(np.median(L[ring]))
        if not np.isfinite(Yn) or Yn < 1e-4:
            continue

        # coverage/quality
        umbra_frac_list.append(float(np.count_nonzero(umbra)) / (area + 1e-6))
        ring_to_area_list.append(float(np.count_nonzero(ring)) / (area + 1e-6))
        Yn_list.append(Yn)

        # strength ratio stats (robust)
        sr = Ys / (Yn + 1e-6)
        sr_med = float(np.median(sr))
        sr_iqr = float(np.percentile(sr,75) - np.percentile(sr,25))
        sr_skew = float(calc_skew(sr))
        sr_kurt = float(calc_kurtosis(sr))
        per_sr4.append([sr_med, sr_iqr, sr_skew, sr_kurt])

        usable += 1

    feats["num_components_usable"] = int(usable)
    feats["usable_frac"] = float(usable / max(1, feats["num_components_total"]))

    # aggregate helpers
    def agg(name, arr):
        arr = np.asarray(arr, dtype=np.float32)
        feats[f"{name}_median"] = float(np.median(arr)) if arr.size else 0.0
        feats[f"{name}_iqr"]    = float(np.percentile(arr,75)-np.percentile(arr,25)) if arr.size else 0.0

    agg("comp_area", area_list)
    agg("boundary_mrl", mrl_list)
    agg("boundary_angle_std", angstd_list)
    agg("umbra_frac", umbra_frac_list)
    agg("ring_to_area", ring_to_area_list)
    agg("Yn", Yn_list)

    # cross-shadow consistency (pairwise in robust Z-space) + tamper score
    sr4 = np.asarray(per_sr4, dtype=np.float32)
    if sr4.shape[0] >= 2:
        med = np.median(sr4, axis=0)
        mad = np.median(np.abs(sr4 - med), axis=0)
        scale = np.where(mad > 1e-8, mad, 1.0)
        Z = (sr4 - med) / scale

        dists = []
        for a in range(Z.shape[0]):
            for b in range(a+1, Z.shape[0]):
                dists.append(float(np.linalg.norm(Z[a]-Z[b])))
        dists = np.asarray(dists, dtype=np.float32)

        feats["sr_dist_median"] = float(np.median(dists))
        feats["sr_dist_mean"]   = float(np.mean(dists))
        feats["sr_dist_p75"]    = float(np.percentile(dists, 75))

        # normalized 0..1 “tamper” score from median distance
        md = feats["sr_dist_median"]
        feats["tamper_score_lighting"] = float(np.clip(1.0 - np.exp(-md/2.0), 0.0, 1.0))
    else:
        feats["sr_dist_median"] = 0.0
        feats["sr_dist_mean"]   = 0.0
        feats["sr_dist_p75"]    = 0.0
        feats["tamper_score_lighting"] = 0.0

    return feats
