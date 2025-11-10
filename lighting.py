# lighting.py - for lighting strength analysis (second feature)

import cv2 as cv
import numpy as np
import io, contextlib

# use same shadow mask as texture.py
from texture import make_shadow_mask as texture_make_shadow_mask

def mask_from_texture(img_bgr, suppress_prints=True, **kwargs): 
    # call texture.py's make_shadow_mask function and return just the mask

    if suppress_prints:
        _sink = io.StringIO()
        with contextlib.redirect_stdout(_sink):
            out = texture_make_shadow_mask(img_bgr, **kwargs)
    else:
        out = texture_make_shadow_mask(img_bgr, **kwargs)

    if isinstance(out, tuple):
        mask = out[0]
    else:
        mask = out
    return mask

# ----------------------------------------------------------
# shadow mask helper functions
# ----------------------------------------------------------
def srgb_to_linear(x8):
    # x in [0,1], sRGB -> linear
    x = x8.astype(np.float32) / 255.0
    a = 0.055
    return np.where(x <= 0.04045, x/12.92, ((x + a)/(1 + a))**2.4).astype(np.float32)

def box_mean(arr, k):
    # using box filter to replace each pixel's value with average of neighboring pixels
    # good for reducing noise
    return cv.boxFilter(arr, ddepth=-1, ksize=(k, k), normalize=True)

def remove_small(mask, min_area=800):
    # remove small connected white noise from the binary mask
    num, labels, stats, _ = cv.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask)
    # background is labeled 0
    for i in range(1, num):
        # use the total area (# of pixels) of the component
        if stats[i, cv.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 255
    return keep

def linear_Y_from_bgr(img_bgr):
    # compute linear luminance from BGR image
    B, G, R = cv.split(srgb_to_linear(img_bgr))
    # coefficients (0.114, 0.587, 0.299) are based on sensitivity of the human eye to diff light wavelengths
    return (0.114*B + 0.587*G + 0.299*R).astype(np.float32)

def normalized_rgb(img_bgr):
    # compute normalized chroma ratios (r and g)
    b, g, r = cv.split(img_bgr.astype(np.float32))
    s = r + g + b + 1e-6
    return r/s, g/s

def unit_rgb_dir(img_bgr):
    # turn each pixel's RGB into a pure color direction
    # checks if two spots are the same color material even if one is in shadow and other is brighter
    lin = srgb_to_linear(img_bgr)
    n = np.linalg.norm(lin, axis=2, keepdims=True) + 1e-6
    return lin/n

# -----------------------------------------------
# getting skew and kurtosis through numpy
# -----------------------------------------------
def np_skew(x):
    # skew measures asymmetry (bias=False)
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    m = x.mean()
    s = x.std(ddof=1)
    if s < 1e-12:
        return 0.0
    z = (x - m) / s
    return float(np.mean(z**3))

def np_kurtosis(x, fisher=True):
    # kurtosis measures "tailedness" or "peakedness" of the distribution (excess if fisher=True)
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return -3.0 if fisher else 0.0
    m = x.mean()
    s = x.std(ddof=1)
    if s < 1e-12:
        return -3.0 if fisher else 0.0
    z = (x - m) / s
    k = float(np.mean(z**4))
    return k - 3.0 if fisher else k

# -------------------------------------------------------
# lighting strength consistency 
# based on mathematical concepts found in "Exposing image forgery by detecting consistency of shadow" by Ke, Qin, Min, Zhang
# use of strength estimation of light source shadow model in paper
# -------------------------------------------------------

# check whether all shadows in image have similar brightness and lighting strength
# pipeline:
# 1) shadow mask reused from texture.py
# 2) separate mask into sections; each section = one shadow
# 3) for each shadow region:
#   - find the umbra (darkest area) inside the shadow
#   - find small bright area right next to shadow on the same surface (comparing how much darker shadow is)
#   - measure "shadow strength" based on how much light is blocked
#       - formula: I_shadow = (approx equal) ((Y_shadow - Y_light) / Y_light) * Y_shadow
#           where Y_shadow and Y_light are pixel brightness values
#   - summarize section using four values: mean, std dev, skewness, kurtosis
# 4) compare all shadow regions and see how similar their stats are
#   - if most are similar, image is likely to be real
# 5) turn similarity into tamper score (s_lighting):
#   s_lighting = 1 - median_similatiry
#   lower score = consistent lighting (real)
#   higher score = inconsistent lighting (tampered)

# paper assumes k_i is 0 for the umbra region
# so erode avoids penumbra and you keep darkest X% inside region
def component_umbra(mask_bool, Y, darkest_frac=0.5, erode_px=3):
    # umbra = intersection beteen eroded component and darkest X% inside component
    # return bool mask or None if too small
    # bool mask = true for pizels inside one shadow, false elsewhere
    # cv.connectedComponents(mask) splits full shadow mask into individual shadows) then
    # bool mask tells which region of image belongs to a particular shadow

    if erode_px > 0:
        mask_bool = cv.erode(mask_bool.astype(np.uint8), np.ones((3,3), np.uint8),
                             iterations=erode_px).astype(bool)
    vals = Y[mask_bool]
    if vals.size == 0:
        return None
    cut = np.quantile(vals, darkest_frac)
    umbra = mask_bool & (Y <= cut)
    return umbra if umbra.sum() > 50 else None

# paper says to compare the shadow area to a nearby non-shadow area of same surface
def same_material_ring(img_bgr, comp_mask_bool, Y, ring_px=4, chroma_thr=0.04):
    # ring outside the component (shadow region)
    # filtered by chroma (linear RGB) to approximate the same material
    # Yn

    dil = cv.dilate(comp_mask_bool.astype(np.uint8), np.ones((3,3), np.uint8),
                    iterations=ring_px).astype(bool)
    ring = dil & (~comp_mask_bool) & (Y > 1e-3)     # lit pixels
    if ring.sum() == 0:
        return None
    
    dirs = unit_rgb_dir(img_bgr)
    edge = comp_mask_bool & (~cv.erode(comp_mask_bool.astype(np.uint8),
                                       np.ones((3,3), np.uint8), 1).astype(bool))
    if edge.sum() == 0:
        return ring

    edge_dir = dirs[edge].mean(axis=0, keepdims=True)   # 1x3
    ridx = np.flatnonzero(ring.ravel())
    d_ring = dirs.reshape(-1,3)[ridx]
    cosd = 1.0 - (d_ring @ edge_dir.T).ravel()  # cosine distance
    keep = np.zeros(ring.size, bool)
    keep[ridx[cosd <= chroma_thr]] = True
    sel = keep.reshape(ring.shape)
    return sel if sel.sum() > 0 else ring

def features_one_shadow(img_bgr, comp_mask_bool, ring_px=4, chroma_thr=0.04, darkest_frac=0.5, erode_px=3):
    # for one shadow component, get the following:
    #   - umbra extraction
    #   - lit ring -> Y_N estimate for median
    #   - I_shadow approx equal to ((Y_S - Y_N)/Y_N) * Y_S over linear Y
    #   - 4D feature with mean, std dev, skew, kurtosis over umbra

    Y = linear_Y_from_bgr(img_bgr)

    umbra = component_umbra(comp_mask_bool, Y, darkest_frac, erode_px)
    if umbra is None:
        return None

    ring = same_material_ring(img_bgr, comp_mask_bool, Y, ring_px, chroma_thr)
    if ring is None or ring.sum() < 30:
        return None

    Yn = float(np.median(Y[ring]))  # lit reference
    if not np.isfinite(Yn) or Yn < 1e-4:
        return None

    # (ATTN: NOT AS ACCURATE FOR DIFF SCENES) equation 12 special case (shadow strength Ishadow) in linear space
    # Ishadow = ((Y - Yn) / Yn) * Y

    # stats of light strength distribution for each shadow
    # vals = Ishadow[umbra]
    # vals = vals[np.isfinite(vals)]
    # if vals.size < 50:
    #    return None
    
    # shadow strength estimation in umbra
    Ys = Y[umbra]
    Ys = Ys[np.isfinite(Ys)]
    if Ys.size < 50:
        return None

    # shadow strength using paper's approach
    # use ratio of shadow to lit intensity (more stable than paper formula for diverse scenes)
    strength_ratios = Ys / (Yn + 1e-6)

    # from paper: 
    # mean (avg strength), std dev (how much it varies),
    # ^ (ATTN: using median instead of mean for robustness; using IQR instead of std dev in case of skewed data)
    # skew (whether it's mostly dark or has brighter spots),
    # kurtosis (how "peaked" or "flat" the brightness dist is)
    return np.array([
        float(np.median(strength_ratios)),
        float(np.percentile(strength_ratios, 75) - np.percentile(strength_ratios, 25)),
        np_skew(strength_ratios),
        np_kurtosis(strength_ratios, fisher=True)
    ], np.float32)   

# --------------------------------------------------
# tamper score
# -------------------------------------------------- 
def light_strength_tamper_score(
        img_bgr,
        min_area=1200,
        ring_px=6,
        chroma_thr=0.08,
        darkest_frac=0.30,
        erode_px=2,
        mask_kwargs=None      
):
    # compute tamper score based on the following:
    # - build mask from texture.py
    # - extract features per-shadow
    # - similarity = cosine(Z-score features), use MEDIAN
    # - score = 1 - median_similarity (between 0 (real) and 1 (tampered))

    mask_kwargs = mask_kwargs or {}
    # Build mask using *the same* function as texture.py
    mask_u8 = mask_from_texture(img_bgr, **mask_kwargs)

    # Cleanup + connected components
    mask_u8 = remove_small((mask_u8 > 0).astype(np.uint8) * 255, min_area=min_area)
    num, labels = cv.connectedComponents(mask_u8)

    feats = []
    for i in range(1, num):  # 0 = background
        comp = (labels == i)
        f = features_one_shadow(img_bgr, comp,
                                 ring_px=ring_px,
                                 chroma_thr=chroma_thr,
                                 darkest_frac=darkest_frac,
                                 erode_px=erode_px)
        if f is not None:
            feats.append(f)

    if len(feats) < 2:
        print(f"components: {int(num-1)}, kept_feats: {len(feats)} (insufficient for comparison)")
        return 0.0  # not enough shadows to compare

    # if correlation coeff r is not close to 1, suspect tampering
    # num_shadows x 4 feature matrix
    F = np.vstack(feats) 

    # normalize each feature using robust stats
    medians = np.median(F, axis=0)
    # use median instead of std dev for robustness
    med = np.median(np.abs(F - medians), axis=0)
    # no division by zero
    scale = np.where(med > 1e-8, med, 1.0)
    # z-score per feature to normalize values so big values don't outweigh small values
    Z = (F - medians) / scale   

    # # how long each feature vector is; for cosine similarity (make comparison fair)
    # norms = np.linalg.norm(Z, axis=1, keepdims=True)

    # # compare every shadow with every other shadow
    # # use cosine similarity between features (S=1 (shadows similar), S=0 (shadows are different))
    # S = (Z @ Z.T) / (norms @ norms.T + 1e-8)  # cosine similarity
    # # no self comparisons or comparing same pair twice
    # tri = S[np.triu_indices_from(S, 1)]

    # # median similarity across all shadow pairs (single value for how similar all shadows are overall)
    # med_sim = float(np.median(tri))           # robust central tendency

    # # convert similarity into tamper score
    # # if shadows are similar, med_sim approx 1, score closer to 0
    # # if shadows are different, med_sim smaller, score closer to 1
    # score = float(np.clip(1.0 - med_sim, 0.0, 1.0))

    n_shadows = Z.shape[0]
    distances = []
    for i in range(n_shadows):
        for j in range(i+1, n_shadows):
            dist = np.linalg.norm(Z[i] - Z[j])
            distances.append(dist)

    if len(distances) == 0:
        return 0.0
    
    # convert distances to similarity scores
    # smaller distances = higher similarity
    distances = np.array(distances)
    median_dist = float(np.median(distances))

    # normalize distance to [0,1] score
    # lower median distance -> lower score (more consistent)
    score = float(np.clip(1.0 - np.exp(-median_dist / 2.0), 0.0, 1.0))

    print(f"components: {int(num-1)}, kept_feats: {len(feats)}")
    print(f"median_sim: {median_dist:.3f}, score: {score:.3f}")

    return score    

# ---------------------------------------------------
# choose image
# ---------------------------------------------------
def analyze_lighting(image_path):
    # print tamper score
    # similar to texture.py
    img = cv.imread(image_path, cv.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(image_path)

    score = light_strength_tamper_score(img)

    print(f"{score:.4f}")
    return score    

if __name__ == "__main__":
    analyze_lighting("data/images/2.jpg")