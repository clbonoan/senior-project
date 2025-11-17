# texture_features.py for texture feature extraction
import numpy as np
import cv2 as cv

# small helpers
def mean_std(arr):
    arr = np.asarray(arr).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 0.0
    return float(np.mean(arr)), float(np.std(arr))

def percentile(arr, p):
    arr = np.asarray(arr).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, p))

def lbp_entropy_np(lbp, region=None):
    vals = lbp[region>0].ravel() if region is not None else lbp.ravel()
    if vals.size == 0:
        return 0.0
    hist, _ = np.histogram(vals, bins=256, range=(0,256), density=True)
    hist = hist + 1e-12
    return float(-np.sum(hist * np.log(hist)))


def component_stats_min(mask):
    # return only num_shadow_components, comp_area_p90, comp_peri2_over_area_p50
    num, labels, stats, _ = cv.connectedComponentsWithStats(mask, connectivity=8)
    areas, peri2_over_area = [], []

    for i in range(1, num):
        # 0 is the background
        a = stats[i, cv.CC_STAT_AREA]
        areas.append(a)
        comp = (labels == i).astype(np.uint8) * 255
        cnts, _ = cv.findContours(comp, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if cnts and a > 0:
            peri = cv.arcLength(max(cnts, key=cv.contourArea), True)
            peri2_over_area.append((peri * peri) / a)

    out = {"num_shadow_components": int(num - 1)}
    out["comp_area_p90"] = percentile(areas, 90)
    out["comp_peri2_over_area_p50"] = percentile(peri2_over_area, 50)
    return out

def angular_variance_min(gx, gy, boundary_mask, eps=1e-6):
    # return only: normal_mrl (alignment) and normal_angle_std (irregularity)
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
    R = float(np.sqrt(C * C + S * S))     # mean resultant length is [0..1]
    ang_std = float(np.sqrt(-2 * np.log(R + eps)))
    return {"normal_angle_std": ang_std, "normal_mrl": R}

# local intermediates in case they were not passed
def to_L8(img_bgr):
    # uint8 luminance proxy (keep simple; lighting module can use linear space separately)
    return cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)

def sobel_xy(L8):
    gx = cv.Sobel(L8, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(L8, cv.CV_32F, 0, 1, ksize=3)
    return gx, gy

def boundary_from_mask(mask):
    se = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    return cv.morphologyEx(mask, cv.MORPH_GRADIENT, se)

# main: minimal strong baseline
def extract(
    img_bgr, 
    *, 
    mask, 
    lbp, 
    L8, 
    gx, 
    gy, 
    chi2_list
) -> dict:
    '''
    to build feature vector:
    - mask (uint8 0/255),
    - lbp (uint8),
    - L8 (uint8 luminance),
    - gx, gy (float32 for Sobel gradients),
    - chi2_list (contains chi-squared distances comparing LBP patches inside vs outside)
    expected inputs: intermediates from texture.py and per-boundary sample arrays
    '''

    feats = {}

    if L8 is None:
        L8 = to_L8(img_bgr)
    if gx is None or gy is None:
        gx, gy = sobel_xy(L8)

    # boundary and masks
    mask = mask.astype(np.uint8)
    boundary = boundary_from_mask(mask)

    in_m = mask > 0
    out_m = ~in_m

    # coverage/edges 
    '''
    mask_frac = how much of the image is covered by a detected shadow
    boundary_frac = how long or detailed the shadow edge is compared to the whole image
    '''
    feats["mask_frac"] = float(in_m.mean())
    feats["boundary_frac"] = float((boundary > 0).mean())

    # luminance contrast (means/stds only)
    '''
    L_in_mean/L_out_mean = average brightness inside and outside the shadow
    (shadows should be darker)
    L_in_std/L_out_std = how much brightness varies (contrast inside and outside)
    (real shadows keep surface texture, fake ones look smooth)
    contrast_shadow_vs_non = how strong the brightness drop is from light to shadow
    (ratio of how dark the shadow region is compared to the lit area)
    '''
    if in_m.any():
        mu_in, sd_in = mean_std(L8[in_m])
    else:
        mu_in, sd_in = 0.0, 0.0
    if out_m.any():
        mu_out, sd_out = mean_std(L8[out_m])
    else:
        mu_out, sd_out = 0.0, 0.0 


    feats["L_in_mean"] = mu_in
    feats["L_in_std"]  = sd_in
    feats["L_out_mean"] = mu_out
    feats["L_out_std"]  = sd_out
    feats["contrast_shadow_vs_non"] = (mu_out - mu_in) / (mu_out + 1e-6)

    # texture entropy (LBP)
    '''
    entropy = measure of randomness in the texture pattern (higher = more detailed texture)
    inside vs outside entropy comparison tells whether the shadowed region kept the same --
    texture randomness as the lit area
    '''
    feats["lbp_entropy_in"]  = lbp_entropy_np(lbp, region=mask)
    feats["lbp_entropy_out"] = lbp_entropy_np(lbp, region=(255 - mask))

    # shadow component shape (minimal)
    '''
    look at shape of the shadow areas
    - how many separate shadow blobs exist
    - how large they are
    - how irregular the boundaries are
    - ^ helps describe whether the detected shadow is one smooth region or lots of tiny parts (noisy)
    '''
    feats |= component_stats_min(mask)

    # boundary geometry consistency
    '''
    use gradient directions gx and gy to see how consistent the shadow edge direction is
    - if all edge directions are aligned, clear and natural boundary
    - if edge directions point everywhere, messy or fake
    '''
    feats |= angular_variance_min(gx, gy, boundary)

    # boundary sampling aggregates (compact set)
    #d_arr = np.asarray(d_list, dtype=np.float32)
    #s_arr = np.asarray(s_arr, dtype=np.float32)

    # feats["valid_boundary_samples"] = int(len(d_arr))

    # # texture chi2 across boundary: mean + high percentile
    # feats["tex_chi2_d_mean"] = float(d_arr.mean()) if d_arr.size else 0.0
    # feats["tex_chi2_d_p85"]  = _percentile(d_arr, 85) if d_arr.size else 0.0
    # feats["share_high_texture_jump"] = float(np.mean(d_arr > d_hi)) if d_arr.size else 0.0

    # # tamper score per-sample: mean + high percentile
    # feats["tamper_sample_s_mean"] = float(s_arr.mean()) if s_arr.size else 0.0
    # feats["tamper_sample_s_p90"]  = _percentile(s_arr, 90) if s_arr.size else 0.0
    # feats["share_high_tamper"]    = float(np.mean(s_arr > 0.7)) if s_arr.size else 0.0

    # # keep these if you want to debug thresholds later; they won't hurt
    # feats["d_lo"] = float(d_lo)
    # feats["d_hi"] = float(d_hi)

    # chi-squared aggregates (texture similarity across boundary)
    '''
    how similar or different the textures are between the shadow side and a nearby lit side
    - chi2_count = how many patch comparisons made
    - chi2_mean = average of how different textures are across the shadow edge
    - chi2_std = how much of the differences vary from one place to another
    - chi2_p25/p50/p75  = low, middle, and high ranges of texture differences
    - share_low_chi2 = fraction of edges where textures look very similar (likely real)
    - share_high_chi2 = fraction of edges where textures look very different (possible fake)
    '''
    d_arr = np.asarray(chi2_list, dtype=np.float32)
    feats["chi2_count"] = int(d_arr.size)
    feats["chi2_mean"] = float(d_arr.mean()) if d_arr.size else 0.0
    feats["chi2_std"] = float(d_arr.std()) if d_arr.size else 0.0
    feats["chi2_p25"] = percentile(d_arr, 25) if d_arr.size else 0.0
    feats["chi2_p50"] = percentile(d_arr, 50) if d_arr.size else 0.0
    feats["chi2_p75"] = percentile(d_arr, 75) if d_arr.size else 0.0

    # how often similarity is high (small chi-squared value) or low (large chi-squared value)
    '''
    low std = region looks smooth (possibly fake shadow)
    high std = region has texture detail (typical of real shadow)
    '''
    feats["share_low_chi2"]  = float(np.mean(d_arr <= (feats["chi2_p25"] if d_arr.size else 0.0))) if d_arr.size else 0.0
    feats["share_high_chi2"] = float(np.mean(d_arr >= (feats["chi2_p75"] if d_arr.size else 0.0))) if d_arr.size else 0.0

    return feats