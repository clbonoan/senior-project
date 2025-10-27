# combines canny edge detection, local binary pattern (LBP), and shadow mask
# for texture analysis (first feature)

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# ========================================================
# LOCAL BINARY PATTERN
# =========================================================
# comparing texture to describe how pixel intensity changes around a central pixel
def get_pixel(img, center, x, y):
    new_value = 0

    try:
        # if local neighbor pixel >= center pixel 
        if img[x][y] >= center:
            new_value = 1
    except:
        # exception where neighbor value of center pixel may not exist
        # i.e., values present at boundaries
        pass

    return new_value

# calculate lbp
def lbp_calculated_pixel(img, x, y): 
    center = img[x][y]

    # create array of pixels
    val_ar = []

    # top left
    val_ar.append(get_pixel(img, center, x-1, y-1))

    # top
    val_ar.append(get_pixel(img, center, x-1, y))

    # top right
    val_ar.append(get_pixel(img, center, x-1, y+1))

    # right
    val_ar.append(get_pixel(img, center, x, y+1))

    # bottom right
    val_ar.append(get_pixel(img, center, x+1, y+1))

    # bottom
    val_ar.append(get_pixel(img, center, x+1, y))

    # bottom left
    val_ar.append(get_pixel(img, center, x+1, y-1))

    # left
    val_ar.append(get_pixel(img, center, x, y-1))

    # convert binary values to decimal
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]

    val = 0

    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]

    return val


def lbp_map(gray):
    # turning the image into grayscale and applying LBP to it
    height,width = gray.shape
    out = np.zeros((height,width), np.uint8)
    for i in range(height):
        for j in range(width):
            out[i,j] = lbp_calculated_pixel(gray, i, j)
    return out

def lbp_hist(patch):
    # 8-neighbor lbp -> values 0 to 255
    # histogram for local texture pattern distribution

    hist, _ = np.histogram(patch.ravel(), bins=256, range=(0,256), density=True)
    return hist


# ========================================================
# SHADOW MASK
# ========================================================
# necessary for isolating shadows from dark non-shadow objexts
# goal is to focus edge detection and texture analysis only on shadow regions

def normalized_rgb(img_bgr):
    # convert BGR image to RGB ratios (more normalized and less 
    # sensitive to lighting changes)

    # r = R/(R+G+B), g = G/(R+G+B), b = B/(R+G+B)
    # ratios change less under shadows, intensity drops but ratios stay similar
    b, g, r = cv.split(img_bgr.astype(np.float32))
    # add epsilon to avoid dividing by zero
    s = r + g + b + 1e-6
    # calculate with depth stack - color channels into 3 dimensional array
    # (height, width, 3)
    return np.dstack((r/s, g/s, b/s))

def box_mean(arr, k):
    # using box filter to replace each pixel's value with average of neighboring pixels
    # good for reducing noise
    return cv.boxFilter(arr, ddepth =- 1, ksize = (k, k), normalize = True)

def remove_small(mask, min_area = 800):
    # remove small connected white noise from the binary mask
    num, labels, stats, _ = cv.connectedComponentsWithStats(mask, connectivity = 8)
    keep = np.zeros_like(mask)
    # background is labeled 0
    for i in range(1, num):
        # use the total area (# of pixels) of the component
        if stats[i, cv.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 255
    return keep

def make_shadow_mask(img_bgr, win=51, k_dark=0.9, dr=0.06, dg=0.06,
                     morph_open=3, morph_close=5, min_area=800, use_clahe=True):
    # build the binary mask that marks the likely shadow regions
    # combine brightness and color consistency cues to distinguish shadows from dark objects

    # get L (brigtness) from Lab
    lab = cv.cvtColor(img_bgr, cv.COLOR_BGR2Lab)
    L = lab[:, :, 0].astype(np.float32)

    # use local contrast enhancement to make subtle shadows stand out
    if use_clahe:
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        L = clahe.apply(L.astype(np.uint8)).astype(np.float32)

    # local brightness mean and darkness condition
    # pixels darker than a fraction of local mean -> shadow candidates
    meanL = box_mean(L, win)
    # dark is true when the pixel is darker than neighbors
    dark = (L < meanL * k_dark)

    # chromacity (color) consistency condition
    # require chromaticity (r,g) to be locally stable (shadows darken, not recolor)
    nrg = normalized_rgb(img_bgr)
    mean_nrg = box_mean(nrg, win)
    chroma_ok = (
        (np.abs(nrg[:, :, 0] - mean_nrg[:, :, 0]) < dr) &
        (np.abs(nrg[:, :, 1] - mean_nrg[:, :, 1]) < dg)
    )

    # combine dark + color consistent (could mean likely a shadow)
    mask = (dark & chroma_ok).astype(np.uint8) * 255

    # morphological image processing to remove specks and fill in small holes
    if morph_open > 1:
        k1 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (morph_open, morph_open))
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, k1)
    if morph_close > 1:
        k2 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (morph_close, morph_close))
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, k2)

    # drop small regions tht represent noise in image
    mask = remove_small(mask, min_area=min_area)
    return mask, L


# =======================================================
# CANNY EDGE DETECTION
# =======================================================
def canny_on_L(L, low=50, high=150):
    # do the edge detection on the L channel (luminance/lighting)
    # detect strong brightness changes and potential shadow boundaries
    L8 = cv.normalize(L, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    return cv.Canny(L8, low, high)

# ========================================================
# TEXTURE COMPARISON AND HELPERS
# COMPARE TEXTURE ACROSS SHADOW BOUNDARIES
# ========================================================
def chi2(a, b, eps=1e-9):
    # chi-square distance between two histograms (to measure texture similarity)
    d = (a - b)
    s = (a + b) + eps
    return 0.5 * np.sum((d * d) / s)

def sample_patches(L, y, x, ny, nx, size=21, offset=6):
    # extract two small patches:
    # - one just inside the shadow (darker side)
    # - one just outside (brighter side)
    # offset: how far from the boundary to sample
    # size: patch width/height in pixels
    
    # move along the edge normal (ny,nx) in both directions
    yi_in  = int(np.clip(y - offset*ny, 0, L.shape[0]-1))
    xi_in  = int(np.clip(x - offset*nx, 0, L.shape[1]-1))
    yi_out = int(np.clip(y + offset*ny, 0, L.shape[0]-1))
    xi_out = int(np.clip(x + offset*nx, 0, L.shape[1]-1))

    h, w = L.shape
    s = size // 2
    def clamp(a, lo, hi): return max(lo, min(hi, a))

    # top-left coordinates for cropping patches
    y0_in  = clamp(yi_in - s, 0, h - size)
    x0_in  = clamp(xi_in - s, 0, w - size)
    y0_out = clamp(yi_out - s, 0, h - size)
    x0_out = clamp(xi_out - s, 0, w - size)

    pin  = L[y0_in:y0_in+size,   x0_in:x0_in+size  ]
    pout = L[y0_out:y0_out+size, x0_out:x0_out+size]
    return pin, pout

# helper functions for the tamper score (0...1)
def clamp01(x):
    # clamp a float value between 0 and 1
    return max(0.0, min(1.0, float(x)))

def contrast_norm(mean_in, mean_out, eps=1e-6):
    # normalized light/luminance contrast between inside and outside patches
    return clamp01((mean_out - mean_in) / (mean_out + eps))

def map_to_01(x, lo, hi):
    # map x linearly from [lo,hi] to [0,1]
    if hi <= lo:
        return 0.0
    return clamp01((x - lo) / (hi - lo))


# ==========================================================
# GLOBAL TEXTURE-INFILL DETECTION
# ==========================================================
def local_variance(img8, k=15):
    # local variance over a k by k window
    # higher variance means more texture, low variance means flat/smooth region
    f  = cv.boxFilter(img8.astype(np.float32), -1, (k,k), normalize=True)
    f2 = cv.boxFilter((img8.astype(np.float32)**2), -1, (k,k), normalize=True)

    # Var[X] = E[X^2] - (E[X])^2
    return np.maximum(0.0, f2 - f*f)

def texture_energy(L8, var_k=15, lap_k=3, alpha=0.6):
    # combine two texture cues
    # local variance (low variance texture) and local mean Laplacian (high variance texture)
    var = local_variance(L8, k=var_k)
    lap = cv.Laplacian(L8, cv.CV_32F, ksize=lap_k)
    lapE = cv.boxFilter(np.abs(lap), -1, (var_k,var_k), normalize=True)

    # normalize both to [0,1] to compare
    var_n = (var  - var.min())  / (var.max()  - var.min()  + 1e-6)
    lap_n = (lapE - lapE.min()) / (lapE.max() - lapE.min() + 1e-6)

    return alpha*var_n + (1.0 - alpha)*lap_n


# adaptive and context-aware to detect over-smoothed regions
def detect_textureless_infill_adaptive(
    img_bgr, L8,
    win=41,                 # neighborhood for mean/std (broader context)
    z_thresh=0.8,           # how many local stds smoother than mean to flag
    abs_def_thresh=0.05,    # absolute deficit floor in [0,1] tex units
    ring_win=61,            # larger window for "surround" comparison
    ring_margin=0.05,       # require surround to be at least +5% rougher
    grad_thresh=3.0,        # suppress inherently smooth (low-gradient) areas
    sat_thresh=0.10,        # and low-saturation areas (e.g., sky, walls)
    morph=5,
    min_area=1200
):
    # adaptive inpaint/infill detector:
    #  - z-score deficit: (mean_tex - tex) / std_tex  > z_thresh
    #  - absolute deficit: (mean_tex - tex)          > abs_def_thresh
    #  - ring test: surround (larger window) rougher than inside by ring_margin
    #  - suppress low-gradient + low-saturation zones (naturally smooth)
    # returns: mask (uint8), tex map (float in [0,1]), z map (float), frac area
    
    # 1) texture energy in [0,1]
    tex = texture_energy(L8, var_k=max(7, win//2), lap_k=3, alpha=0.6)

    # 2) local mean/std of texture energy
    mean_tex = cv.boxFilter(tex, -1, (win,win), normalize=True)
    mean_tex2 = cv.boxFilter(tex*tex, -1, (win,win), normalize=True)
    var_tex = np.maximum(0.0, mean_tex2 - mean_tex*mean_tex)
    std_tex = np.sqrt(var_tex + 1e-6)

    deficit = mean_tex - tex           # how much smoother than local mean
    z = deficit / (std_tex + 1e-6)     # z-score

    # 3) ring test: compare to a larger neighborhood
    mean_ring = cv.boxFilter(tex, -1, (ring_win, ring_win), normalize=True)
    ring_ok = (mean_ring >= (mean_tex * (1.0 + ring_margin)))  # surround rougher than inside

    # 4) suppress inherently smooth regions (low gradient + low saturation)
    gx = cv.Sobel(L8, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(L8, cv.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx*gx + gy*gy)  # not normalized; threshold is absolute-ish
    hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV).astype(np.float32)
    S = hsv[:,:,1] / 255.0

    not_inherently_smooth = ~((grad < grad_thresh) & (S < sat_thresh))

    # 5) combine gates
    cand = (z > z_thresh) & (deficit > abs_def_thresh) & ring_ok & not_inherently_smooth

    # 6) clean-up and area filter
    mask = (cand.astype(np.uint8) * 255)
    if morph > 1:
        k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (morph,morph))
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, k)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, k)
    mask = remove_small(mask, min_area=min_area)

    frac = float(np.count_nonzero(mask)) / float(mask.size)
    return mask, tex, z, frac


# ============================================================
# MAIN FUNCTION
# ============================================================
def analyze_image(path,
                  win=31, k_dark=0.8, dr=0.04, dg=0.04,
                  canny_low=50, canny_high=150,
                  patch_size=21, patch_offset=6,
                  lbp_sim_thresh=0.4, luminance_gap=10):
    # full shadow analysis:
    # 1. create shadow mask
    # 2. detect edges with Canny
    # 3. compare textures across edges using LBP
    # 4. estimate how many edges are true shadow boundaries
    
    img = cv.imread(path, cv.IMREAD_COLOR)
    assert img is not None, f"Cannot read image: {path}"

    # detect likely shadow areas (mask) and get luminance (L)
    mask, L = make_shadow_mask(img, win, k_dark, dr, dg)

    # find edges and keep only those along shadow boundaries
    edges = canny_on_L(L, canny_low, canny_high)
    boundary = cv.morphologyEx(mask, 
                               cv.MORPH_GRADIENT,
                               cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
    )
    shadow_edges = cv.bitwise_and(edges, edges, mask=boundary)

    # compute full LBP map (texture descriptor)
    L8 = cv.normalize(L, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    lbp = lbp_map(L8)

    # output LBP image with shadow outline
    # normalize LBP for visualizatation
    lbp_gray = cv.normalize(lbp, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    # outline the shadow areas (using mask boundaries)
    cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(lbp_gray, cnts, -1, (0, 255, 0), thickness = 3)

    # for each boundary point, compare texture inside vs outside shadow
    gx = cv.Sobel(L, cv.CV_32F, 1, 0, ksize=3)  # gradient X
    gy = cv.Sobel(L, cv.CV_32F, 0, 1, ksize=3)  # gradient Y

    ys, xs = np.where(shadow_edges > 0)
    take_every = max(1, len(ys)//1200)  # subsample to reduce computation
    
    #good_shadow_points = 0
    #tested_points = 0

    # tamper scoring
    tamper_scores = []
    valid_samples = 0

    # parameters for scoring
    min_contrast = 0.06     # require at least this much luminance L drop to consider a shadow edge
    d_good = 0.25       # LBP chi2 distance considered "very similar"
    d_bad = 0.80        # LBP chi2 distance considered "very different"
    contrast_weight = 0.75      # higher values trust strong shadows more


    for idx in range(0, len(ys), take_every):
        y, x = int(ys[idx]), int(xs[idx])
        g = np.array([gx[y,x], gy[y,x]], dtype=np.float32)
        nrm = np.linalg.norm(g)
        if nrm < 1e-3:
            continue
        nx, ny = g[0]/nrm, g[1]/nrm  # normalized gradient → edge normal direction

        # extract paired patches: darker side (inside shadow) and brighter side (outside)
        pinL, poutL = sample_patches(L8, y, x, ny, nx, size=patch_size, offset=patch_offset)
        pinB, poutB = sample_patches(lbp, y, x, ny, nx, size=patch_size, offset=patch_offset)

        # skip if patch dimensions invalid (near borders)
        if pinL.shape != (patch_size, patch_size) or poutL.shape != (patch_size, patch_size):
            continue

        # 1) luminance contrast (expect inside to be darker than outside)
        c = contrast_norm(np.mean(pinL), np.mean(poutL))
        if c < min_contrast:
            # if contrast is lower than minimum contrast, not enough to use for tamper score
            continue 

        # 2) texture difference (LBP chi2 distance)
        d = chi2(lbp_hist(pinB), lbp_hist(poutB))   # smaller distance means more similar

        # 3) map texture distance to [0,1] anomaly (0=similar, 1=different)
        t = map_to_01(d, d_good, d_bad)

        # 4) combine with contrast (emphasize anomalies when contrast is strong)
        s = clamp01((1.0 - contrast_weight) * t + contrast_weight * (t * c))
        tamper_scores.append(s)
        valid_samples += 1


    # compute a "shadow-likeness" score = fraction of edges that behave like shadows

    tamper_score_shadow = float(np.mean(tamper_scores)) if tamper_scores else 0.0
    print(f"Valid boundary samples: {valid_samples}")
    print(f"Tamper score: {tamper_score_shadow:.2f} (0 = normal, 1 = likely tampered)")

    # GLOBAL TEXTURE-INFILL TAMPER SCORE
    # looks for regions that are too smooth compared to their local neighborhood
    tex_mask, tex_map, z_map, tex_frac = detect_textureless_infill_adaptive(
        img, L8,
        win=31,            # smaller local context
        z_thresh=0.6,      # easier to pass
        abs_def_thresh=0.04,
        ring_win=41,        
        ring_margin=0.02,
        grad_thresh=2.5,
        sat_thresh=0.08,
        morph=3,
        min_area=400
    )
    # map suspicious area fraction → [0,1]; at ~2% area flagged, score ~1
    area_pivot = 0.10
    area_score = min(1.0, tex_frac / area_pivot)

    # - strength score: average excess z above threshold (clipped to [0,1])
    excess = np.maximum(0.0, z_map - 1.0)  # relative to z_thresh (1.0)
    if np.count_nonzero(tex_mask) > 0:
        strength_score = float(np.mean(excess[tex_mask > 0]))
        strength_score = max(0.0, min(1.0, strength_score / 2.0))  # normalize: z excess of ~2 → 1.0
    else:
        strength_score = 0.0
    
    tamper_score_texture = max(area_score, strength_score)

    print(f"Tamper score (texture infill): {tamper_score_texture:.2f}  (suspicious frac={tex_frac:.4f})")


    # COMBINE TAMPER SCORES
    # Conservative fusion: if either detector flags, final score is high
    tamper_score = max(tamper_score_shadow, tamper_score_texture)
    print(f"Tamper score (combined): {tamper_score:.2f}")

    # visualize the results
    overlay = img.copy()
    overlay[mask == 255] = (0, 0, 255)  # show detected shadow mask in red
    vis = cv.addWeighted(img, 0.7, overlay, 0.3, 0)

    cv.namedWindow("Shadow mask", cv.WINDOW_NORMAL)
    cv.resizeWindow("Shadow mask", 800, 600)       
    cv.imshow("Shadow mask", mask)
    cv.namedWindow("Shadow edges (Canny)", cv.WINDOW_NORMAL)
    cv.resizeWindow("Shadow edges (Canny)", 800, 600)
    cv.imshow("Shadow edges (Canny)", shadow_edges)
    cv.namedWindow("Overlay", cv.WINDOW_NORMAL)
    cv.resizeWindow("Overlay", 800, 600)
    cv.imshow("Overlay", vis)
    #cv.imshow("LBP + Shadow ouline", lbp_gray)
    plt.figure(figsize=(8,6))
    plt.imshow(cv.cvtColor(lbp_gray, cv.COLOR_BGR2RGB))
    plt.show()
    cv.waitKey(0)
    cv.destroyAllWindows()

    return {
        "mask": mask,
        "edges": edges,
        "shadow_edges": shadow_edges,
        "score": tamper_score
    }


if __name__ == "__main__":
    analyze_image("test.jpg")