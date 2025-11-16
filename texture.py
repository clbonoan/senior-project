# combines canny edge detection, local binary pattern (LBP), and shadow mask
# for texture analysis (first feature)

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from features import features_from_products

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
    val_ar.append(get_pixel(img, center, x-1, y-1)) # top left
    val_ar.append(get_pixel(img, center, x-1, y))   # top
    val_ar.append(get_pixel(img, center, x-1, y+1))  # top right
    val_ar.append(get_pixel(img, center, x, y+1))    # right
    val_ar.append(get_pixel(img, center, x+1, y+1))   # bottom right
    val_ar.append(get_pixel(img, center, x+1, y))    # bottom
    val_ar.append(get_pixel(img, center, x+1, y-1))    # bottom left
    val_ar.append(get_pixel(img, center, x, y-1))    # left
    
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

# ----------------------------------------------------------
# SHADOW MASK HELPERS
# ----------------------------------------------------------
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

# RGB to HSI conversion
def bgr_to_hsi(img_bgr):
    img_bgr = img_bgr.astype(np.float32) / 255.0
    B, G, R = cv.split(img_bgr)
    
    # intensity
    I = (R + G + B) / 3.0
    
    # saturation
    min_rgb = np.minimum(np.minimum(R, G), B)
    S = np.where(I > 1e-6, 1 - (min_rgb / (I + 1e-6)), 0)
    
    # hue (not needed for shadow detection)
    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G)**2 + (R - B) * (G - B)) + 1e-6
    theta = np.arccos(np.clip(num / den, -1, 1))
    H = np.where(B <= G, theta, 2 * np.pi - theta)
    H = H * 180 / np.pi  # convert to degrees
    
    return H, S, I

def srgb_to_linear(x8):
    # x in [0,1], sRGB -> linear
    x = x8.astype(np.float32) / 255.0
    a = 0.055
    return np.where(x <= 0.04045, x/12.92, ((x + a)/(1 + a))**2.4)

def normalized_rgb(img_bgr):
    b, g, r = cv.split(img_bgr.astype(np.float32))
    # calculate sum of pixel values
    # total light intensity (similar to luminance) at each pixel location
    s = r + g + b + 1e-6
    return r/s, g/s     # chroma ratios red and green since they are more stable than blue channel

def bgr_to_hsi_linear(img_bgr):
    # compute H,S,I in linear luminance
    # this is important to adjust brightness without affecting shadow's hue or saturation
    B, G, R = cv.split(img_bgr.astype(np.float32) / 255.0)
    Rl, Gl, Bl = srgb_to_linear(R), srgb_to_linear(G), srgb_to_linear(B)
    I = (Rl + Gl + Bl) / 3.0
    min_rgb = np.minimum(np.minimum(Rl, Gl), Bl)
    S = np.where(I > 1e-6, 1.0 - (min_rgb / (I + 1e-6)), 0.0)

    # hue not used; return a dummy H to keep signature the same
    H = np.zeros_like(I, dtype=np.float32)
    return H, S, I

# ------------------------------------------------------------
# SHADOW MASK
# ------------------------------------------------------------
# mask to further distinguish shadows
def make_shadow_mask(
    img_bgr,
    beta = 0.8,    # projection weight of Y into I' (0 < beta <= 1)
    win_scales = (21, 41, 81),    # local mean windows (pixels)
    k_dark = (0.92, 0.95, 0.98),    # darker than local mean factors
    dr = 0.06, dg = 0.06,   # chroma-consistency tolerance in normalized RGB
    morph_open = 3,
    morph_close = 7,
    min_area = 400,
): 
    '''
    based on paper by Uddin, Khanam, Khan, Deb, and Jo detailing color models HSI and YCbCr
    1. chromatic attainment on S: Im = S - log(S + delta)
    2. intensity attainment: I' = I + beta * Y    (Y is from YCrCb color model)
    3. shadow if I' is highly saturated and S' (boosted) is low
    '''
    
    # HSV for saturation and intensity (V (value) is used for intensity)
    #hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV).astype(np.float32)
    #H, S, V = cv.split(hsv)
    #Sn = S / 255.0
    #In = V / 255.0


    # convert BGR to HSI in linear light (hue, saturation, intensity)
    _, S, I = bgr_to_hsi_linear(img_bgr)

    # multi-scale local darkness region of interest (ROI)
    meanI = [box_mean(I, w) for w in win_scales]
    darks = [(I < (kk * m)) for m, kk in zip(meanI, k_dark)]
    roi_dark = darks[0] | darks[1] | darks[2]
    if not np.any(roi_dark):
        print("roi_dark is empty → returning empty mask.")
        return np.zeros(I.shape, np.uint8), (I * 255).astype(np.float32)
    
    # chromatic attainment (1) 
    # Sm = S - np.log(S + delta)
    # SKIPPED chromatic attainment; used raw saturation with thresholds instead
    # since shadows typically have low saturation in the image 

    # get Y channel from YCrCb for intensity attainment
    B, G, R = cv.split(srgb_to_linear(img_bgr))
    # common approximation to convert RGB to linear luminance
    # coefficients (0.114, 0.587, 0.299) are based on sensitivity of the human eye to diff light wavelengths
    Y_lin = 0.114*B + 0.587*G + 0.299*R
    Iprime_raw = I + beta * Y_lin

    # normalize I' by a high percentile inside dark region of interest (adaptive to scene/image)
    scale = float(np.percentile(Iprime_raw[roi_dark], 95))
    Iprime = np.clip(Iprime_raw / max(scale, 1e-6), 0.0, 1.0)

    # chroma-consistency (checked since shadows dim but do not change color)
    nr, ng = normalized_rgb(img_bgr)
    mnr = box_mean(nr, 41)  # mean of normalized r
    mng = box_mean(ng, 41)  # mean of normalized g
    chroma_ok = (np.abs(nr - mnr) < dr) & (np.abs(ng - mng) < dg)

    # self-tuning thresholds from region of interest percentiles
    S_thr = float(min(0.30, np.percentile(S[roi_dark], 40) + 0.02))
    Ip_thr = float(np.percentile(Iprime[roi_dark], 60))

    '''
    constraints for shadow mask based on if I' = 255 & S about 0
    to make it adaptive to all images, shadows have:
    - low saturation (S below given threshold)
    - low intensity even after intensity boost (I' < I_threshold) -> shadows stay dark
    - were already dark before intensity boost (roi_dark) -> must start dark
    - shadows do not change color even after boost (chroma_ok)
    '''
    mask = (roi_dark & chroma_ok & (S <= S_thr) & (Iprime <= Ip_thr)).astype(np.uint8) * 255

    # refine to put weak edges at boundaries
    # L8 = (I * 255).astype(np.uint8)
    # edges = cv.Canny(L8, 50, 150)
    # outline = cv.morphologyEx(mask, cv.MORPH_GRADIENT, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3)))
    # snap = cv.bitwise_and(outline, cv.dilate(edges, cv.getStructuringElement(cv.MORPH_RECT, (3,3)), 1))
    # mask = cv.bitwise_or(mask, snap)

    # use morphological image processing to remove specks and fill in small holes
    if morph_open > 1:
        k1 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (morph_open, morph_open))
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, k1)
    if morph_close > 1:
        k2 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (morph_close, morph_close))
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, k2)
    mask = remove_small(mask, min_area=min_area)
    
    # return mask and luminance channel to reuse (V)
    return mask, (I * 255).astype(np.float32)

# ---------------------------------------------------------
# Texture Comparison and Helpers
# Compare texture across shadow boundaries
# ---------------------------------------------------------
def chi2(a, b, eps=1e-9):
    # chi-squared distance between two histograms (measuring texture simularity)
    d = (a - b)
    s = (a + b) + eps
    return 0.5 * np.sum((d * d) / s)

# helper functions for the tamper score (0...1)
def clamp01(x):
    # clamp a float value between 0 and 1
    return max(0.0, min(1.0, float(x)))

def contrast_norm(mean_in, mean_out, eps=1e-6):
    # how much darker the inside is vs outside; normalized
    return clamp01((mean_out - mean_in) / (mean_out + eps))

def map_to_01(x, lo, hi):
    # map x linearly from [lo,hi] to [0,1]
    if hi <= lo:
        return 0.0
    # interpolate
    y = (x - lo) / (hi - lo)
    # clip 
    return max(0.0, min(1.0, y))

# return patches and top-left coordinates so they can be drawn
def sample_patches(img2d, y, x, ny, nx, size=21, offset=6):
    # extract two small patches:
    # - one inside the shadow (darker side)
    # - one outside (brighter side)
    # offset: how far from the boundary to sample
    # size: patch width/height in pixels
    
    h, w = img2d.shape
    s = size // 2

    # move along the edge normal (ny,nx) in both directions
    yi_in  = int(np.clip(y - offset*ny, 0, img2d.shape[0]-1))
    xi_in  = int(np.clip(x - offset*nx, 0, img2d.shape[1]-1))
    yi_out = int(np.clip(y + offset*ny, 0, img2d.shape[0]-1))
    xi_out = int(np.clip(x + offset*nx, 0, img2d.shape[1]-1))

    def clamp(v, lo, hi): return max(lo, min(hi, v))

    # top-left coordinates for cropping patches
    y0_in  = clamp(yi_in - s, 0, h - size)
    x0_in  = clamp(xi_in - s, 0, w - size)
    y0_out = clamp(yi_out - s, 0, h - size)
    x0_out = clamp(xi_out - s, 0, w - size)

    pin  = img2d[y0_in:y0_in+size, x0_in:x0_in+size]
    pout = img2d[y0_out:y0_out+size, x0_out:x0_out+size]
    return pin, pout, (x0_in, y0_in), (x0_out, y0_out)

# ---------------------------------------------------------
# choosing image
# ---------------------------------------------------------
def analyze_texture(image_input, visualize=True, max_pairs_vis=200):
    '''
    analyze texture across shadow boundaries using LBP and return chi-square similarities
    - highlight the exact patches inside and outside shadow that are compared
    - return raw chi-square distances for ML feature engineering
    '''
    if isinstance(image_input, str):
        img = cv.imread(image_input)
    else:
        img = image_input

    assert img is not None, f"Cannot read image: {image_input}"

    # shadow mask
    mask,L = make_shadow_mask(
        img, 
        beta = 0.8,  # trying 0.6-0.8 (street level) or 0.8-1.0 (aerial)
        win_scales = (21, 41, 81),
        k_dark = (0.92, 0.95, 0.98),
        dr=0.06, dg=0.06,
        morph_open = 3, morph_close = 7, min_area = 300
    )

    # canny edge
    # outline of the mask
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    mask_outline = cv.morphologyEx(mask, cv.MORPH_GRADIENT, k)
    L8 = np.uint8(np.clip(L, 0, 255))

    # gradients of light to estimate boundary normal directions
    gx = cv.Sobel(L8, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(L8, cv.CV_32F, 0, 1, ksize=3)

    # 3. lbp map
    lbp = lbp_map(L8)
    lbp_vis = cv.normalize(lbp, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    # outline the shadow areas (using mask boundaries)
    lbp_color = cv.applyColorMap(lbp_vis, cv.COLORMAP_MAGMA)
    lbp_color[mask_outline > 0] = (255, 0, 0)

    # tamper score based on texture consistency across shadow bounds
    ys, xs = np.where(mask_outline > 0)              # use the mask boundary itself
    
    # initialize lists/vars
    #d_list, c_list, s_arr = [], [], []
    #d_lo, d_hi = 0.0, 1.0
    chi2_list = []  # per-pair chi-square distances (lower = more similar)
    patch_pairs = []    # rectangles for visualization
    
    valid = 0

    if len(ys) == 0:
    #     print("Tamper score: 0.00 (no boundary)")
    # else:
    
        # parameters
        patch_size   = 25
        patch_offset = 9
        min_contrast = 0.04       # slightly lower gate
        #contrast_weight = 0.75

        step = max(1, len(ys)//1200)  # use a subsample for speed
        
        for idx in range(0, len(ys), step):
            y, x = int(ys[idx]), int(xs[idx])

            # boundary normal from luminance gradient
            g = np.array([gx[y, x], gy[y, x]], dtype=np.float32)
            nrm = float(np.linalg.norm(g))
            if nrm < 1e-3:
                continue
            nx, ny = g[0]/nrm, g[1]/nrm

            # sample patches on L (luminance) channel and on LBP
            pinL,  poutL, in_xy, out_xy = sample_patches(L8,  y, x, ny, nx, size=patch_size, offset=patch_offset)
            pinLBP, poutLBP, _, _ = sample_patches(lbp, y, x, ny, nx, size=patch_size, offset=patch_offset)
            if pinL.shape != (patch_size, patch_size) or poutL.shape != (patch_size, patch_size):
                continue

            # ensure we compare "inside (darker) vs outside (brighter)"
            m_in, m_out = float(np.mean(pinL)), float(np.mean(poutL))
            if m_out <= m_in:
                # swap if orientation came out reversed or flat
                pinL, poutL = poutL, pinL
                pinLBP, poutLBP = poutLBP, pinLBP
                in_xy, out_xy = out_xy, in_xy
                m_in, m_out = m_out, m_in

            # contrast gate
            c = contrast_norm(m_in, m_out)
            if c < min_contrast:
                continue

            # chi-squared distance between LBP histograms
            d = chi2(lbp_hist(pinLBP), lbp_hist(poutLBP))

            #d_list.append(d)
            #c_list.append(c)
            chi2_list.append(d)
            patch_pairs.append((in_xy, out_xy, patch_size))
            valid += 1

        # if valid == 0:
        #     print("Tamper score: 0.00 (no valid samples after gates)")
        # else:
        #     d_arr = np.array(d_list, dtype=np.float32)
        #     c_arr = np.array(c_list, dtype=np.float32)

        #     # adaptive mapping of chi-squared → [0,1]
        #     # using 35th percentile and 85th percentile
        #     d_lo = float(np.percentile(d_arr, 35))   # “similar” boundary
        #     d_hi = float(np.percentile(d_arr, 85))   # “different” boundary
        #     if d_hi <= d_lo:                          # safety
        #         d_hi = d_lo + 1e-3

        #     # map each sample to anomaly score and weight by contrast
        #     t_arr = np.clip((d_arr - d_lo) / (d_hi - d_lo), 0.0, 1.0)
        #     s_arr = np.clip((1.0 - contrast_weight) * t_arr +
        #                     contrast_weight * (t_arr * c_arr), 0.0, 1.0)

        #     tamper = float(np.mean(s_arr))
        #     print(f"Valid boundary samples: {valid}")
        #     print(f"Chi-squared percentiles: P35={d_lo:.3f}, P85={d_hi:.3f}")
        #     print(f"Tamper score (boundary): {tamper:.2f} (0 = normal, 1 = likely tampered)")

    # build feature measurements for training
    features = features_from_products(
        mask = mask,
        lbp = lbp,
        L8 = L8, 
        gx = gx,
        gy = gy,
        chi2_list = chi2_list,
    )

    if visualize:
        # visualize results
        mask_overlay = img.copy()
        mask_overlay[mask == 255] = (0, 0, 255)  # red mask overlay
        mask_overlay = cv.addWeighted(img, 0.7, mask_overlay, 0.3, 0)

        cv.namedWindow("Shadow Mask", cv.WINDOW_NORMAL)
        cv.resizeWindow("Shadow Mask", 800, 600)
        cv.imshow("Shadow Mask", mask)

        cv.namedWindow("Overlay", cv.WINDOW_NORMAL)
        cv.resizeWindow("Overlay", 800, 600)
        cv.imshow("Overlay", mask_overlay)

        cv.waitKey(1)

        # show lbp with matplotlib
        plt.figure(figsize=(8, 6))
        plt.imshow(lbp_vis, cmap="gray")
        plt.imshow(cv.cvtColor(lbp_color, cv.COLOR_BGR2RGB))
        plt.title("LBP (on L) with bounds in red")
        plt.show()

        # patch pair rectangles (inside=red, outside=green)
        overlay_pairs = img.copy()
        vis_n = min(len(patch_pairs), max_pairs_vis)
        for i in range(vis_n):
            (x_in, y_in), (x_out, y_out), s = patch_pairs[i]
            cv.rectangle(overlay_pairs, (x_in, y_in), (x_in + s, y_in + s), (0, 0, 255), 2)
            cv.rectangle(overlay_pairs, (x_out, y_out), (x_out + s, y_out + s), (0, 255, 0), 2)
            cv.putText(overlay_pairs, str(i), (x_in, max(0, y_in - 3)), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1, cv.LINE_AA)
            cv.putText(overlay_pairs, str(i), (x_out, max(0, y_out - 3)), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1, cv.LINE_AA)                

        cv.namedWindow("Patch Pairs (red=inside shadow, green=outside lit)", cv.WINDOW_NORMAL)
        cv.resizeWindow("Patch Pairs (red=inside shadow, green=outside lit)", 1000, 750)
        cv.imshow("Patch Pairs (red=inside shadow, green=outside lit)", overlay_pairs)

        # chi-squared histogram
        if len(chi2_list) > 0:
            plt.figure(figsize=(8,5))
            plt.hist(chi2_list, bins=40)
            plt.title("LBP chi-squared distances across shadow boundary (lower = more similar)")
            plt.xlabel("chi-squared distance")
            plt.ylabel("count")
            plt.tight_layout()
            plt.show()

        cv.waitKey(0)
        cv.destroyAllWindows()

    # features for ML
    # return raw chi-square distances plus some simple aggregates to feed a model
    chi2_arr = np.array(chi2_list, dtype=np.float32) if len(chi2_list) else np.array([], dtype=np.float32)
    feature_summary = {
        "chi2_count": int(chi2_arr.size),
        "chi2_mean": float(np.mean(chi2_arr)) if chi2_arr.size else 0.0,
        "chi2_std": float(np.std(chi2_arr)) if chi2_arr.size else 0.0,
        "chi2_p25": float(np.percentile(chi2_arr, 25)) if chi2_arr.size else 0.0,
        "chi2_p50": float(np.percentile(chi2_arr, 50)) if chi2_arr.size else 0.0,
        "chi2_p75": float(np.percentile(chi2_arr, 75)) if chi2_arr.size else 0.0,
    }

    return {
        "mask": mask, 
        "lbp": lbp,
        "L8": L8, 
        "chi2_distances": chi2_list,
        "patch_pairs": patch_pairs,
        "feature_summary": feature_summary,
        "features": features, #optional
    }

if __name__ == "__main__":
    analyze_texture("data/images/5.jpg", visualize=True)
