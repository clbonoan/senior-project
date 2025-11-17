# First Feature: edge detection and texture analysis
# Detects shadows in images and compares textures inside and outside shadows

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from texture_features import extract as extract_texture

# ----------------------------------------------------------
# LOCAL BINARY PATTERN (LBP) for texture analysis
# ----------------------------------------------------------
'''
look at each pixel and compare it to its 8 neighbors (top, bottom, left, right, diagonals);
if its neighbor is brighter, it is marked as 1, if darker then 0;
a binary pattern is created to describe the texture around a pixel
'''

def get_pixel(img, center, x, y):
    # check if neighbor pixels are brighter than the center pixel
    new_value = 0

    try:
        # if local neighbor pixel >= center pixel 
        if img[x][y] >= center:
            new_value = 1
    except:
        # return 0 if the pixel is at the edge of the image
        pass

    return new_value

def lbp_calculated_pixel(img, x, y): 
    # calculating LBP value for a single pixel by comparing it to 8 neighbors
    center = img[x][y]

    # create array of pixels based on the 8 neighbors
    val_arr = []
    val_arr.append(get_pixel(img, center, x-1, y-1)) # top left
    val_arr.append(get_pixel(img, center, x-1, y))   # top
    val_arr.append(get_pixel(img, center, x-1, y+1))  # top right
    val_arr.append(get_pixel(img, center, x, y+1))    # right
    val_arr.append(get_pixel(img, center, x+1, y+1))   # bottom right
    val_arr.append(get_pixel(img, center, x+1, y))    # bottom
    val_arr.append(get_pixel(img, center, x+1, y-1))    # bottom left
    val_arr.append(get_pixel(img, center, x, y-1))    # left
    
    # convert binary pattern values to decimal
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]    
    val = 0
    for i in range(len(val_arr)):
        val += val_arr[i] * power_val[i]
    return val

def lbp_map(gray):
    # creating an LBP map for the image after turning it to grayscale
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
    # calculating the average of neighboring pixels with box filter
    # good for reducing noise since it smooths
    return cv.boxFilter(arr, ddepth =- 1, ksize = (k, k), normalize = True)

def remove_small(mask, min_area = 800):
    # remove small noisy regions from the mask
    num, labels, stats, _ = cv.connectedComponentsWithStats(mask, connectivity = 8)
    keep = np.zeros_like(mask)
    
    # background is labeled 0
    # keep only the large regions
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
    # convert the sRGB color space to linear space for better shadow detection 
    # sRGB is for visual/display, linear for math operations
    x = x8.astype(np.float32) / 255.0       # x in [0,1], sRGB -> linear
    a = 0.055
    return np.where(x <= 0.04045, x/12.92, ((x + a)/(1 + a))**2.4)

def normalized_rgb(img_bgr):
    # normalize the RGB channels to help identify shadows through color consistency
    b, g, r = cv.split(img_bgr.astype(np.float32))

    # calculate sum of pixel values
    # total light intensity (similar to luminance) at each pixel location
    sum = r + g + b + 1e-6
    return r/sum, g/sum     # chroma ratios red and green since they are more stable than blue channel

def bgr_to_hsi_linear(img_bgr):
    # convert BGR to HSI color space (hue, saturation, intensity)
    # this is important to adjust brightness without affecting shadow's hue or saturation
    B, G, R = cv.split(img_bgr.astype(np.float32) / 255.0)

    # convert to linear space for calculations
    Rl = srgb_to_linear(R)
    Gl = srgb_to_linear(G)
    Bl = srgb_to_linear(B)

    # calculate the intensity (the average of rgb)
    I = (Rl + Gl + Bl) / 3.0

    # calculate the saturation (how colorful vs how gray)
    min_rgb = np.minimum(np.minimum(Rl, Gl), Bl)
    S = np.where(I > 1e-6, 1.0 - (min_rgb / (I + 1e-6)), 0.0)

    # hue not used since hue is not used in shadow detection 
    H = np.zeros_like(I, dtype=np.float32)
    return H, S, I

# ----------------------------------------------------------
# SHADOW MASK
# ----------------------------------------------------------
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
    based on paper by Uddin, Khanam, Khan, Deb, and Jo detailing color models HSI and YCbCr:
    1. chromatic attainment on S: Im = S - log(S + delta)
    2. intensity attainment: I' = I + beta * Y    (Y is from YCrCb color model)
    3. shadow if I' is highly saturated and S' (boosted) is low
    SUMMARY: shadow are dark regions that
    - have low saturation
    - stay dark after brightness is adjusted
    - maintain consistent color but just get darker
    '''


    # convert BGR to HSI in linear light (hue, saturation, intensity)
    _, S, I = bgr_to_hsi_linear(img_bgr)

    # find dark regions (regions of interest) using multiple window sizes
    meanI = [box_mean(I, w) for w in win_scales]
    darks = [(I < (kk * m)) for m, kk in zip(meanI, k_dark)]
    roi_dark = darks[0] | darks[1] | darks[2]
    
    if not np.any(roi_dark):
        print("roi_dark is empty → returning empty mask.")
        return np.zeros(I.shape, np.uint8), (I * 255).astype(np.float32)
    
    # chromatic attainment (1) 
    # Sm = S - np.log(S + delta)
    # SKIPPED chromatic attainment; used raw saturation with thresholds instead since shadows typically have low saturation in the image 

    # boost intensity using luminance (Y channel from YCrCb)
    # B, G, R = cv.split(srgb_to_linear(img_bgr))
    B, G, R = cv.split(img_bgr.astype(np.float32) / 255.0)
    Rl = srgb_to_linear(R * 255) / 255.0
    Gl = srgb_to_linear(G * 255) / 255.0
    Bl = srgb_to_linear(B * 255) / 255.0
    
    # common approximation to convert RGB to linear luminance (how bright each pixel appears to human eye)
    # coefficients (0.114, 0.587, 0.299) are based on sensitivity of the human eye to diff light wavelengths
    Y_lin = 0.114*Bl + 0.587*Gl + 0.299*Rl
    Iprime_raw = I + beta * Y_lin

    # normalize the boosted intensity I' by a high percentile inside dark region of interest (adaptive to scene/image)
    scale = float(np.percentile(Iprime_raw[roi_dark], 95))
    Iprime = np.clip(Iprime_raw / max(scale, 1e-6), 0.0, 1.0)

    # check chroma-consistency (checked since shadows dim/get darker but do not change color)
    nr, ng = normalized_rgb(img_bgr)
    mnr = box_mean(nr, 41)  # mean of normalized r
    mng = box_mean(ng, 41)  # mean of normalized g
    chroma_ok = (np.abs(nr - mnr) < dr) & (np.abs(ng - mng) < dg)

    # self-tuning thresholds (saturation and intensity) from region of interest percentiles
    S_thr = float(min(0.30, np.percentile(S[roi_dark], 40) + 0.02))
    Ip_thr = float(np.percentile(Iprime[roi_dark], 60))

    '''
    constraints for shadow mask based on if I' = 255 & S about 0;
    to make it adaptive to all images, a pixel is considered a shadow if it has:
    - low saturation (S below given threshold)
    - low intensity even after intensity boost (I' < I_threshold) -> (shadows stay dark after boost)
    - were already dark before intensity boost (roi_dark) -> (dark to begin with)
    - shadows do not change color even after boost (chroma_ok) -> (consistent color)
    '''
    mask = (roi_dark & chroma_ok & (S <= S_thr) & (Iprime <= Ip_thr)).astype(np.uint8) * 255

    # use morphological image processing to remove specks and fill in small holes (cleaning up the mask)
    if morph_open > 1:
        k1 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (morph_open, morph_open))
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, k1)
    if morph_close > 1:
        k2 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (morph_close, morph_close))
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, k2)
    mask = remove_small(mask, min_area=min_area)
    
    # return mask and luminance channel to reuse (considered V in paper)
    return mask, (I * 255).astype(np.float32)

# ----------------------------------------------------------
# TAMPER SCORE CALCULATION
# ----------------------------------------------------------
def calculate_tamper_score(chi2_list, mask, L8):
    '''
    tamper score between 0 to 1 based on texture comparisons of each pair of patches for each shadow component:
    0.0 = likely real shadows (textures match)
    0.1 = likely fake shadows (textures are very different)

    considering:
    - how different the textures are (chi-squared distances for similarity)
    - how many suspicious shadow regions there are
    - consistency across all shadows
    '''

    if len(chi2_list) == 0:
        return 0.0  # no shadows are found so assume real
    
    chi2_arr = np.array(chi2_list)

    # thresholds
    LOW_THRESHOLD = 0.10   # similar texture
    HIGH_THRESHOLD = 0.30   # very different texture

    # average texture difference for first score component
    mean_chi2 = float(np.mean(chi2_arr))
    if mean_chi2 <= LOW_THRESHOLD:
        score_mean = 0.0
    elif mean_chi2 >= HIGH_THRESHOLD:
        score_mean = 1.0
    else:
        # linear interpolation if mean is between thresholds
        score_mean = (mean_chi2 - LOW_THRESHOLD) / (HIGH_THRESHOLD - LOW_THRESHOLD)

    # max difference (worst case) for second score component
    # flag if even one shadow is very suspicious
    max_chi2 = float(np.max(chi2_arr))
    if max_chi2 <= LOW_THRESHOLD:
        score_max = 0.0
    elif max_chi2 >= HIGH_THRESHOLD:
        score_max = 1.0
    else:
        score_max = (max_chi2 - LOW_THRESHOLD) / (HIGH_THRESHOLD - LOW_THRESHOLD)

    # consistency of values for third score component
    # real shadows should have similar values
    # fake shadows might have mixed results (some match, some don't)
    std_chi2 = float(np.std(chi2_arr))
    if std_chi2 > 0.1:
        score_consistency = 0.3  # penalty for inconsistency
    else:
        score_consistency = 0.0
    
    # percentage of suspicious shadows for fourth score component
    # count how many shadows exceed the high threshold
    num_suspicious = int(np.sum(chi2_arr > HIGH_THRESHOLD))
    pct_suspicious = num_suspicious / len(chi2_arr)
    score_percentage = pct_suspicious
    
    # combine all components with weights
    # mean is most important, max catches extreme cases
    tamper_score = (
        0.40 * score_mean +        # average behavior
        0.30 * score_max +         # worst case
        0.15 * score_consistency + # how consistent
        0.15 * score_percentage    # how many are bad
    )
    
    # clamp score to [0, 1]
    tamper_score = max(0.0, min(1.0, tamper_score))
    
    return tamper_score        

# ----------------------------------------------------------
# TEXTURE COMPARISON AND HELPERS FOR TAMPER SCORE
# compare texture between shadow patches and nearby lit patches
# ----------------------------------------------------------
def chi2(a, b, eps=1e-9):
    # chi-squared distance between two histograms (measuring texture simularity)
    d = (a - b)
    s = (a + b) + eps
    return 0.5 * np.sum((d * d) / s)

# # helper functions for the tamper score (0...1)
# def clamp01(x):
#     # clamp a float value between 0 and 1
#     return max(0.0, min(1.0, float(x)))

# def contrast_norm(mean_in, mean_out, eps=1e-6):
#     # how much darker the inside is vs outside; normalized
#     return clamp01((mean_out - mean_in) / (mean_out + eps))

# def map_to_01(x, lo, hi):
#     # map x linearly from [lo,hi] to [0,1]
#     if hi <= lo:
#         return 0.0
#     # interpolate
#     y = (x - lo) / (hi - lo)
#     # clip 
#     return max(0.0, min(1.0, y))

def pairs_one_per_component(
    mask_u8, L8, lbp, gx, gy,
    min_area=300,            # keep consistent with your mask cleanup
    patch_size=25, in_offset=6, out_offset=9, max_step=30
):
    '''
    for each connected shadow component in mask_u8, pick one pair of patches to compare:
      - inside patch: near boundary but inside (stable, not on edge)
      - outside patch: just outside the boundary in the lit area
    return:
      chi2_list (distances): list[float]
      patch_pairs: [((x_in,y_in),(x_out,y_out), size)]
    '''
    h, w = L8.shape
    s = patch_size // 2
    chi2_list, patch_pairs = [], []

    # find all separate shadow components/regions
    num, labels, stats, _ = cv.connectedComponentsWithStats(mask_u8, connectivity=8)
    k3 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))

    def in_bounds(y, x):
        # check if patch fits in image
        return (y - s) >= 0 and (x - s) >= 0 and (y + s) < h and (x + s) < w

    def crop(img, y, x):
        # extract patch from image
        return img[y - s:y + s + 1, x - s:x + s + 1]

    # go through each shadow region
    for i in range(1, num):  # 0 is background
        area = stats[i, cv.CC_STAT_AREA]
        # skip small regions
        if area < min_area:
            continue

        # grab shadow component    
        comp = (labels == i).astype(np.uint8) * 255
        boundary = cv.morphologyEx(comp, cv.MORPH_GRADIENT, k3)
        
        if boundary.sum() == 0:
            continue

        # find a good point inside the shadow that is far from edges
        dist = cv.distanceTransform((comp > 0).astype(np.uint8), cv.DIST_L2, 5)
        yi_in, xi_in = np.unravel_index(np.argmax(dist), dist.shape)
        yi_in = int(np.clip(yi_in, s, h - s - 1))
        xi_in = int(np.clip(xi_in, s, w - s - 1))
        
        if not in_bounds(yi_in, xi_in):
            continue

        # find the closest boundary point
        by, bx = np.where(boundary > 0)
        # calculate the squared euclidean distance between a boundary point with coords bx, by and each point ->
        # in a set of points with coords xi_in, yi_in;
        # the index of the point with closest distance to bx, by is assigned to j
        j = int(np.argmin((by - yi_in)**2 + (bx - xi_in)**2))
        yb, xb = int(by[j]), int(bx[j])

        # find out which direction is outward (going toward lit area)
        g = np.array([gx[yb, xb], gy[yb, xb]], dtype=np.float32)
        nrm = float(np.linalg.norm(g))

        if nrm > 1e-3:
            # gradient direction
            nx, ny = g[0] / nrm, g[1] / nrm
        else:
            # use direction from center to boundary
            M = cv.moments(comp, binaryImage=True)
            # total area in shadow; if area is 0 (empty shadow), use bound point as center
            if abs(M["m00"]) < 1e-6:
                cx, cy = xb, yb
            else:
                # calculate actual center
                # average x = sum of all x values / num of pixels
                cx = M["m10"] / M["m00"] 
                cy = M["m01"] / M["m00"]
            v = np.array([xb - cx, yb - cy], dtype=np.float32)
            vn = np.linalg.norm(v)
            nx, ny = (v / vn) if vn > 1e-6 else (1.0, 0.0)

        # get the inside shadow patch
        yi_samp = int(np.clip(yb - in_offset * ny, s, h - s - 1))
        xi_samp = int(np.clip(xb - in_offset * nx, s, w - s - 1))
        
        if not in_bounds(yi_samp, xi_samp):
            continue

        # get the outside shadow patch
        step = out_offset
        yo = int(np.clip(yb + step * ny, s, h - s - 1))
        xo = int(np.clip(xb + step * nx, s, w - s - 1))
        
        # keep moving outward until outside the shadow
        while step < max_step and comp[yo, xo] > 0:
            step += 2
            yo = int(np.clip(yb + step * ny, s, h - s - 1))
            xo = int(np.clip(xb + step * nx, s, w - s - 1))
        
        if not in_bounds(yo, xo):
            continue

        # extract patches
        pinL = crop(L8,  yi_samp, xi_samp)
        poutL = crop(L8,  yo, xo)
        pinLBP = crop(lbp, yi_samp, xi_samp)
        poutLBP = crop(lbp, yo, xo)

        # make sure the inside patch is darker than outside
        m_in = float(np.mean(pinL))
        m_out = float(np.mean(poutL))

        if m_out <= m_in:
            # swap to keep outside brighter
            pinL, poutL = poutL, pinL
            pinLBP, poutLBP = poutLBP, pinLBP
            (xi_samp, yi_samp), (xo, yo) = (xo, yo), (xi_samp, yi_samp)
            m_in, m_out = m_out, m_in

        # compare the textures with chi-squared distasnce
        h_in,  _ = np.histogram(pinLBP.ravel(),  bins=256, range=(0, 256), density=True)
        h_out, _ = np.histogram(poutLBP.ravel(), bins=256, range=(0, 256), density=True)
        
        # calculate the chi-squared distance (lower = more similar)
        '''
        ideal range for chi-squared distance:
        - 0-0.2: very similar textures (likely real shadows)
        - 0.2-0.5: moderately similar (could be real shadows with some variation)
        - 0.5-1.0: different textures (suspicious, might be fake)
        - >1.0: very different textures (likely fake; different materials)
        for our detection:
        - <0.3: likely a shadow
        - 0.3-0.8: some texture difference and may need more investigation
        ->0.8: likely a fake shadow
        '''
        d = 0.5 * np.sum(((h_in - h_out) ** 2) / (h_in + h_out + 1e-9))

        chi2_list.append(float(d))
        x0_in, y0_in = xi_samp - s, yi_samp - s
        x0_out, y0_out = xo - s, yo - s
        patch_pairs.append(((x0_in, y0_in), (x0_out, y0_out), patch_size))

    return chi2_list, patch_pairs

# ----------------------------------------------------------
# CHOOSING THE IMAGE - MAIN ANALYSIS
# ----------------------------------------------------------
def analyze_texture(image_input, visualize=True, max_pairs_vis=200):
    '''
    analyze texture across shadow boundaries using LBP and return chi-square similarities
    - highlight the pairs of patches (one pair per shadow) that are compared
    - return raw chi-square distances for ML feature engineering
    '''
    if isinstance(image_input, str):
        img = cv.imread(image_input)
    else:
        img = image_input

    assert img is not None, f"Cannot read image: {image_input}"

    # detect shadows with shadow mask
    mask,L = make_shadow_mask(
        img, 
        beta = 0.8,  # trying 0.6-0.8 (street level) or 0.8-1.0 (aerial)
        win_scales = (21, 41, 81),
        k_dark = (0.92, 0.95, 0.98),
        dr=0.06, dg=0.06,
        morph_open = 3, 
        morph_close = 7, 
        min_area = 300
    )

    # outline of the mask (shadow boundaries)
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    mask_outline = cv.morphologyEx(mask, cv.MORPH_GRADIENT, k)

    # convert intensity to 8-bit
    L8 = np.uint8(np.clip(L*255, 0, 255))

    # calculate gradients to find outward direction of shadow
    gx = cv.Sobel(L8, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(L8, cv.CV_32F, 0, 1, ksize=3)

    # make LBP texture map 
    lbp = lbp_map(L8)
    lbp_vis = cv.normalize(lbp, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    # outline the shadow areas (using mask boundaries)
    lbp_color = cv.applyColorMap(lbp_vis, cv.COLORMAP_BONE)
    lbp_color[mask_outline > 0] = (255, 0, 0)

    # EXACTLY one pair per component
    chi2_list, patch_pairs = pairs_one_per_component(
        mask_u8=mask, 
        L8=L8, 
        lbp=lbp, 
        gx=gx, 
        gy=gy,
        min_area=300, 
        patch_size=25, 
        in_offset=6, 
        out_offset=9
    )

    print("\nTexture similarity scores (chi-squared distance):")
    print("Lower values = more similar texture = likely real shadow")
    print("Higher values = different texture = possible fake shadow")
    for i, d in enumerate(chi2_list, start=1):
        print(f"  Pair {i:02d}: {d:.4f}")

    # Calculate tamper score
    tamper_score = calculate_tamper_score(chi2_list, mask, L8)
    print(f"\n{'='*60}")
    print(f"TAMPER SCORE: {tamper_score:.3f}")
    print(f"{'='*60}")

    if tamper_score < 0.4:
        print("Assessment: LIKELY REAL - shadows appear authentic")
    elif tamper_score < 0.7:
        print("Assessment: SUSPICIOUS - some inconsistencies detected")
    else:
        print("Assessment: LIKELY FAKE - strong evidence of manipulation")


    # build feature measurements for ML training
    features = extract_texture(
        img,
        mask = mask,
        lbp = lbp,
        L8 = L8, 
        gx = gx,
        gy = gy,
        chi2_list = chi2_list,
    )

    if visualize:
        # show shadow mask
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

        # show lbp map with matplotlib
        plt.figure(figsize=(8, 6))
        plt.imshow(lbp_vis, cmap='gray')
        plt.contour(mask_outline > 0, colors='red', linewidths=0.5)
        plt.title("LBP with bounds in red")
        plt.show(block=False)
        plt.pause(0.001)
        plt.show()

        # patch pair rectangles (inside=red, outside=green)
        overlay_pairs = lbp_color.copy()
        vis_n = min(len(patch_pairs), max_pairs_vis)

        for i in range(vis_n):
            (x_in, y_in), (x_out, y_out), s = patch_pairs[i]
            # red box = inside shadow
            cv.rectangle(overlay_pairs, (x_in, y_in), (x_in + s, y_in + s), (0, 0, 255), 2)
            # green box = outside shadow (lit area)
            cv.rectangle(overlay_pairs, (x_out, y_out), (x_out + s, y_out + s), (0, 255, 0), 2)
            # label pairs
            cv.putText(overlay_pairs, str(i), (x_in, max(0, y_in - 3)), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1, cv.LINE_AA)
            cv.putText(overlay_pairs, str(i), (x_out, max(0, y_out - 3)), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1, cv.LINE_AA)                

        cv.namedWindow("Comparison Patches (red=inside shadow, green=outside shadow)", cv.WINDOW_NORMAL)
        cv.resizeWindow("Comparison Patches (red=inside shadow, green=outside shadow)", 1000, 750)
        cv.imshow("Comparison Patches (red=inside shadow, green=outside shadow)", overlay_pairs)

        # chi-squared histogram (similarity scores)
        if len(chi2_list) > 0:
            plt.figure(figsize=(8,5))
            plt.hist(chi2_list, bins=40)
            plt.title("Texture Similarity Scores - LBP chi-squared distances")
            plt.xlabel("Chi-squared distance (lower = more similar)")
            plt.ylabel("Number of shadow regions")
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
        "features": features,
    }

if __name__ == "__main__":
    analyze_texture("data/images/5.jpg", visualize=True)
