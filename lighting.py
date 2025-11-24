# Second Feature - lighting consistency and ...
# Checks if shadows have consistent lighting

import cv2 as cv
import numpy as np
import io, contextlib

# use same shadow mask as texture.py
from texture import make_shadow_mask as texture_make_shadow_mask

def mask_from_texture(img_bgr, suppress_prints=True, **kwargs): 
    # call texture.py's make_shadow_mask function to get the shadow mask
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
# SHADOW MASK HELPERS
# ----------------------------------------------------------
def srgb_to_linear(x8):
    # x in [0,1], sRGB -> linear
    x = x8.astype(np.float32) / 255.0
    a = 0.055
    result = np.where(x <= 0.04045, x/12.92, ((x + a)/(1 + a))**2.4)
    return result.astype(np.float32)

def box_mean(arr, k):
    # calculating the average of neighboring pixels with box filter
    # good for reducing noise since it smooths
    return cv.boxFilter(arr, ddepth=-1, ksize=(k, k), normalize=True)

def remove_small(mask, min_area=800):
    # remove small noisy regions from the mask
    num, labels, stats, _ = cv.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask)
    
    # background is labeled 0
    # keep only the large regions
    for i in range(1, num):
        # use the total area (# of pixels) of the component
        if stats[i, cv.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 255
    return keep

def get_brightness(img_bgr):
    # calculate brightness (luminance) from BGR image
    # convert to linear first
    linear = srgb_to_linear(img_bgr)
    B, G, R = cv.split(linear)
    # standard brightness formula
    brightness = 0.114*B + 0.587*G + 0.299*R
    return brightness.astype(np.float32)

def normalized_rgb(img_bgr):
    # normalize the RGB channels to help identify shadows through color consistency
    b, g, r = cv.split(img_bgr.astype(np.float32))

    # calculate sum of pixel values
    # total light intensity (similar to luminance) at each pixel location
    # add 1e-6 to avoid diving by zero
    sum = r + g + b + 1e-6
    return r/sum, g/sum     # chroma ratios red and green since they are more stable than blue channel

def get_rgb_direction(img_bgr):
    # get RGB direction for color matching
    linear = srgb_to_linear(img_bgr)
    length = np.linalg.norm(linear, axis=2, keepdims=True) + 1e-6
    return linear/length

# ----------------------------------------------------------
# CALCULATE SKEW AND KURTOSIS
# ----------------------------------------------------------
def calc_skew(values):
    # skew measures asymmetry (bias=False)
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    mean_val = values.mean()
    std_val = values.std(ddof=1)
    if std_val < 1e-12:
        return 0.0
    z_scores = (values - mean_val) / std_val
    return float(np.mean(z_scores**3))

def calc_kurtosis(values):
    # kurtosis measures "tailedness" or "peakedness" of the distribution
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return -3.0
    mean_val = values.mean()
    std_val = values.std(ddof=1)
    if std_val < 1e-12:
        return -3.0
    z_scores = (values - mean_val) / std_val
    kurt = float(np.mean(z_scores**4))
    return kurt - 3.0  # excess kurtosis

# ----------------------------------------------------------
# HELPERS FOR ML
# ----------------------------------------------------------
def sobel_xy(gray32): 
    gx = cv.Sobel(gray32, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(gray32, cv.CV_32F, 0, 1, ksize=3)
    return gx, gy

def boundary_from_mask(mask):
    se = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
    return cv.morphologyEx(mask, cv.MORPH_GRADIENT, se)

# ----------------------------------------------------------
# PUT SAME IMAGE TYPES IN ONE WINDOW
# ----------------------------------------------------------
def to_color(img):
    if img is None:
        return None
    if img.ndim == 2:
        return cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    return img

def make_image_grid(images, cols=3, size=None, pad=6, pad_color=(30,30,30)):
    '''
    make a grid from the images generated for visualization:
    - images: np.ndarray based on gray or BGR
    - cols: max columns used in grid
    - size (w,h): if there is no cell size, the first image dictates the size
    '''
    if not images:
        return None
    
    imgs = []
    for img in images:
        if img is None:
            continue
        if img.ndim == 2:
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        if size is None:
            size = (img.shape[1], img.shape[0])  # (w,h)
        imgs.append(cv.resize(img, size, interpolation=cv.INTER_AREA))
    
    if not imgs:
        return None
    
    num_images = len(imgs)
    cols = max(1, min(int(cols), num_images))
    rows = (num_images + cols - 1) // cols
    w, h = size
    
    # create canvas
    canvas_w = pad + cols * (w + pad)
    canvas_h = pad + rows * (h + pad)
    canvas = np.full((canvas_h, canvas_w, 3), pad_color, dtype=np.uint8)
    
    # place images
    k = 0
    for r in range(rows):
        for c in range(cols):
            if k >= num_images:
                break
            y = pad + r * (h + pad)
            x = pad + c * (w + pad)
            canvas[y:y+h, x:x+w] = imgs[k]
            k += 1
    return canvas

# resize to fit screen
def resize_to_fit(img, max_w=1920, max_h=1080):
    # scale the image to fit within the max width and height to keep its aspect ratio
    h, w = img.shape[:2]
    scale = min(max_w / max(1,w), max_h / max(1,h), 1.0)  # don't upscale past limits
    if scale < 1.0:
        img = cv.resize(img, (int(w*scale), int(h*scale)), interpolation=cv.INTER_AREA)
    return img

# class to collect debug visualizations
class DebugVisualizer:
    def __init__(self):
        self.shadow_masks = []
        self.mask_overlays = []
        self.components = []
        self.shadow_highlight = []
        #self.strength_maps = []

    def add_mask(self, mask):
        self.shadow_masks.append(mask.copy())

    def add_mask_overlay(self, overlay):
        self.mask_overlays.append(overlay.copy())

    def add_components_labels(self, labels):
        max_label = labels.max() + 1 if labels.size else 1
        colored = cv.applyColorMap(((labels.astype(np.float32)/max(1,max_label))*255).astype(np.uint8),
                                   cv.COLORMAP_JET)
        self.components.append(colored)

    def add_shadow_and_bright(self, overlay_bgr):
        self.shadow_highlight.append(overlay_bgr.copy())

    #def add_strength_map(self, ishadow_vis):
    #    self.strength_maps.append(ishadow_vis.copy())

    def show_all(self, cols=3, size=(800, 600), fit_to_screen=True,
                 max_w=1920, max_h=1080):
        sets = [
            ("Shadow Masks", self.shadow_masks),
            ("Mask Overlays", self.mask_overlays),
            ("Shadow Components", self.components),
            ("Umbra vs Ring", self.shadow_highlight),
            #("Shadow Strength Maps", self.strength_maps),
        ]
        for title, arr in sets:
            if not arr:
                continue
            grid = make_image_grid(arr, cols=cols, size=size)
            if grid is None:
                continue
            if fit_to_screen:
                grid = resize_to_fit(grid, max_w=max_w, max_h=max_h)

            cv.namedWindow(title, cv.WINDOW_NORMAL)
            cv.imshow(title, grid)
            # size window exactly to montage
            h, w = grid.shape[:2]
            cv.resizeWindow(title, w, h)
            
        cv.waitKey(0)
        cv.destroyAllWindows()

# -------------------------------------------------------
# LIGHTING STRENGTH CONSISTENCY ANALYSIS
# based on mathematical concepts found in "Exposing image forgery by detecting consistency of shadow" by Ke, Qin, Min, Zhang
# use of strength estimation of light source shadow model in paper
# -------------------------------------------------------
'''
check whether all shadows in image have similar brightness and lighting strength
pipeline:
1) shadow mask reused from texture.py
2) separate mask into sections; each section = one shadow
3) for each shadow region:
  - find the umbra (darkest area) inside the shadow
  - find small bright area right next to shadow on the same surface (comparing how much darker shadow is)
  - measure "shadow strength" based on how much light is blocked
      - formula: I_shadow = (approx equal) ((Y_shadow - Y_light) / Y_light) * Y_shadow
          where Y_shadow and Y_light are pixel brightness values
  - summarize section using four values: mean, std dev, skewness, kurtosis
4) compare all shadow regions and see how similar their stats are
  - if most are similar, image is likely to be real
5) turn similarity into tamper score (s_lighting):
  s_lighting = 1 - median_similarity
  lower score = consistent lighting (real)
  higher score = inconsistent lighting (tampered)
'''

# paper assumes k_i is 0 for the umbra region
# so shrink avoids penumbra and you keep darkest % inside region

# find darkest part of shadow (umbra) (darkest 30% of shadow is focused on to avoid penumbra)
def find_dark_part(shadow_mask, brightness, keep_fraction=0.5, shrink=3):
    # shrink mask to avoid edges
    if shrink > 0:
        kernel = np.ones((3,3), np.uint8)
        shadow_mask = cv.erode(shadow_mask.astype(np.uint8), kernel, 
                               iterations=shrink).astype(bool)
    
    shadow_pixels = brightness[shadow_mask]
    if shadow_pixels.size == 0:
        return None
    
    # keep only darkest pixels
    threshold = np.quantile(shadow_pixels, keep_fraction)
    dark_area = shadow_mask & (brightness <= threshold)
    
    if dark_area.sum() < 50:
        return None
    return dark_area

# paper says to compare the shadow area to a nearby non-shadow area of same surface

# find bright area near shadow with similar color
def find_bright_nearby(img_bgr, shadow_mask, brightness, expand=4, color_diff=0.03):
    # expand shadow to get ring around it
    kernel = np.ones((3,3), np.uint8)
    sm = (shadow_mask > 0)
    expanded = cv.dilate(sm.astype(np.uint8), kernel, iterations=expand).astype(bool)
    
    # ring is expanded area minus shadow
    ring = expanded & (~sm) & (brightness > 0.001)
    if ring.sum() == 0:
        return None
    
    # get color direction
    color_direction = get_rgb_direction(img_bgr)
    
    # use inside of shadow for color reference
    # erosion removes pixels from shadow's boundaries
    eroded = cv.erode(sm.astype(np.uint8), kernel, iterations=2).astype(bool)
    if eroded.sum() < 50:
        eroded = sm  # fallback
    
    # average color inside shadow
    avg_color = color_direction[eroded].mean(axis=0, keepdims=True)
    
    # check which ring pixels have similar color
    ring_indices = np.flatnonzero(ring.ravel())
    ring_colors = color_direction.reshape(-1,3)[ring_indices]
    
    # cosine distance
    # matrix multiplication @
    similarity = 1.0 - (ring_colors @ avg_color.T).ravel()
    
    # keep similar colors
    similar = np.zeros(ring.size, bool)
    similar[ring_indices[similarity >= (1.0 - color_diff)]] = True
    result = similar.reshape(ring.shape)
    
    # require minimum pixels or return ring if there aren't enough pixels
    if result.sum() > 100:
        return result
    return ring

def features_one_shadow(
        img_bgr, 
        shadow_mask, 
        brightness,
        gx, gy,
        expand=2, 
        color_diff=0.03, 
        darkest_frac=0.30, 
        shrink=2,
        debug=False,
        win_name_prefix="shadow", 
        viz: DebugVisualizer | None = None,
    ):
    '''
    for one shadow component, get the following:
      - umbra extraction
      - lit ring -> Y_N estimate for median
      - I_shadow approx equal to ((Y_S - Y_N)/Y_N) * Y_S over linear Y
      - 4D feature with mean, std dev, skew, kurtosis over umbra
    return (features_dict, log_ratio_median) or None if invalid
    '''
    area = int(np.count_nonzero(shadow_mask))

    # boundary alignment features (Sobel on L)
    boundary = boundary_from_mask((shadow_mask > 0).astype(np.uint8) * 255)
    ys, xs = np.where(boundary > 0)
    if ys.size > 0:
        gxs, gys = gx[ys, xs].astype(np.float32), gy[ys, xs].astype(np.float32)
        mag = np.sqrt(gxs*gys + gys*gys) + 1e-6
        nx, ny = gxs/mag, gys/mag
        ang = np.arctan2(ny, nx)
        C, S = float(np.mean(np.cos(ang))), float(np.mean(np.sin(ang)))
        R = float(np.sqrt(C*C + S*S))
        angle_std = float(np.sqrt(max(0.0, -2*np.log(R + 1e-6))))
    else:
        R, angle_std = 0.0, 0.0

    # calculate luminance (linear) (Y)
    brightness = get_brightness(img_bgr)

    # find dark part (umbra) of shadow
    dark_part = find_dark_part(shadow_mask, brightness, darkest_frac, shrink)
    if dark_part is None:
        # print(f"  Skipping shadow - could not find dark part")
        return None

    # find bright area (ring) nearby
    bright_area = find_bright_nearby(img_bgr, shadow_mask, brightness, expand, color_diff)
    if bright_area is None or bright_area.sum() < 100:
        # print(f"  Skipping shadow - not enough bright reference pixels")
        return None

    # get median brightness of bright area (Yn)
    bright_val = float(np.median(brightness[bright_area]))  # lit reference
    if not np.isfinite(bright_val) or bright_val < 1e-4:
        return None
    
    # print(f"\nBright area median brightness: {bright_val:.4f}")

    # visualize in windows based on type of image
    if debug and viz is not None:
        # eps = 1e-6
        # eq.12 (adjusted) in linear space
        # calculate shadow strength
        # shadow_strength = ((brightness - bright_val) / (bright_val + eps)) * brightness  
        # display_map = np.zeros_like(brightness, dtype=np.float32)
        # display_map[shadow_mask] = shadow_strength[shadow_mask]
        
        # normalize for display
        # vals = display_map[shadow_mask]
        # if vals.size > 0:
        #     low, high = np.percentile(vals, [1, 99])
        #     if high > low:
        #         disp = np.clip(display_map, low, high)

        # display_map = cv.normalize(display_map, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

        # create the overlay    (BGR)
        overlay = img_bgr.copy()
        overlay[dark_part] = [0, 0, 255]  # red
        overlay[bright_area]  = [0, 255, 0]  # green

        #viz.add_strength_map(display_map)
        viz.add_shadow_and_bright(overlay)

    # (ATTN: NOT AS ACCURATE FOR DIFF SCENES) equation 12 special case (shadow strength Ishadow) in linear space
    # Ishadow = ((Y - Yn) / Yn) * Y

    # stats of light strength distribution for each shadow
    # vals = Ishadow[umbra]
    # vals = vals[np.isfinite(vals)]
    # if vals.size < 50:
    #    return None
    
    # get brightness in dark part (Ys)
    # shadow strength estimation in umbra
    dark_values = brightness[dark_part]
    dark_values = dark_values[np.isfinite(dark_values)]
    if dark_values.size < 50:
        # print(f"  Skipping shadow: not enough dark pixels")
        return None
    
    # check: shadows should be darker than reference
    dark_median = np.median(dark_values)
    if dark_median >= bright_val * 0.9:
        # print(f"  Skipping shadow: shadow not darker than reference (dark={dark_median:.4f}, bright={bright_val:.4f})")
        return None
    
    # print(f"  Dark area brightness range: [{dark_values.min():.4f}, {dark_values.max():.4f}]")
    # print(f"  Dark area median brightness: {np.median(dark_values):.4f}")    

    # shadow strength using paper's approach
    # use ratio of shadow to bright (lit intensity) (more stable than paper formula for diverse scenes)
    # strength_ratios = Ys / (Yn + 1e-6)
    #strength_ratios = dark_values / (bright_val + 1e-6)

    # log ratio for shadow strength - more stable for different lighting conditions
    strength_ratios = np.log(np.maximum(dark_values, 1e-6)) - np.log(max(bright_val, 1e-6))

    sr_median = float(np.median(strength_ratios))
    sr_iqr = float(np.percentile(strength_ratios, 75) - np.percentile(strength_ratios, 25))
    sr_skew = calc_skew(strength_ratios)
    sr_kurtosis = calc_kurtosis(strength_ratios)

    # print(f"  Strength ratios - min: {strength_ratios.min():.4f}, max: {strength_ratios.max():.4f}, median: {np.median(strength_ratios):.4f}")
    # print(f"  Number of ratio values: {len(strength_ratios)}")

    '''
    from paper: 
    mean (avg strength), std dev (how much it varies),
    ^ (ATTN: using median instead of mean for robustness; using IQR instead of std dev in case of skewed data)
    skew (whether it's mostly dark or has brighter spots),
    kurtosis (how "peaked" or "flat" the brightness distribution is)
    extracting 4 features: median, IQR, skew, kurtosis)
    '''
    features = {
        'area': area,
        'mrl': R,
        'angle_std': angle_std,
        'umbra_frac': float(np.count_nonzero(dark_part)) / (area + 1e-6),
        'ring_to_area': float(np.count_nonzero(bright_area)) / (area + 1e-6),
        'Yn': bright_val,
        'sr_median': sr_median,
        'sr_iqr': sr_iqr,
        'sr_skew': sr_skew,
        'sr_kurtosis': sr_kurtosis,
    }

    # print(f"  4 features extracted: {features}")

    return features, sr_median  

# --------------------------------------------------
# FEATURE EXTRACTION FOR ML
# --------------------------------------------------
def extract_features(
        img_bgr,
        mask=None,
        min_area=1200,
        expand=2,
        color_diff=0.03,
        darkest_frac=0.30,
        shrink=2,
        mask_kwargs=None,
        debug=False,
        viz : DebugVisualizer | None = None                    
):
    '''
    extract ML features from lighting analysis; do not include tamper score
    return:
    - ml_features: dict of features for ML
    '''
    mask_kwargs = mask_kwargs or {}
    # get shadow mask using same function as texture.py
    if mask is None: 
        shadow_mask = mask_from_texture(img_bgr, **mask_kwargs)
    else:
        shadow_mask = mask

    # clean up mask
    shadow_mask = remove_small((shadow_mask > 0).astype(np.uint8) * 255, min_area=min_area)

    # separate into individual shadows
    num_shadows, labels = cv.connectedComponents(shadow_mask)

    # compute brightness and gradients once
    L = get_brightness(img_bgr)
    gx, gy = sobel_xy(L)

    # save debug info
    if debug and viz is not None:
        viz.add_mask(shadow_mask)
        viz.add_components_labels(labels)    

        # overlay of mask on original image
        overlay = img_bgr.copy()
        overlay[shadow_mask > 0] = [0, 0, 255]  # yellow for shadows
        blended = cv.addWeighted(img_bgr, 0.6, overlay, 0.4, 0)
        viz.add_mask_overlay(blended)        

    # initialize ML features dict
    ml_features = {}

    # scene-level features
    Lv = L[np.isfinite(L)]
    ml_features["image_median_brightness"] = float(np.median(Lv)) if Lv.size else 0.0
    ml_features["image_brightness_iqr"] = float(np.percentile(Lv,75)-np.percentile(Lv,25)) if Lv.size else 0.0
    ml_features["num_components_total"] = int(num_shadows - 1)

    # analyze each shadow
    per_shadow_features = []
    log_ratios = []

    for i in range(1, num_shadows):  # 0 = background
        # print(f"\n--- Analyzing Shadow {i} ---")

        current_shadow = (labels == i)
        result = features_one_shadow(
                img_bgr, current_shadow, L, gx, gy,
                expand=expand,
                color_diff=color_diff,
                darkest_frac=darkest_frac,
                shrink=shrink,
                debug=debug,
                win_name_prefix=f"shadow {i}",
                viz=viz
        )

        if result is not None:
            features, log_ratio = result
            per_shadow_features.append(features)
            if log_ratio is not None:
                log_ratios.append(log_ratio)

    usable = len(per_shadow_features)
    ml_features["num_components_usable"] = usable
    ml_features["usable_frac"] = float(usable / max(1, ml_features["num_components_total"]))
    

    # aggregate per-shadow features for ML
    def agg(name, values):
        arr = np.asarray(values, dtype=np.float32)
        ml_features[f"{name}_median"] = float(np.median(arr)) if arr.size else 0.0
        ml_features[f"{name}_iqr"] = float(np.percentile(arr,75)-np.percentile(arr,25)) if arr.size else 0.0

    if usable > 0:
        agg("comp_area", [f['area'] for f in per_shadow_features])
        agg("boundary_mrl", [f['mrl'] for f in per_shadow_features])
        agg("boundary_angle_std", [f['angle_std'] for f in per_shadow_features])
        agg("umbra_frac", [f['umbra_frac'] for f in per_shadow_features])
        agg("ring_to_area", [f['ring_to_area'] for f in per_shadow_features])
        agg("Yn", [f['Yn'] for f in per_shadow_features])
        agg("sr_median", [f['sr_median'] for f in per_shadow_features])
        agg("sr_iqr", [f['sr_iqr'] for f in per_shadow_features])
        agg("sr_skew", [f['sr_skew'] for f in per_shadow_features])
        agg("sr_kurtosis", [f['sr_kurtosis'] for f in per_shadow_features])
    else:
        for name in ["comp_area", "boundary_mrl", "boundary_angle_std", "umbra_frac", 
                     "ring_to_area", "Yn", "sr_median", "sr_iqr", "sr_skew", "sr_kurtosis"]:
            ml_features[f"{name}_median"] = 0.0
            ml_features[f"{name}_iqr"] = 0.0

    # cross-shadow consistency features for ML
    if usable >= 2:
        # num_shadows x 4 feature matrix
        #feature_matrix = np.vstack(all_features)
        feature_matrix = np.array([[f['sr_median'], f['sr_iqr'], f['sr_skew'], f['sr_kurtosis']]
                                   for f in per_shadow_features], dtype=np.float32)

        # pairwise distances in Z-space (all 4 features)
        med = np.median(feature_matrix, axis=0)
        mad = np.median(np.abs(feature_matrix - med), axis=0)
        scale = np.where(mad > 1e-8, mad, 1.0)
        Z = (feature_matrix - med) / scale

        dists = []
        for a in range(Z.shape[0]):
            for b in range(a+1, Z.shape[0]):
                dists.append(float(np.linalg.norm(Z[a]-Z[b])))
        dists = np.asarray(dists, dtype=np.float32)

        ml_features["sr_dist_median"] = float(np.median(dists))
        ml_features["sr_dist_mean"] = float(np.mean(dists))
        ml_features["sr_dist_p75"] = float(np.percentile(dists, 75))

        median_ratios = feature_matrix[:, 0]        
        # robust outlier removal using MAD (median absolute deviation)
        med_ratio = np.median(median_ratios)
        mad_ratio = np.median(np.abs(median_ratios - med_ratio))
        
        # MAD-based threshold (2.5 * MAD is common for outlier detection)
        # scale MAD to approximate std dev: std is about 1.4826 * MAD
        # use MAD since it is less sensitive to outliers than std
        if mad_ratio > 1e-6:
            threshold = 2.5 * 1.4826 * mad_ratio
            inliers = median_ratios[np.abs(median_ratios - med_ratio) <= threshold]
        else:
            inliers = median_ratios
        
        if len(inliers) < 2:
            inliers = median_ratios

        # use IQR (interquartile range) for spread since it's more robust than min/max
        q75 = np.percentile(inliers, 75)
        q25 = np.percentile(inliers, 25)
        iqr = q75 - q25
        
        # also compute trimmed range for comparison
        spread = inliers.max() - inliers.min()

        ml_features["sr_log_ratio_iqr"] = float(iqr)
        ml_features["sr_log_ratio_spread"] = float(spread)
        ml_features["sr_num_inliers"] = int(len(inliers))
        ml_features["sr_confidence"] = float(len(inliers) / len(median_ratios))

        # use IQR for scoring since real shadows typically have IQR < 0.5
        # Scale: IQR 0.3 -> score 0.12, IQR 0.8 -> score 0.32, IQR 2.5 -> score 0.6
        score = float(np.clip(iqr / 2.5, 0.0, 1.0))

        # calculate confidence level based on number of usable shadows after outliers are removed
        confidence = len(inliers) / max(len(median_ratios), 1)
        # print(f"Ratios after outlier removal: {len(inliers)}/{len(median_ratios)}")
        # print(f"IQR: {iqr:.3f}, Trimmed spread: {spread:.3f}")
        # print(f"Confidence: {confidence:.2f} ({len(inliers)} inliers / {len(median_ratios)} total)")

    else:
        ml_features["sr_dist_median"] = 0.0
        ml_features["sr_dist_mean"] = 0.0
        ml_features["sr_dist_p75"] = 0.0
        ml_features["sr_log_ratio_iqr"] = 0.0
        ml_features["sr_log_ratio_spread"] = 0.0
        ml_features["sr_num_inliers"] = 0
        ml_features["sr_confidence"] = 0.0
        #score = 0.0  # uncertain

    print(f"Found {int(num_shadows-1)} shadows, {usable} usable")
    print(f"Rule-Based Tamper Score: {score:.4f}")

    return ml_features    

# --------------------------------------------------
# TAMPER SCORE
# -------------------------------------------------- 
def calculate_light_tamper_score(ml_features):
    '''
    compute tamper score based on the following (can be grabbed from ML features):
    - build mask from texture.py
    - extract features per-shadow
    - similarity = cosine(Z-score features), use MEDIAN
    - score = 1 - median_similarity (between 0 (real) and 1 (tampered))
    return: 
    score - float (0-1), rule-based tamper score
    ml_features: dict of features for ML
    '''
    usable = ml_features.get("num_components_usable", 0)

    if usable < 2:
        return 0.0

    # use IQR for scoring since real shadows typically have IQR < 0.5
    iqr = ml_features.get("sr_log_ratio_iqr", 0.0)
    # Scale: IQR 0.3 -> score 0.12, IQR 0.8 -> score 0.32, IQR 2.5 -> score 0.6
    score = float(np.clip(iqr / 2.5, 0.0, 1.0))


    # print(f"Found {int(num_shadows-1)} shadows, {usable} usable")
    # print(f"Rule-Based Tamper Score: {score:.4f}")

    return score

# ---------------------------------------------------
# CHOOSING IMAGE 
# ---------------------------------------------------
def analyze_lighting(image_path, show_debug=False, compute_tamper_score=True):
    # print tamper score
    # similar to texture.py
    img = cv.imread(image_path, cv.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(image_path)

    if show_debug:
        viz = DebugVisualizer()
    else:
        viz = None

    mask = mask_from_texture(img)

    # extract ML features
    features = extract_features(img, mask, debug=show_debug, viz=viz)

    # calculate tamper score (rule-based)
    tamper_score = None
    if compute_tamper_score:
        tamper_score = calculate_light_tamper_score(features)
        print(f"\n{'='*60}")
        print(f"LIGHTING TAMPER SCORE: {tamper_score:.3f}")
        print(f"{'='*60}")

    # show visualization
    if show_debug and viz is not None:
        viz.show_all(cols=3, size=(600, 500))

    # build result dict
    result = {
        "features": features,
        "mask": mask,
    }

    if tamper_score is not None:
        result["tamper_score"] = tamper_score

    return result

if __name__ == "__main__":
    result = analyze_lighting("data/images/17-edited.jpg", show_debug=True)
    print("\nExtracted features:", result["features"])