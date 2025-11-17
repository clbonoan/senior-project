# Second Feature - lighting consistency and ...
# Checks if shadows have consistent lighting

import cv2 as cv
import numpy as np
import io, contextlib

from lighting_features import extract as extract_lighting

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

# def linear_Y_from_bgr(img_bgr):
#     # compute linear luminance from BGR image
#     B, G, R = cv.split(srgb_to_linear(img_bgr))
#     # coefficients (0.114, 0.587, 0.299) are based on sensitivity of the human eye to diff light wavelengths
#     return (0.114*B + 0.587*G + 0.299*R).astype(np.float32)

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

# def unit_rgb_dir(img_bgr):
#     # turn each pixel's RGB into a pure color direction
#     # checks if two spots are the same color material even if one is in shadow and other is brighter
#     lin = srgb_to_linear(img_bgr)
#     n = np.linalg.norm(lin, axis=2, keepdims=True) + 1e-6
#     return lin/n

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
        self.strength_maps = []

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

    def add_strength_map(self, ishadow_vis):
        self.strength_maps.append(ishadow_vis.copy())

    def show_all(self, cols=3, size=(800, 600), fit_to_screen=True,
                 max_w=1920, max_h=1080):
        sets = [
            ("Shadow Masks", self.shadow_masks),
            ("Mask Overlays", self.mask_overlays),
            ("Shadow Components", self.components),
            ("Umbra vs Ring", self.shadow_highlight),
            ("Shadow Strength Maps", self.strength_maps),
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
# LIGHTING STRENGTH CONSISTENCY
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
  s_lighting = 1 - median_similatiry
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
def find_bright_nearby(img_bgr, shadow_mask, brightness, expand=4, color_diff=0.04):
    # expand shadow to get ring around it
    kernel = np.ones((3,3), np.uint8)
    expanded = cv.dilate(shadow_mask.astype(np.uint8), kernel, 
                        iterations=expand).astype(bool)
    
    # ring is expanded area minus shadow
    ring = expanded & (~shadow_mask) & (brightness > 0.001)
    if ring.sum() == 0:
        return None
    
    # get color direction
    color_dirs = get_rgb_direction(img_bgr)
    
    # find edge of shadow
    edge = shadow_mask & (~cv.erode(shadow_mask.astype(np.uint8), kernel, 1).astype(bool))
    if edge.sum() == 0:
        return ring
    
    # average color at edge
    avg_color = color_dirs[edge].mean(axis=0, keepdims=True)
    
    # check which ring pixels have similar color
    ring_indices = np.flatnonzero(ring.ravel())
    ring_colors = color_dirs.reshape(-1,3)[ring_indices]
    
    # cosine distance
    similarity = 1.0 - (ring_colors @ avg_color.T).ravel()
    
    # keep similar colors
    similar = np.zeros(ring.size, bool)
    similar[ring_indices[similarity <= color_diff]] = True
    result = similar.reshape(ring.shape)
    
    if result.sum() > 0:
        return result
    return ring

def features_one_shadow(
        img_bgr, 
        shadow_mask, 
        expand=4, 
        color_diff=0.04, 
        darkest_frac=0.5, 
        shrink=3,
        debug=False,
        win_name_prefix="shadow", 
        viz: DebugVisualizer | None = None,
    ):
    # for one shadow component, get the following:
    #   - umbra extraction
    #   - lit ring -> Y_N estimate for median
    #   - I_shadow approx equal to ((Y_S - Y_N)/Y_N) * Y_S over linear Y
    #   - 4D feature with mean, std dev, skew, kurtosis over umbra

    # calculate luminance (linear) (Y)
    brightness = get_brightness(img_bgr)

    # find dark part (umbra) of shadow
    dark_part = find_dark_part(shadow_mask, brightness, darkest_frac, shrink)
    if dark_part is None:
        return None

    # find bright area (ring) nearby
    bright_area = find_bright_nearby(img_bgr, shadow_mask, brightness, expand, color_diff)
    if bright_area is None or bright_area.sum() < 30:
        return None

    # get median brightness of bright area (Yn)
    bright_val = float(np.median(brightness[bright_area]))  # lit reference
    if not np.isfinite(bright_val) or bright_val < 1e-4:
        return None
    
    print(f"\nBright area median brightness: {bright_val:.4f}")

    # visualize in windows based on type of image
    if debug and viz is not None:
        eps = 1e-6
        # eq.12 (adjusted) in linear space
        # calculate shadow strength
        shadow_strength = ((brightness - bright_val) / (bright_val + eps)) * brightness  
        display_map = np.zeros_like(brightness, dtype=np.float32)
        display_map[shadow_mask] = shadow_strength[shadow_mask]
        
        # normalize for display
        vals = display_map[shadow_mask]
        if vals.size > 0:
            low, high = np.percentile(vals, [1, 99])
            if high > low:
                disp = np.clip(display_map, low, high)

        display_map = cv.normalize(display_map, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

        # create the overlay    (BGR)
        overlay = img_bgr.copy()
        overlay[dark_part] = [0, 0, 255]  # red
        overlay[bright_area]  = [0, 255, 0]  # green

        viz.add_strength_map(display_map)
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
        return None
    
    print(f"  Dark area brightness range: [{dark_values.min():.4f}, {dark_values.max():.4f}]")
    print(f"  Dark area median brightness: {np.median(dark_values):.4f}")    

    # shadow strength using paper's approach
    # use ratio of shadow to bright (lit intensity) (more stable than paper formula for diverse scenes)
    # strength_ratios = Ys / (Yn + 1e-6)
    strength_ratios = dark_values / (bright_val + 1e-6)

    print(f"  Strength ratios - min: {strength_ratios.min():.4f}, max: {strength_ratios.max():.4f}, median: {np.median(strength_ratios):.4f}")
    print(f"  Number of ratio values: {len(strength_ratios)}")


    # from paper: 
    # mean (avg strength), std dev (how much it varies),
    # ^ (ATTN: using median instead of mean for robustness; using IQR instead of std dev in case of skewed data)
    # skew (whether it's mostly dark or has brighter spots),
    # kurtosis (how "peaked" or "flat" the brightness distribution is)
    # extracting 4 features: median, IQR, skew, kurtosis)
    features = np.array([
        float(np.median(strength_ratios)),
        float(np.percentile(strength_ratios, 75) - np.percentile(strength_ratios, 25)),
        calc_skew(strength_ratios),
        calc_kurtosis(strength_ratios)
    ], np.float32) 

    print(f"  4 features extracted: {features}")

    return features  

# --------------------------------------------------
# tamper score
# -------------------------------------------------- 
def calculate_light_tamper_score(
        img_bgr, 
        mask,
        min_area=1200,
        expand=6,
        color_diff=0.08,
        darkest_frac=0.30,
        shrink=2,
        mask_kwargs=None,
        debug=False,
        viz : DebugVisualizer | None = None      
):
    # compute tamper score based on the following:
    # - build mask from texture.py
    # - extract features per-shadow
    # - similarity = cosine(Z-score features), use MEDIAN
    # - score = 1 - median_similarity (between 0 (real) and 1 (tampered))

    mask_kwargs = mask_kwargs or {}
    # get shadow mask using same function as texture.py
    shadow_mask = mask_from_texture(img_bgr, **mask_kwargs)


    # clean up mask
    shadow_mask = remove_small((shadow_mask > 0).astype(np.uint8) * 255, min_area=min_area)

    # separate into individual shadows
    num_shadows, labels = cv.connectedComponents(shadow_mask)

    # save debug info
    if debug and viz is not None:
        viz.add_mask(shadow_mask)
        viz.add_components_labels(labels)    

        # overlay of mask on original image
        overlay = img_bgr.copy()
        overlay[shadow_mask > 0] = [0, 0, 255]  # yellow for shadows
        blended = cv.addWeighted(img_bgr, 0.6, overlay, 0.4, 0)
        viz.add_mask_overlay(blended)        

    # analyze each shadow
    all_features = []
    for i in range(1, num_shadows):  # 0 = background
        # print(f"\n--- Analyzing Shadow {i} ---")

        current_shadow = (labels == i)
        features = features_one_shadow(
                img_bgr, current_shadow,
                expand=expand,
                color_diff=color_diff,
                darkest_frac=darkest_frac,
                shrink=shrink,
                debug=True,
                win_name_prefix=f"shadow {i}",
                viz=viz
        )

        if features is not None:
            all_features.append(features)

    # we want at least 2 shadows to compare
    if len(all_features) < 2:
        print(f"Found {int(num_shadows-1)} shadows, only {len(all_features)} usable. Need at least 2")
        return 0.0  # not enough shadows to compare

    # print(f"\n{'='*60}")
    # print(f"FEATURE COMPARISON PHASE")
    # print(f"{'='*60}")

    # if correlation coeff r is not close to 1, suspect tampering
    # num_shadows x 4 feature matrix
    feature_matrix = np.vstack(all_features)
    # print(f"\nFeature matrix shape: {feature_matrix.shape} (rows=shadows, cols=features)")
    # print(f"Feature matrix:")
    # for idx, feat_row in enumerate(feature_matrix):
    #     print(f"  Shadow {idx+1}: {feat_row}")
    
    # normalize each feature using robust stats with a modified z-score
    # median and median absolute deviation (MAD) instead of the mean and standard deviation
    medians = np.median(feature_matrix, axis=0)
    # use median instead of std dev for robustness
    mad = np.median(np.abs(feature_matrix - medians), axis=0)
    # no division by zero
    scale = np.where(mad > 1e-8, mad, 1.0)
    # z-score per feature to normalize values so big values don't outweigh small values
    normalized = (feature_matrix - medians) / scale   

    # print(f"\nMedians across all shadows: {medians}")
    # print(f"MAD (median absolute deviation): {mad}")
    # print(f"Scale factors: {scale}")
    # print(f"\nNormalized features:")
    # for idx, norm_row in enumerate(normalized):
    #     print(f"  Shadow {idx+1}: {norm_row}")

    # calculate distances between all pairs
    num = normalized.shape[0]
    distances = []
    # print(f"\n{'='*60}")
    # print(f"PAIRWISE DISTANCE CALCULATIONS")
    # print(f"{'='*60}")
    for i in range(num):
        for j in range(i+1, num):
            dist = np.linalg.norm(normalized[i] - normalized[j])
            distances.append(dist)
            # print(f"Distance between Shadow {i+1} and Shadow {j+1}: {dist:.4f}")

    if len(distances) == 0:
        return 0.0
    
    # convert distances to similarity scores
    # smaller distances = higher similarity
    # get median distance
    '''
    rule of thumb for lighting strength ratios:
    - if 
        - median_distance = 0 (all shadow identical), score = 0.0 (likely real)
        - median_distance = 
    '''
    distances = np.array(distances)
    median_distance = float(np.median(distances))

    # print(f"\n{'='*60}")
    # print(f"TAMPER SCORE CALCULATION")
    # print(f"{'='*60}")
    # print(f"All distances: {distances}")
    # print(f"Median distance: {median_distance:.4f}")

    # normalize distance to [0,1] score
    # lower median distance -> lower score (more consistent)
    score = float(np.clip(1.0 - np.exp(-median_distance / 2.0), 0.0, 1.0))

    print(f"Median distance: {median_distance:.3f}, Tamper score: {score:.3f}")

    # print(f"\nFormula: score = 1 - exp(-median_distance / 2.0)")
    # print(f"         score = 1 - exp(-{median_distance:.4f} / 2.0)")
    # print(f"         score = 1 - exp({-median_distance/2.0:.4f})")
    # print(f"         score = 1 - {np.exp(-median_distance/2.0):.4f}")
    # print(f"         score = {score:.4f}")
    
    # print(f"\n{'='*60}")
    # print(f"SUMMARY")
    # print(f"{'='*60}")
    print(f"Found {int(num_shadows-1)} shadows, {len(all_features)} usable")
    print(f"Final Tamper Score: {score:.4f}")
    # if score < 0.4:
    #     print(f"Interpretation: CONSISTENT lighting (likely real)")
    # elif score < 0.7:
    #     print(f"Interpretation: MODERATE inconsistency (suspicious)")
    # else:
    #     print(f"Interpretation: HIGH inconsistency (likely tampered)")
    # print(f"{'='*60}\n")

    # build feature measurements for ML training
    light_features = extract_lighting(
        img_bgr,
        mask,
        min_area,
        expand,
        color_diff,
        darkest_frac,
        shrink,
    )

    return score, light_features    

# ---------------------------------------------------
# CHOOSING IMAGE 
# ---------------------------------------------------
def analyze_lighting(image_path, show_debug=False):
    # print tamper score
    # similar to texture.py
    img = cv.imread(image_path, cv.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(image_path)

    if show_debug:
        viz = DebugVisualizer()
    else:
        None

    score, light_features = calculate_light_tamper_score(img, debug=show_debug, viz=viz)

    print(f"{score:.4f}")

    if show_debug and viz is not None:
        viz.show_all(cols=3, size=(600, 500))

    return score

if __name__ == "__main__":
    analyze_lighting("data/images/5.jpg", show_debug=True)