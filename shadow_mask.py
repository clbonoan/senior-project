# shadow detection and mask using color space analysis
# important: normalize values to rescale them to a common range (makes for data consistency and fair comparisons)

import numpy as np
import cv2
from skimage.color import rgb2lab, lab2lch
from skimage.filters import threshold_multiotsu, threshold_otsu
#from skimage.morphology import disk, closing
from skimage.exposure import rescale_intensity
from scipy.ndimage import uniform_filter
from scipy import ndimage

# ----------------------------------------------------------
# ORIGINAL SHADOW MASK HELPERS
# ----------------------------------------------------------
def box_mean(arr, k):
    '''
    smooths image by averaging nearby pixels; good for reducing noise without blurring edges too much
    k = box size ; larger k means more smoothing
    '''
    return cv2.boxFilter(arr, ddepth =- 1, ksize = (k, k), normalize = True)

def remove_small(mask, min_area = 800):
    '''
    remove small noisy regions from the mask

    finds connected regions in the mask, counts pixels in each region, and keeps those larger than min_area
    '''
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity = 8)
    # start with an empty mask
    keep = np.zeros_like(mask)
    
    # background is labeled 0, so skip; loop through each found region/component
    # keep only the large regions
    for i in range(1, num):
        # use the total area (# of pixels) of the component
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            keep[labels == i] = 255

    return keep

# RGB to HSI conversion
def bgr_to_hsi(img_bgr):
    '''
    BGR to HSI (hue, saturation, intensity) color space:
    - hue: the actual color (red, blue, green, etc.)
    - saturation: how colorful vs gray
    - intensity: how bright

    rule of thumb for shadows:
    - shadows affect intensity but not much of the hue
    '''
    # normalize the image to 0-1 range
    img_bgr = img_bgr.astype(np.float32) / 255.0
    B, G, R = cv2.split(img_bgr)
    
    # intensity = average of RGB (this is the brightness)
    I = (R + G + B) / 3.0
    
    # saturation = how much color there is vs gray
    # add 1e-6 to avoid dividing by zero
    min_rgb = np.minimum(np.minimum(R, G), B)
    S = np.where(I > 1e-6, 1 - (min_rgb / (I + 1e-6)), 0)
    
    # hue (not needed for shadow detection)
    numerator = 0.5 * ((R - G) + (R - B))
    denominator = np.sqrt((R - G)**2 + (R - B) * (G - B)) + 1e-6
    theta = np.arccos(np.clip(numerator / denominator, -1, 1))
    # adjust hue based on whether blue > green (B>G)
    H = np.where(B <= G, theta, 2 * np.pi - theta)
    H = H * 180 / np.pi  # convert from radians to degrees
    
    return H, S, I

def srgb_to_linear(img):
    '''
    convert the sRGB color space to linear space for better shadow detection 
    sRGB is for visual/display, linear for math operations
    '''
    # normalize to 0-1
    x = img.astype(np.float32) / 255.0 
    # normalize only if we're in 0..255 domain
    if x.max() > 1.0:
        x = x / 255.0
    # sRGB -> linear conversion formula (two different formulas for dark vs bright pixels)
    a = 0.055
    linear = np.where(
        x <= 0.04045,
        x / 12.92,   # for dark pixels
        ((x + a)/(1 + a)) ** 2.4  # for bright pixels
    )
    return linear

def normalized_rgb(img_bgr):
    '''
    normalize the RGB channels to help identify shadows through color consistency
    - divide each channel by total brightness to get pure color without brightness affecting it

    - shadows change brightness (intensity) but not color ratios so normalized RGB should be more stable for shadow detection
    '''
    b, g, r = cv2.split(img_bgr.astype(np.float32))

    # calculate sum of all pixel values = total light intensity (similar to luminance) at each pixel location
    sum = r + g + b + 1e-6

    # normalized red and green since they are more stable than blue channel (blue is nosier)
    return r/sum, g/sum     

def bgr_to_hsi_linear(img_bgr):
    '''
    BGR to HSI in linear color space
    - this is important to adjust brightness without affecting shadow's hue or saturation
    - HSI color space is better for shadows, and linear RGB is better for calculations
    '''
    B, G, R = cv2.split(img_bgr.astype(np.float32))

    # convert each channel to linear space for calculations
    Rl = srgb_to_linear(R)
    Gl = srgb_to_linear(G)
    Bl = srgb_to_linear(B)

    # calculate the intensity (the average of rgb) on the linear values
    I = (Rl + Gl + Bl) / 3.0

    # calculate the saturation (how colorful vs how gray)
    min_rgb = np.minimum(np.minimum(Rl, Gl), Bl)
    S = np.where(I > 1e-6, 1.0 - (min_rgb / (I + 1e-6)), 0.0)

    # hue not used since hue is not used in shadow detection; made zero 
    H = np.zeros_like(I, dtype=np.float32)

    return H, S, I

def get_brightness(img_bgr):
    '''
    calculate brightness (luminance) from BGR image
    - standard formula: Y = 0.299*R + 0.587*G + 0.114*B
    - based on human perception: human eye is more sensitive to green light, less to red, least to blue
    - coefficient weights accurately reflect human perception of brightness for the specific colors

    convert to linear space, then do a weighted average
    '''
    # convert to linear first
    linear = srgb_to_linear(img_bgr)
    B, G, R = cv2.split(linear)
    # weighted sum = standard brightness formula
    brightness = 0.114*B + 0.587*G + 0.299*R

    return brightness.astype(np.float32)

def get_rgb_direction(img_bgr):
    '''
    RGB direction vector with normalized RGB
    - each pixel's rgb as a 3d vector, so normalize the vectors to unit lengths
    - used to compare colors independent of their brightness
    '''
    # get RGB direction for color matching
    linear = srgb_to_linear(img_bgr)

    # length of RGB vector at each pixel
    length = np.linalg.norm(linear, axis=2, keepdims=True) + 1e-6

    # normalize by dividing by length
    return linear/length

# ----------------------------------------------------------
# PREPROCESSING - GAUSSIAN BLUR AND LAPLACIAN OF GAUSSIAN
# ----------------------------------------------------------
def preprocess_for_shadow(
    img_bgr,
    gauss_ksize=2,
    gauss_sigma=0.5,
    log_ksize=3,
    log_sigma=0.0,
    log_alpha=0.3     
):
    '''
    applying preprocessing techniques of Gaussian blur and LoG to reduce noise and emphasize edges
    - gaussian blur: smooths noise
    - laplacian of gaussian (LoG) - finds edges

    what i've tested with the parameters:
    - higher gauss_sigma added more blur but more shadows disappeared in mask
    - lower log_alpha = less edge info added back
    '''
    # gaussian blur on the color image
    # kernel size must be odd and at least 3
    if gauss_ksize < 3:
        gauss_ksize = 3
    if gauss_ksize % 2 == 0:
        gauss_ksize += 1
    img_blur = cv2.GaussianBlur(img_bgr, (gauss_ksize, gauss_ksize), gauss_sigma)

    # LoG applied to grayscale for edge detection
    gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    # adding more blur before laplacian if needed
    if log_sigma > 0.0:
        if log_ksize < 3:
            log_ksize = 3
        if log_ksize % 2 == 0:
            log_ksize += 1
        gray_for_log = cv2.GaussianBlur(gray, (log_ksize, log_ksize), log_sigma)
    else:
        gray_for_log = gray

    # laplacian on the (pre)blurred gray to find edges
    lap = cv2.Laplacian(gray_for_log, cv2.CV_32F, ksize=log_ksize if log_ksize >= 1 else 3)

    gray_float = gray.astype(np.float32)

    # normalize lap to a similar scale as image values
    # scale laplacian roughly into 8-bit
    lap_norm = lap / (np.max(np.abs(lap)) + 1e-6) * 127.0

    # add edge info back to grayscale image
    gray_enhanced = gray_float + log_alpha * lap_norm

    # clip back to [0,255]
    gray_enhanced = np.clip(gray_enhanced, 0, 255).astype(np.uint8)

    # map enhanced gray back into color BGR (want to keep the colors but use the new brightness)
    # compute ratio between enhanced and original luminance then apply to each channel
    # calculate how much brightness changed
    old_brightness = gray.astype(np.float32) + 1.0  # add 1 to avoid dividing by 0
    new_brightness = (gray_enhanced.astype(np.float32) + 1.0) / old_brightness
    brightness_ratio = new_brightness / old_brightness
    
    # limit extreme changes (like avoiding weird colors)
    brightness_ratio = np.clip(brightness_ratio, 0.25, 4.0)  

    # apply the ratio to each color channel
    img_preprocessed = img_blur.astype(np.float32)
    img_preprocessed[..., 0] *= brightness_ratio     # blue channel
    img_preprocessed[..., 1] *= brightness_ratio     # green channel
    img_preprocessed[..., 2] *= brightness_ratio     # red channel

    # clip and convert back to uint8
    img_preprocessed = np.clip(img_preprocessed, 0, 255).astype(np.uint8)

    return img_preprocessed    

# # ----------------------------------------------------------
# # OLD SHADOW MASK (NOT BEING USED)
# # ----------------------------------------------------------
# # mask to further distinguish shadows
# def make_shadow_mask(
#     img_bgr,
#     beta = 0.8,    # projection weight of Y into I' (0 < beta <= 1)
#     win_scales = (21, 41, 81),    # local mean windows (pixels)
#     k_dark = (0.92, 0.95, 0.98),    # darker than local mean factors
#     dr = 0.06, dg = 0.06,   # chroma-consistency tolerance in normalized RGB
#     morph_open = 3,
#     morph_close = 7,
#     min_area = 400,
# ): 
#     '''
#     based on paper by Uddin, Khanam, Khan, Deb, and Jo detailing color models HSI and YCbCr:
#     1. chromatic attainment on S: Im = S - log(S + delta)
#     2. intensity attainment: I' = I + beta * Y    (Y is from YCrCb color model)
#     3. shadow if I' is highly saturated and S' (boosted) is low
#     SUMMARY: shadow are dark regions that
#     - have low saturation
#     - stay dark after brightness is adjusted
#     - maintain consistent color but just get darker
#     '''

#     '''
#     IMPROVEMENTS THAT NEED TO BE MADE:
#     - handle extreme brightness and darkness
#     - parameter tuning based on image stats
#     - refine to catch missed shadows
#     - more robust threshold selections 
#     '''
#     # apply preprocessing to image before masking
#     img_bgr = preprocess_for_shadow(img_bgr)

#     # convert BGR to HSI in linear light (hue, saturation, intensity)
#     _, S, I = bgr_to_hsi_linear(img_bgr)

#     # find dark regions (regions of interest) using multiple window sizes
#     meanI = [box_mean(I, w) for w in win_scales]
#     darks = [(I < (kk * m)) for m, kk in zip(meanI, k_dark)]
#     roi_dark = darks[0] | darks[1] | darks[2]
    
#     if not np.any(roi_dark):
#         print("roi_dark is empty → returning empty mask.")
#         return np.zeros(I.shape, np.uint8), (I * 255).astype(np.float32)
    
#     # chromatic attainment (1) 
#     # Sm = S - np.log(S + delta)
#     # SKIPPED chromatic attainment; used raw saturation with thresholds instead since shadows typically have low saturation in the image 

#     # boost intensity using luminance (Y channel from YCrCb)
#     # B, G, R = cv.split(srgb_to_linear(img_bgr))
#     B, G, R = cv.split(img_bgr.astype(np.float32))
#     Rl = srgb_to_linear(R)
#     Gl = srgb_to_linear(G)
#     Bl = srgb_to_linear(B)
    
#     # common approximation to convert RGB to linear luminance (how bright each pixel appears to human eye)
#     # coefficients (0.114, 0.587, 0.299) are based on sensitivity of the human eye to diff light wavelengths
#     Y_lin = 0.114*Bl + 0.587*Gl + 0.299*Rl
#     Iprime_raw = I + beta * Y_lin

#     # normalize the boosted intensity I' by a high percentile inside dark region of interest (adaptive to scene/image)
#     scale = float(np.percentile(Iprime_raw[roi_dark], 95))
#     Iprime = np.clip(Iprime_raw / max(scale, 1e-6), 0.0, 1.0)

#     # check chroma-consistency (checked since shadows dim/get darker but do not change color)
#     nr, ng = normalized_rgb(img_bgr)
#     mnr = box_mean(nr, 41)  # mean of normalized r
#     mng = box_mean(ng, 41)  # mean of normalized g
#     chroma_ok = (np.abs(nr - mnr) < dr) & (np.abs(ng - mng) < dg)

#     # self-tuning thresholds (saturation and intensity) from region of interest percentiles
#     S_thr = float(min(0.30, np.percentile(S[roi_dark], 40) + 0.02))
#     Ip_thr = float(np.percentile(Iprime[roi_dark], 60))

#     '''
#     constraints for shadow mask based on if I' = 255 & S about 0;
#     to make it adaptive to all images, a pixel is considered a shadow if it has:
#     - low saturation (S below given threshold)
#     - low intensity even after intensity boost (I' < I_threshold) -> (shadows stay dark after boost)
#     - were already dark before intensity boost (roi_dark) -> (dark to begin with)
#     - shadows do not change color even after boost (chroma_ok) -> (consistent color)
#     '''
#     mask = (roi_dark & chroma_ok & (S <= S_thr) & (Iprime <= Ip_thr)).astype(np.uint8) * 255

#     # use morphological image processing to remove specks and fill in small holes (cleaning up the mask)
#     if morph_open > 1:
#         k1 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (morph_open, morph_open))
#         mask = cv.morphologyEx(mask, cv.MORPH_OPEN, k1)
#     if morph_close > 1:
#         k2 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (morph_close, morph_close))
#         mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, k2)
#     mask = remove_small(mask, min_area=min_area)
    
#     # return mask and luminance channel to reuse (considered V in paper)
#     return mask, (I * 255).astype(np.float32)  

# --------------------------------------------------------------------
# NEW SHADOW MASK - MAIN SHADOW DETECTION
# --------------------------------------------------------------------
def detect_shadows_skimage(img, morph_size=9):
    '''
    using LAB/LCH color space

    idea: shadows have low lightness (brightness) and low chroma (gray-like)
    color spaces:
    - LAB: L=lightness, A=green-red, B=blue-yellow
    - LCH: L=lightness, C=chroma (colorfulness), H=hue

    LCH should be better because chroma (C) measures the colorfulness

    function:
    - run image through preprocessing (gaussian blur and LoG)
    - convert to LCH color space
    - find pixels with low lightness (L) and low chroma (C)
    - use multi-otsu to automatically find the threshold
        - multi-otsu is used because it automatically finds 3 classes: bright, medium, and dark
        - note: we keep only the darkest class as shadows
    '''
    # more accurate shadow masking using scikit-image + opencv

    # load ; comment this part out if not testing shadow_mask.py only
    # img_bgr = cv2.imread(img)
    # if img_bgr is None:
    #     raise ValueError("Image could not be loaded.")
    
    # run image through preprocess to reduce noise
    img_preprocessed = preprocess_for_shadow(img)

    # convert bgr to rgb
    img_rgb = cv2.cvtColor(img_preprocessed, cv2.COLOR_BGR2RGB)

    # convert to float [0,1] to normalize for skimage
    img_normalized = img_rgb.astype(np.float32) / 255.0
    img_normalized = np.nan_to_num(img_normalized, nan=0.0, posinf=1.0, neginf=0.0)

    # convert from RGB to LAB to LCH (skimage is more accurate than opencv)
    # LAB is "perceptually uniform" - distance between two colors corresponds to the amount of change humans can perceive
    lab = rgb2lab(img_normalized)
    # LCH is a cylindrical representation of LAB - chroma and hue directly relate to intensity and color type
    lch = lab2lch(lab)

    # get individual lch channels
    # [:, :, 0] (example) means select all rows, select all columns, get element at index 0 (slicing a 3d array)
    L = lch[:, :, 0]    # lighting (value from 0 to 100)
    C = lch[:, :, 1]    # chroma (how colorful)
    H = lch[:, :, 2]    # hue (color angle)

    # normalize L and C to 0-1 range for comparison
    Ln = rescale_intensity(L, out_range=(0,1))
    Cn = rescale_intensity(C, out_range=(0,1))

    ''' 
    shadows tend to have:
    - low lightness (Ln): (1-Ln) is high for dark areas
    - low chroma (Cn): (1-Cn) is high for grayish areas
    so calculating shadow score:
    - high score = likely to be a shadow
    - multiply the values since both must be true for high shadow score
    '''     
    # score = dark * low chroma
    shadow_score = (1-Ln) * (1-Cn)

    # threshold automatically - multi-otsu gives 3 cluster thresholds: light, mid, shadow
    # divides pixels into 3 classes based on shadow score
    thresholds = threshold_multiotsu(shadow_score, classes=3)

    # highest shadow scores make up the darkest class/cluster
    # thresholds[0] = light/medium boundary
    # threshold[1] = medium/dark boundary
    # use darkest class only
    t_shadow = thresholds[1]

    # create the binary mask
    shadow_mask = (shadow_score > t_shadow).astype(np.uint8)

    # NOTE: i tried to add morphological closing here but it made shadows mask too big in this step
    # clean with morphological operations
    # selem = disk(morph_size)
    # shadow_mask = closing(shadow_mask, selem)

    return shadow_mask.astype(np.uint8), img_preprocessed

def refine_shadow_mask_local(shadow_mask, img_bgr,
                             smooth_kernel=21,
                             rel_dark_min=0.01,
                             max_keep_ratio=0.9,
                             sat_clip=0.30):
    '''
    refine an initial shadow mask using local relative darkness and saturation

    issue: initial mask would include dark objects that look like shadows but aren't
    - so i wanted to look at local brightness differences based on the initial mask (shadows are darker than their surroundings)

    steps:
    - calculate local average brightness
    - find pixels darker than their nearby areas (neighbors)
    - use otsu on this relative darkness (otsu finds single threshold to separate image into two classes: foreground and background)
    - filter out saturated pixels that may be colored objects

    parameters tuned throughout testing:
    - smooth_kernel: size of neighborhood to compare to (originally was 41)
    - rel_dark_min: minimum local darkness difference to even be considered
    - max_keep_ratio: keep at most this percentage of the original masked pixels
    - sat_clip: HSV saturation (how colorful vs gray) above this value is rejected (considered not a shadow)
    '''

    # convert and get L (lightness) from LAB color space
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    lab = rgb2lab(img_rgb)
    # get lightness in range from 0 to 100
    L = lab[:, :, 0]        
    # normalize value between 0-1
    Ln = rescale_intensity(L, out_range=(0, 1))

    # calculate local average brightness
    # uniform_filter is used to calculate the mean over a given window size
    local_brightness = uniform_filter(Ln, size=smooth_kernel)

    # calculate how much darker each pixel is than its neighborhood
    # value > 0 when pixel is darker than local average (surroundings)
    relative_dark = local_brightness - Ln 

    # get the relative darkness values from initial shadow mask
    mask_bool = (shadow_mask.astype(np.uint8) > 0)

    # only consider possible pixels inside original mask
    relative_vals = relative_dark[mask_bool]

    # edge case - if mask is too small, just return the mask as it is
    if relative_vals.size < 50:
        return shadow_mask

    # filter out pixels that aren't much darker
    relative_vals = relative_vals[relative_vals > rel_dark_min]

    # if there are no pixels that are significantly darker, there are probably no real shadows
    if relative_vals.size < 50:
        return np.zeros_like(shadow_mask, dtype=np.uint8)

    # find otsu threshold on local darkness within the mask
    # automatically finds the best threshold to separate foreground and background (shadows and dark objects)
    otsu_threshold = threshold_otsu(relative_vals)

    # enforce max_keep_ratio to prevent mask from keeping too many pixels
    all_relative_in_mask = relative_dark[mask_bool]
    percentile_threshold = np.quantile(all_relative_in_mask, 1 - max_keep_ratio)

    # final threshold: need to be significantly darker AND in the top (1 - max_keep_ratio) of relative_dark
    final_threshold = max(otsu_threshold, percentile_threshold, rel_dark_min)

    # create refined mask
    refined_mask = np.zeros_like(shadow_mask, dtype=np.uint8)
    refined_mask[mask_bool & (relative_dark > final_threshold)] = 1

    # filter out high saturated pixels since shadows should be grayish, not colorful
    # NOTE: as noticed, colorful pixels in the images are mostly grass, cars, etc.
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
    S = img_hsv[:, :, 1]  # saturation channel [0,1]

    # remove the pixels that are too saturated
    refined_mask[(refined_mask == 1) & (S > sat_clip)] = 0

    return refined_mask

def grow_from_seeds(initial_mask, seed_mask):
    '''
    keep only shadow regions that contain high-confidence "seeds"

    issue: initial mask can have false positives (like non-shadows) and refined mask might miss some shadow areas (false negatives)
    - so i wanted to use the refined mask as "seeds" (high confidence that these are shadows) and grow these seeds to fill
        the initial mask regions they touch

    note: refined mask should tell that it's sure those are shadows in the image, initial mask shoudl tell that they might be shadows,
        final mask should create regions from the initial mask that contains seeds

    uses: connected components of the initial mask that contain at least one seed pixel (seed pixel is "sure" to be a shadow)
    - fill connected components/regions

    initial_mask: original binary 0/1 mask from detect_shadows_skimage
    seed_mask: refined binary 0/1 mask (sparse, mostly correct)
    '''
    # convert values to binary
    initial_binary = (initial_mask > 0).astype(np.uint8)
    seeds_binary = (seed_mask > 0).astype(np.uint8)

    # label all the connected components in the initial mask
    # each separate blob (shadow component) gets a unique number
    labeled, num_components = ndimage.label(initial_binary)

    # if there are no shadows at all, return 0s
    if num_components == 0:
        return np.zeros_like(initial_mask, dtype=np.uint8)

    # find which labels contain at least one seed pixel
    # used np.unique gives to give all the different label numbers where seeds exist
    labels_with_seed = np.unique(labeled[seeds_binary == 1])

    # remove background label (0)
    labels_with_seed = labels_with_seed[labels_with_seed != 0]

    if labels_with_seed.size == 0:
        # no seeds found at all, so no confident shadows
        return np.zeros_like(initial_mask, dtype=np.uint8)

    # keep only the components that have seeds
    # used np.isin to create a boolean mask for pixels with those labels
    final_mask = np.isin(labeled, labels_with_seed).astype(np.uint8)

    return final_mask

# --------------------------------------------------------------------
# VISUALIZATION
# --------------------------------------------------------------------
def overlay_mask(img, mask, color=(0,0,255), alpha=0.5):
    '''
    overlay a binary shadow mask (in red) on the original image to visualize the mask
    
    parameters:
    - img: original BGR image
    - mask: binary mask (0/1 or 0/255)
    - color: overlay color (BGR format)
    - alpha: transparency (0=transparent, 1=solid)
    '''
    # ensure mask is binary 0 or 1
    if mask.max() > 1:
        mask = (mask > 127).astype(np.uint8)

    # make overlay in red
    overlay = img.copy()
    # set shadow areas to overlay color
    overlay[mask == 1] = color  

    # blend original image and mask overlay
    result = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    
    return result

# --------------------------------------------------------------------
# FUNCTION RETURNING FINAL MASK TO TEXTURE, LIGHTING, DEPTH
# --------------------------------------------------------------------
def final_shadow_mask(img_bgr):
    '''
    return final shadow mask (binary shadow mask where 255=shadow and 0=not shadow)

    pipeline:
    - initial detection using LCH color space
    - refine initial using local brightness analysis
    - region of shadows that grow from confident seeds

    '''
    initial_mask, _ = detect_shadows_skimage(img_bgr)
    seed_mask = refine_shadow_mask_local(initial_mask, img_bgr)
    final_mask = grow_from_seeds(initial_mask, seed_mask)

    return (final_mask * 255).astype(np.uint8)

# comment main out if not testing shadow_mask.py only
# if __name__ == "__main__":
#     # initial mask
#     initial_mask, img = detect_shadows_skimage("data/images/19.jpg")

#     # second refinement
#     seed_mask = refine_shadow_mask_local(
#         initial_mask, img,
#         smooth_kernel=21,    # should be large enough to span a car / chunk of pavement
#         rel_dark_min=0.012,
#         max_keep_ratio=0.85,
#         sat_clip=0.30
#     )

#     initial_overlay = overlay_mask(img, initial_mask, color=(0,0,255), alpha=0.5)

#     seed_overlay = overlay_mask(img, seed_mask, color=(0,0,255), alpha=0.5)

#     final_mask = grow_from_seeds(initial_mask, seed_mask)

#     # cv2.namedWindow("Initial", cv2.WINDOW_NORMAL)
#     # cv2.resizeWindow("Initial", 800, 600)
#     # cv2.imshow("Initial", initial_overlay)

#     cv2.namedWindow("Seed", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("Seed", 800, 600)
#     cv2.imshow("Seed", seed_overlay)

#     overlay = overlay_mask(img, final_mask, color=(0,0,255), alpha=0.5)

#     cv2.namedWindow("Shadow Overlay", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("Shadow Overlay", 800, 600)
#     cv2.imshow("Shadow Overlay", overlay)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()