# testing a different masking functions for shadow mask

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
    # calculating the average of neighboring pixels with box filter
    # good for reducing noise since it smooths
    return cv2.boxFilter(arr, ddepth =- 1, ksize = (k, k), normalize = True)

def remove_small(mask, min_area = 800):
    # remove small noisy regions from the mask
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity = 8)
    keep = np.zeros_like(mask)
    
    # background is labeled 0
    # keep only the large regions
    for i in range(1, num):
        # use the total area (# of pixels) of the component
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 255

    return keep

# RGB to HSI conversion
def bgr_to_hsi(img_bgr):
    img_bgr = img_bgr.astype(np.float32) / 255.0
    B, G, R = cv2.split(img_bgr)
    
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
    # Normalize only if we're in 0..255 domain
    if x.max() > 1.0:
        x = x / 255.0
    a = 0.055
    return np.where(x <= 0.04045, x/12.92, ((x + a)/(1 + a))**2.4)

def normalized_rgb(img_bgr):
    # normalize the RGB channels to help identify shadows through color consistency
    b, g, r = cv2.split(img_bgr.astype(np.float32))

    # calculate sum of pixel values
    # total light intensity (similar to luminance) at each pixel location
    sum = r + g + b + 1e-6
    return r/sum, g/sum     # chroma ratios red and green since they are more stable than blue channel

def bgr_to_hsi_linear(img_bgr):
    # convert BGR to HSI color space (hue, saturation, intensity)
    # this is important to adjust brightness without affecting shadow's hue or saturation
    B, G, R = cv2.split(img_bgr.astype(np.float32))

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

def get_brightness(img_bgr):
    # calculate brightness (luminance) from BGR image
    # convert to linear first
    linear = srgb_to_linear(img_bgr)
    B, G, R = cv2.split(linear)
    # standard brightness formula
    brightness = 0.114*B + 0.587*G + 0.299*R
    return brightness.astype(np.float32)

def get_rgb_direction(img_bgr):
    # get RGB direction for color matching
    linear = srgb_to_linear(img_bgr)
    length = np.linalg.norm(linear, axis=2, keepdims=True) + 1e-6
    return linear/length

# ----------------------------------------------------------
# PREPROCESSING 
# ----------------------------------------------------------
def preprocess_for_shadow(
    img_bgr,
    gauss_ksize=5,
    gauss_sigma=1.0,
    log_ksize=3,
    log_sigma=0.0,
    log_alpha=0.35      
):
    '''
    applying preprocessing techniques of Gaussian blur and LoG to reduce noise and emphasis edges
    '''
    # gaussian blur on the color image
    if gauss_ksize < 3:
        gauss_ksize = 3
    if gauss_ksize % 2 == 0:
        gauss_ksize += 1
    img_blur = cv2.GaussianBlur(img_bgr, (gauss_ksize, gauss_ksize), gauss_sigma)

    # LoG applied to grayscale
    gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    if log_sigma > 0.0:
        if log_ksize < 3:
            log_ksize = 3
        if log_ksize % 2 == 0:
            log_ksize += 1
        gray_for_log = cv2.GaussianBlur(gray, (log_ksize, log_ksize), log_sigma)
    else:
        gray_for_log = gray

    # laplacian on the (pre)blurred gray
    lap = cv2.Laplacian(gray_for_log, cv2.CV_32F, ksize=log_ksize if log_ksize >= 1 else 3)

    # normalize lap to a comparable scale and add to luminance
    # (log_alpha controls contribution)
    gray32 = gray.astype(np.float32)
    # scale Laplacian roughly into 8-bit
    lap_norm = lap / (np.max(np.abs(lap)) + 1e-6) * 127.0
    gray_enh = gray32 + log_alpha * lap_norm

    # clip back to [0,255] and cast
    gray_enh = np.clip(gray_enh, 0, 255).astype(np.uint8)

    # map enhanced gray back into BGR while keeping chroma roughly constant
    # compute ratio between enhanced and original luminance then apply to each channel
    denom = gray.astype(np.float32) + 1.0
    ratio = (gray_enh.astype(np.float32) + 1.0) / denom
    ratio = np.clip(ratio, 0.25, 4.0)  # avoid wild gains

    img_pre = img_blur.astype(np.float32)
    img_pre[..., 0] *= ratio
    img_pre[..., 1] *= ratio
    img_pre[..., 2] *= ratio

    img_pre = np.clip(img_pre, 0, 255).astype(np.uint8)
    return img_pre    

def detect_shadows_skimage(img, morph_size=9):
    # more accurate shadow masking using scikit-image + opencv

    # load
    # img_bgr = cv2.imread(img_path)
    # if img_bgr is None:
    #     raise ValueError("Image could not be loaded.")
    img_bgr = preprocess_for_shadow(img)

    # convert bgr to rgb
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # convert to float [0,1] for skimage
    img_rgb_f = img_rgb.astype(np.float32) / 255.0
    img_rgb_f = np.nan_to_num(img_rgb_f, nan=0.0, posinf=1.0, neginf=0.0)

    # LAB to LCH (skimage is more accurate than opencv)
    lab = rgb2lab(img_rgb_f)
    lch = lab2lch(lab)

    L = lch[:, :, 0]    # lighting
    C = lch[:, :, 1]    # chroma
    H = lch[:, :, 2]    # hue

    # normalize
    Ln = rescale_intensity(L, out_range=(0,1))
    Cn = rescale_intensity(C, out_range=(0,1))

    ''' 
    shadows tend to have:
    - low light (Ln)
    - low chroma (Cn)
    '''     
    # score = dark + low chroma
    shadow_score = (1-Ln) * (1-Cn)

    # multi-otsu gives 3 cluster thresholds: light, mid, shadow
    # automatic thresholding
    thresholds = threshold_multiotsu(shadow_score, classes=3)
    t_shadow = thresholds[1]    # darkest cluster only

    shadow_mask = (shadow_score > t_shadow).astype(np.uint8)

    # clean with morphological operations
    # selem = disk(morph_size)
    # shadow_mask = closing(shadow_mask, selem)

    return shadow_mask.astype(np.uint8), img_bgr

def refine_shadow_mask_local(shadow_mask, img_bgr,
                             smooth_kernel=21,
                             rel_dark_min=0.01,
                             max_keep_ratio=0.9,
                             sat_clip=0.30):
    '''
    refine an initial shadow mask using local relative darkness and saturation

    shadow_mask : initial 0/1 mask from your Ln/Cn + multi-Otsu step
    smooth_kernel : window size for local L smoothing (odd, e.g. 31–51)
    rel_dark_min : minimum local darkness difference to even be considered
    max_keep_ratio : keep at most this fraction of the original masked pixels
    sat_clip : HSV saturation above this is rejected as non-shadow
    '''

    # convert and get L (lightness)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    lab = rgb2lab(img_rgb)
    L = lab[:, :, 0]        # already ~[0,100]
    Ln = rescale_intensity(L, out_range=(0, 1))

    # local illumination estimate with smoothing 
    L_smooth = uniform_filter(Ln, size=smooth_kernel)
    rel_dark = L_smooth - Ln   # >0 when pixel is darker than local average

    mask_bool = (shadow_mask.astype(np.uint8) > 0)

    # only consider candidate pixels inside original mask
    rel_vals = rel_dark[mask_bool]

    if rel_vals.size < 50:
        # not enough pixels to do anything smarter
        return shadow_mask

    # reject very small differences
    rel_vals = rel_vals[rel_vals > rel_dark_min]
    if rel_vals.size < 50:
        return np.zeros_like(shadow_mask, dtype=np.uint8)

    # otsu threshold on local darkness within the mask
    t_otsu = threshold_otsu(rel_vals)

    # enforce that we don't keep more than max_keep_ratio
    all_rel_in_mask = rel_dark[mask_bool]
    t_percentile = np.quantile(all_rel_in_mask, 1 - max_keep_ratio)

    # final threshold: need to be significantly darker AND
    # in the top (1 - max_keep_ratio) of rel_dark
    t_rel = max(t_otsu, t_percentile, rel_dark_min)

    refined = np.zeros_like(shadow_mask, dtype=np.uint8)
    refined[mask_bool & (rel_dark > t_rel)] = 1

    # optional saturation filter to kill cars / vegetation
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
    S = img_hsv[:, :, 1]  # saturation in [0,1]
    refined[(refined == 1) & (S > sat_clip)] = 0

    return refined

def grow_from_seeds(initial_mask, seed_mask):
    '''
    use refined_mask as high-confidence seeds and keep only those
    connected components of the initial_mask that contain at least
    one seed pixel

    initial_mask : original 0/1 mask from detect_shadows_skimage
    seed_mask    : refined 0/1 mask (sparse, mostly correct)
    '''
    init = (initial_mask > 0).astype(np.uint8)
    seeds = (seed_mask > 0).astype(np.uint8)

    # label connected components in the initial mask
    labeled, num = ndimage.label(init)

    if num == 0:
        return np.zeros_like(initial_mask, dtype=np.uint8)

    # find which labels contain at least one seed
    labels_with_seed = np.unique(labeled[seeds == 1])
    labels_with_seed = labels_with_seed[labels_with_seed != 0]

    if labels_with_seed.size == 0:
        # no seeds at all → no shadows
        return np.zeros_like(initial_mask, dtype=np.uint8)

    # keep only those components
    final_mask = np.isin(labeled, labels_with_seed).astype(np.uint8)

    return final_mask


def overlay_mask(img, mask, color=(0,0,255), alpha=0.5):
    '''
    overlay a binary shadow mask on the original image
    
    parameters:
        img - original BGR image
        mask - binary mask (0/1 or 0/255)
        color - overlay color (B, G, R)
        alpha - transparency (0 = transparent, 1 = opaque)
    '''
    # ensure mask is 0 or 1
    if mask.max() > 1:
        mask = (mask > 127).astype(np.uint8)

    # make overlay image
    overlay = img.copy()
    overlay[mask == 1] = color  # set shadow areas to overlay color

    # blend original + overlay
    output = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    return output

def final_shadow_mask(img_bgr):
    '''
    return final shadow mask for shadow feature analysis files
    '''
    initial_mask, _ = detect_shadows_skimage(img_bgr)
    seed_mask = refine_shadow_mask_local(initial_mask, img_bgr)
    final_mask = grow_from_seeds(initial_mask, seed_mask)

    return (final_mask * 255).astype(np.uint8)

# if __name__ == "__main__":
#     # initial mask
#     initial_mask, img = detect_shadows_skimage("data/images/14.jpg")

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