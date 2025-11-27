# testing a different masking functions for shadow mask

import numpy as np
import cv2
from skimage.color import rgb2lab, lab2lch
from skimage.filters import threshold_multiotsu
from skimage.morphology import disk, closing
from skimage.exposure import rescale_intensity
from scipy.ndimage import uniform_filter

def detect_shadows_skimage(img_path, morph_size=9):
    # more accurate shadow masking using scikit-image + opencv

    # load
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise ValueError("Image could not be loaded.")
    
    # convert bgr to rgb
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # vonvert to float [0,1] for skimage
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

    # multi-Otsu gives 3 cluster thresholds: light, mid, shadow
    # automatic thresholding
    thresholds = threshold_multiotsu(shadow_score, classes=3)
    t_shadow = thresholds[0]    # lowest-intensity cluster

    shadow_mask = (shadow_score > t_shadow).astype(np.uint8)

    # clean with morphological operations
    selem = disk(morph_size)
    shadow_mask = closing(shadow_mask, selem)

    return shadow_mask.astype(np.uint8), img_bgr

def overlay_mask(img, mask, color=(0,0,255), alpha=0.5):
    '''
    Overlay a binary shadow mask on the original image.
    
    Parameters:
        img   : original BGR image
        mask  : binary mask (0/1 or 0/255)
        color : overlay color (B, G, R)
        alpha : transparency (0 = transparent, 1 = opaque)
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

if __name__ == "__main__":
    shadow_mask, img = detect_shadows_skimage("data/images/28.jpg")

    overlay = overlay_mask(img, shadow_mask, color=(0,0,255), alpha=0.5)

    cv2.imshow("Shadow Overlay", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()