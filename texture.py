# combines canny edge detection, local binary pattern (LBP), and shadow mask
# for texture analysis (first feature)

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# LOCAL BINARY PATTERN
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


# SHADOW MASK
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
    # build the binary bask that marks the likely shadow regions
    # combine brightness and color consistency cues to distinguish shadows from dark objects

    # convert to Lab color space to isolate brightness
    lab = cv.cvtColor(img_bgr, cv.COLOR_BGR2Lab)
    L = lab[:, :, 0].astype(np.float32)

    if use_clahe:
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        L = clahe.apply(L.astype(np.uint8)).astype(np.float32)

    # local brightness mean and darkness condition
    meanL = box_mean(L, win)
    # dark is true when the pixel is darker than neighbors
    dark = (L < meanL * k_dark)

    # chromacity (color) consistency condition
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


# CANNY EDGE DETECTION
def canny_on_L(L, low=50, high=150):
    # do the edge detection on the L channel (luminance/lighting)
    # detect strong brightness changes and potential shadow boundaries
    L8 = cv.normalize(L, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    return cv.Canny(L8, low, high)

# COMPARE TEXTURE ACROSS SHADOW BOUNDARIES
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


# MAIN FUNCTION
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
    boundary = cv.morphologyEx(mask, cv.MORPH_GRADIENT,
                               cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3)))
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
    good_shadow_points = 0
    tested_points = 0

    for idx in range(0, len(ys), take_every):
        y, x = int(ys[idx]), int(xs[idx])
        g = np.array([gx[y,x], gy[y,x]], dtype=np.float32)
        nrm = np.linalg.norm(g)
        if nrm < 1e-3:
            continue
        nx, ny = g[0]/nrm, g[1]/nrm  # normalized gradient → edge normal direction

        # extract paired patches: darker side (inside shadow) & brighter side (outside)
        pinL, poutL = sample_patches(L8, y, x, ny, nx, size=patch_size, offset=patch_offset)
        pinB, poutB = sample_patches(lbp, y, x, ny, nx, size=patch_size, offset=patch_offset)

        # skip if patch dimensions invalid (near borders)
        if pinL.shape != (patch_size, patch_size) or poutL.shape != (patch_size, patch_size):
            continue

        # compute LBP histograms (texture descriptors)
        hin = lbp_hist(pinB)
        hout = lbp_hist(poutB)
        d = chi2(hin, hout)  # texture distance (lower = more similar)

        # shadow condition:
        #   (1) textures are similar → same surface
        #   (2) inside patch is darker than outside by luminance_gap
        if d < lbp_sim_thresh and (float(np.mean(pinL)) + luminance_gap < float(np.mean(poutL))):
            good_shadow_points += 1
        tested_points += 1

    # compute a "shadow-likeness" score = fraction of edges that behave like shadows
    score = (good_shadow_points / tested_points) if tested_points else 0.0
    print(f"Shadow-likeness score along boundary: {score:.2f} ({good_shadow_points}/{tested_points})")

    # visualize the results
    overlay = img.copy()
    overlay[mask == 255] = (0, 0, 255)  # show detected shadow mask in red
    vis = cv.addWeighted(img, 0.7, overlay, 0.3, 0)

    cv.imshow("Shadow mask", mask)
    cv.imshow("Shadow edges (Canny)", shadow_edges)
    cv.imshow("Overlay", vis)
    #cv.imshow("LBP + Shadow ouline", lbp_gray)
    plt.imshow(cv.cvtColor(lbp_gray, cv.COLOR_BGR2RGB))
    plt.show()
    cv.waitKey(0)
    cv.destroyAllWindows()

    return {
        "mask": mask,
        "edges": edges,
        "shadow_edges": shadow_edges,
        "score": score
    }


if __name__ == "__main__":
    analyze_image("77.jpg")