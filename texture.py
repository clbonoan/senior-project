# First Feature: edge detection and texture analysis
# Detects shadows in images and compares textures inside and outside shadows

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from shadow_mask import final_shadow_mask, bgr_to_hsi_linear

# ----------------------------------------------------------
# FEATURE EXTRACTION HELPERS
# ----------------------------------------------------------
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
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas, peri2_over_area = [], []

    for i in range(1, num):
        # 0 is the background
        a = stats[i, cv2.CC_STAT_AREA]
        areas.append(a)
        comp = (labels == i).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts and a > 0:
            peri = cv2.arcLength(max(cnts, key=cv2.contourArea), True)
            peri2_over_area.append((peri * peri) / a)

    out = {"num_shadow_components": int(num - 1)}
    out["comp_area_p90"] = percentile(areas, 90)
    out["comp_peri2_over_area_p50"] = percentile(peri2_over_area, 50)
    return out

def angular_variance_min(gx, gy, boundary_mask, eps=1e-6):
    '''
    normalized mean resultant length measures directional consistency: 
    REAL SHADOW: the score is high (near 1.0) because the light comes from one 
    source, making all edge pixels point in the same direction
    TAMPERED SHADOW: the score is low (near 0.0) because manual edits or copying 
    create messy edges that point in conflicting directions

    normalized angular std measures how much the directions of the shadow's edge 
    pixels "wiggle" or deviate from the average direction:
    REAL SHADOW: score is low because the angles are steady and consistent 
    with natural physics
    TAMPERED SHADOW: score is high because digital brushes or "drop shadow" 
    effects create "noisy" angles that fluctuate too much.
    '''
    # return: normal_mrl (alignment) and normal_angle_std (irregularity)
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

def boundary_from_mask(mask):
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, se)

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
    height, width = img.shape

    if x < 0 or y < 0 or x >= height or y >= width:
        return 0
    return 1 if img[x, y] >= center else 0

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

# def lbp_hist(patch):
#     # 8-neighbor lbp -> values 0 to 255
#     # histogram for local texture pattern distribution
#     hist, _ = np.histogram(patch.ravel(), bins=256, range=(0,256), density=True)
#     return hist

def patch_entropy(lbp_u8, mask_u8=None, patch_size=32, stride=None, prefix=None, z_thresh=2.5):
    '''
    PREVIOUS: divide the image into patches and find LBP entropy per patch to see if there were
    any copied textures (from clone stamping)

    CURRENT: compute patch-level LBP entropy using overlapping patches
    - entropy tells how random/detailed texture is in a patch
    - uses overlapping patches (to capture local edits better)
    - uses robust stats (mean and std - std measures spread around the mean)
    - use different patch sizes to look at texture at different scales
    - detects outlier patches (possible tampered regions (unusual))
    '''
    # if stride is not given, move half a patch each time (creates overlap between patches)
    if stride is None: 
        stride = max(8, patch_size // 2)

    # prefix the feature names so they represent each patch size
    if prefix is None:
        prefix = f"patch_entropy_s{patch_size}"

    height, width = lbp_u8.shape[:2]
    # handle floating-point precision issues (prevent divide-by-zero)
    eps = 1e-12

    # entropy for all patches
    entropy_list = []
    # entropy for patches inside shadow
    entropy_inside = []
    # entropy for patches outside shadow
    entropy_outside = []
    
    # looping over overlapping patches
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            # split the image into patches/tiles
            row_start = y
            row_end = y + patch_size
            col_start = x
            col_end = x + patch_size
            # get one patch from the LBP image
            # .ravel() turns 2D grid into a single list so np.bincount() can use the 1D list
            patch = lbp_u8[row_start:row_end, col_start:col_end].ravel()

            # compute entropy with a histogram (list of probabilities based on how often
            # each pixel value (0-255) showed up in a patch)
            hist = np.bincount(patch, minlength=256).astype(np.float32)
            # calculate probabilities from histogram counts
            hist /= hist.sum() + eps

            # entropy = -(sum of all: probability of pixel appearing * log(probability))
            entropy = -float(np.sum(hist * np.log(hist + eps)))
            entropy_list.append(entropy)

            # separate entropy patches by inside and outside shadow mask
            if mask_u8 is not None:
                mask_patch = mask_u8[row_start:row_end, col_start:col_end].ravel()

                # fraction of the patch that is inside the shadow
                fraction_inside = (mask_patch > 0).sum() / mask_patch.size
                if fraction_inside >= 0.7:
                    '''
                    put entropy value in inside list if more than 70% of the patch is
                    inside a shadow found by the shadow mask; between 30-70% is not used
                    because those may represent patches that are on a shadow edge 
                    (hard to identify)
                    '''
                    entropy_inside.append(entropy)
                elif fraction_inside <= 0.3:
                    entropy_outside.append(entropy) 

    # return 0 if no patches are found
    if len(entropy_list) == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_max_dev": 0.0,
            f"{prefix}_in_mean": 0.0,
            f"{prefix}_out_mean": 0.0,
            f"{prefix}_in_out_gap": 0.0,
            f"{prefix}_outlier_frac": 0.0,
        }
    
    # turn the list into an array for stats
    entropy_arr = np.asarray(entropy_list, dtype=np.float32)

    # average entropy across all patches
    mean = float(np.mean(entropy_arr))
    # spread of entropy values (standard deviation)
    std = float(np.std(entropy_arr))

    # patches that are more than 2 standard deviations from mean (unusual patches)
    if std > 1e-6:
        # a large z-score means it is far from the mean, so it is unlike the shadow
        z_scores = (entropy_arr - mean) / std

        # fraction of patches that are far from the average
        outlier_frac = float(np.mean(np.abs(z_scores) > 2.5))

        # strongest deviation from the mean
        max_dev = float(np.max(np.abs(entropy_arr - mean)))
    else:
        outlier_frac = 0.0
        max_dev = 0.0

    # average entropy inside and outside shadow patches
    in_mean = float(np.mean(entropy_inside)) if entropy_inside else 0.0
    out_mean = float(np.mean(entropy_outside)) if entropy_outside else 0.0

    # return the features with names that correlate to the patch size
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_max_dev": max_dev,
        f"{prefix}_in_mean": in_mean,
        f"{prefix}_out_mean": out_mean,
        f"{prefix}_in_out_gap": abs(in_mean - out_mean),
        f"{prefix}_outlier_frac": outlier_frac,
    }

def patch_luminance_std(L8_u8, mask_u8=None, patch_size=32, stride=None, prefix=None, z_thresh=2.5):
    '''
    compute patch-level luminance standard deviation using overlapping patches

    why:
    - std shows how much brightness varies inside a patch
    - real shadows usually keep some surface detail / variation
    - fake shadows may smooth or flatten that variation
    - using multiple patch sizes lets us look at local and larger texture strength

    returns summary features like:
    - mean std across patches
    - std of the patch-std values
    - max deviation from the mean
    - average patch std inside shadows
    - average patch std outside shadows
    - inside/outside gap
    - fraction of unusual patches
    '''

    # if stride is not given, move half a patch each time
    # this creates overlap between patches
    if stride is None:
        stride = max(8, patch_size // 2)

    # use the patch size in the feature names
    if prefix is None:
        prefix = f"patch_lumstd_s{patch_size}"

    height, width = L8_u8.shape[:2]

    # store the standard deviation value for every patch
    std_list = []

    # store patch std values for patches mostly inside shadow
    std_inside = []

    # store patch std values for patches mostly outside shadow
    std_outside = []

    # loop over overlapping patches
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            # grab one patch from the luminance image
            patch = L8_u8[y:y + patch_size, x:x + patch_size].astype(np.float32)

            # standard deviation of brightness values in this patch
            patch_std = float(np.std(patch))
            std_list.append(patch_std)

            # if mask is given, separate inside-shadow and outside-shadow patches
            if mask_u8 is not None:
                mask_patch = mask_u8[y:y + patch_size, x:x + patch_size].ravel()

                # fraction of this patch inside the shadow
                fraction_inside = (mask_patch > 0).sum() / mask_patch.size

                # only use clearly inside or clearly outside patches
                if fraction_inside >= 0.7:
                    std_inside.append(patch_std)
                elif fraction_inside <= 0.3:
                    std_outside.append(patch_std)

    # if no patches found, return zeros so code does not crash
    if len(std_list) == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_max_dev": 0.0,
            f"{prefix}_in_mean": 0.0,
            f"{prefix}_out_mean": 0.0,
            f"{prefix}_in_out_gap": 0.0,
            f"{prefix}_outlier_frac": 0.0,
        }

    # convert list to numpy array for stats
    std_arr = np.asarray(std_list, dtype=np.float32)

    # average patch std across all patches
    mean = float(np.mean(std_arr))

    # spread of patch std values
    std = float(np.std(std_arr))

    # find unusual patches with z-scores
    if std > 1e-6:
        z_scores = (std_arr - mean) / std

        # fraction of patches that are far from average
        outlier_frac = float(np.mean(np.abs(z_scores) > z_thresh))

        # strongest deviation from average
        max_dev = float(np.max(np.abs(std_arr - mean)))
    else:
        outlier_frac = 0.0
        max_dev = 0.0

    # average patch std inside and outside shadows
    in_mean = float(np.mean(std_inside)) if std_inside else 0.0
    out_mean = float(np.mean(std_outside)) if std_outside else 0.0

    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_max_dev": max_dev,
        f"{prefix}_in_mean": in_mean,
        f"{prefix}_out_mean": out_mean,
        f"{prefix}_in_out_gap": abs(in_mean - out_mean),
        f"{prefix}_outlier_frac": outlier_frac,
    }

# DISABLED: clone_detection is not currently used in the pipeline.
# kept here for reference in case it is re-enabled later.
# to re-enable: uncomment the call in analyze_texture and add clone_score back to calculate_texture_tamper_score
def clone_detection(L8_u8, patch_size=40, max_patches=100, similar_thresh=0.95):
    '''
    sample patches throughout the image and compare them to each other; if two patches
    look very similar, possibly copy-pasted (clone stamped).
    return highest similarity found and how many pairs were similar (suspicious)

    return:
        clone_feats: dict
        patches: list of patch metadata for visualization
    '''

    height, width = L8_u8.shape[:2]
    patches = []

    # evenly spread patches across image instead of checking every single spot
    # step is how many pixels to move down before starting next row of patches
    step_y = max(patch_size, height // int(np.sqrt(max_patches)))   
    step_x = max(patch_size, width // int(np.sqrt(max_patches)))

    for y in range(0, height - patch_size + 1, step_y):
        for x in range(0, width - patch_size + 1, step_x):
            row_start = y 
            row_end = y + patch_size
            col_start = x 
            col_end = x + patch_size
            patch = L8_u8[row_start:row_end, col_start:col_end].astype(np.float32)

            # shrink the patch from 24x24 to 8x8 -> faster (less) comparisons and less sensitive
            # average the pixels in each area when shrinking so you don't lose overall pattern
            small_patch = cv2.resize(patch, (8,8), interpolation=cv2.INTER_AREA).ravel()

            # subtract average brightness so patterns are compared, not brightness
            # ex: if you have identical patches but one is in shade and one is in light,
            # they will look different without subtracting avg brightness
            small_patch = small_patch - small_patch.mean()
            patch_length = np.linalg.norm(small_patch)
            if patch_length < 1e-6:     # skip flat/empty patches
                continue

            # skip patches that are nearly uniform like the sky
            # variance of the normalized patch values
            if np.var(small_patch) < 0.01:
                continue
            
            small_patch = small_patch / patch_length
            #patches.append(small_patch)
            patches.append({"vec": small_patch, "row": row_start, "col": col_start})

            if len(patches) >= max_patches:
                break
        if len(patches) >= max_patches:
            break
    
    num_patches = len(patches)
    if num_patches < 2:
        return {
            "clone_similar_max": 0.0, 
            "clone_similar_count": 0,
            "clone_similar_frac": 0.0
        }, patches
    
    # compare every patch against every other patch (brute force)
    similarities = []
    suspicious_pairs = 0
    valid_pairs = 0

    for i in range(num_patches):
        for j in range(i + 1, num_patches):
            # dot product of two normalized vectors = cosine similarity (1.0 = identical)
            # patches must be far enough apart to be suspicious
            dist = np.sqrt((patches[i]["row"] - patches[j]["row"])**2 + 
                            (patches[i]["col"] - patches[j]["col"])**2)
                
            # compare patches that are far apart, at least 2 patches away (skip nearby patches)
            if dist <= patch_size * 2:   
                continue

            valid_pairs += 1
            similarity = float(np.dot(patches[i]["vec"], patches[j]["vec"]))
            similarities.append(similarity)

            if similarity >= similar_thresh:
                suspicious_pairs += 1

    if len(similarities) == 0:
        return {"clone_similar_max": 0.0, 
                "clone_similar_count": 0,
                "clone_similar_frac": 0.0
            }, patches
    
    max_similarity = float(np.array(similarities).max())
    similar_frac = float(suspicious_pairs / valid_pairs) if valid_pairs > 0 else 0.0

    return {"clone_similar_max": max_similarity, 
            "clone_similar_count": int(suspicious_pairs),
            "clone_similar_frac": similar_frac
        }, patches


# ----------------------------------------------------------
# TEXTURE COMPARISON AND HELPERS FOR TAMPER SCORE
# compare texture between shadow patches and nearby lit patches
# ----------------------------------------------------------
def pairs_one_per_component(
    mask_u8, L8, lbp, gx, gy,
    min_area=300,
    patch_size=25,
    in_offset=6,
    out_offset=9,
    max_step=30
):
    '''
    for each connected shadow component in mask_u8, pick one pair of patches to compare:
      - inside patch: near boundary but inside (stable, not on edge)
      - outside patch: just outside the boundary in the lit area
    return:
      chi2_list (distances): list[float]
      patch_pairs: [((x_in,y_in),(x_out,y_out), size)]
    '''
    height, width = L8.shape
    half_patch = patch_size // 2
    chi2_list, patch_pairs = [], []

    # find all separate shadow components/regions
    num_components, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    ellipse_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def patch_fits_in_image(center_y, center_x):
        # check that a patch centered at (center_y, center_x) stays within image bounds
        top_edge = center_y - half_patch >= 0
        left_edge = center_x - half_patch >= 0
        bottom_edge = center_y + half_patch < height
        right_edge = center_x + half_patch < width
        return top_edge and left_edge and bottom_edge and right_edge

    def crop_patch(img, center_y, center_x):
        # extract a square patch centered at (center_y, center_x)
        return img[center_y - half_patch : center_y + half_patch + 1,
                   center_x - half_patch : center_x + half_patch + 1]

    # go through each shadow region
    for component_idx in range(1, num_components):  # 0 is background
        component_area = stats[component_idx, cv2.CC_STAT_AREA]
        # skip small regions
        if component_area < min_area:
            continue

        # grab shadow component
        component = (labels == component_idx).astype(np.uint8) * 255

        boundary = cv2.morphologyEx(component, cv2.MORPH_GRADIENT, ellipse_kernel)

        if boundary.sum() == 0:
            continue

        # find a good point inside the shadow that is far from edges
        dist = cv2.distanceTransform((component > 0).astype(np.uint8), cv2.DIST_L2, 5)
        yi_in, xi_in = np.unravel_index(np.argmax(dist), dist.shape)
        yi_in = int(np.clip(yi_in, half_patch, height - half_patch - 1))
        xi_in = int(np.clip(xi_in, half_patch, width  - half_patch - 1))

        if not patch_fits_in_image(yi_in, xi_in):
            continue

        # find the closest boundary point to the deep interior point
        boundary_ys, boundary_xs = np.where(boundary > 0)
        # calculate squared euclidean distance from each boundary pixel to the interior point
        # and keep the index of the closest one
        closest_idx = int(np.argmin((boundary_ys - yi_in)**2 + (boundary_xs - xi_in)**2))
        boundary_y, boundary_x = int(boundary_ys[closest_idx]), int(boundary_xs[closest_idx])

        # find the outward direction (pointing away from shadow interior, toward lit area)
        gradient_vec = np.array([gx[boundary_y, boundary_x], gy[boundary_y, boundary_x]], dtype=np.float32)
        gradient_magnitude = float(np.linalg.norm(gradient_vec))

        # default outward direction if gradient is too weak to use
        outward_x, outward_y = 1.0, 0.0

        if gradient_magnitude > 1e-3:
            outward_x = gradient_vec[0] / gradient_magnitude
            outward_y = gradient_vec[1] / gradient_magnitude

        # shadow centroid (the weighted average position of all points in shadow shape)
        # the "geometric center", not always visually in the center
        shadow_moments = cv2.moments(component, binaryImage=True)

        # refine outward direction using centroid-to-boundary vector if centroid is valid
        if abs(shadow_moments["m00"]) >= 1e-6:
            centroid_x = shadow_moments["m10"] / shadow_moments["m00"]
            centroid_y = shadow_moments["m01"] / shadow_moments["m00"]

            # vector pointing from the shadow's center outward to the boundary point
            centroid_to_boundary = np.array([boundary_x - centroid_x, boundary_y - centroid_y], dtype=np.float32)
            centroid_to_boundary_magnitude = float(np.linalg.norm(centroid_to_boundary))

            if centroid_to_boundary_magnitude > 1e-6:
                outward_x, outward_y = centroid_to_boundary / centroid_to_boundary_magnitude
            else:
                outward_x, outward_y = 1.0, 0.0

        # inside patch center: step inward from the boundary point
        inside_patch_y = int(np.clip(boundary_y - in_offset * outward_y, half_patch, height - half_patch - 1))
        inside_patch_x = int(np.clip(boundary_x - in_offset * outward_x, half_patch, width  - half_patch - 1))

        if not patch_fits_in_image(inside_patch_y, inside_patch_x):
            continue

        # outside patch center: step outward until we leave the shadow mask
        outward_step = out_offset
        outside_patch_y = int(np.clip(boundary_y + outward_step * outward_y, half_patch, height - half_patch - 1))
        outside_patch_x = int(np.clip(boundary_x + outward_step * outward_x, half_patch, width  - half_patch - 1))

        while outward_step < max_step and component[outside_patch_y, outside_patch_x] > 0:
            outward_step += 2
            outside_patch_y = int(np.clip(boundary_y + outward_step * outward_y, half_patch, height - half_patch - 1))
            outside_patch_x = int(np.clip(boundary_x + outward_step * outward_x, half_patch, width  - half_patch - 1))

        if not patch_fits_in_image(outside_patch_y, outside_patch_x):
            continue

        # extract the inside and outside patches (luminance and LBP)
        inside_patch_luminance = crop_patch(L8,  inside_patch_y,  inside_patch_x)
        outside_patch_luminance = crop_patch(L8,  outside_patch_y, outside_patch_x)
        inside_patch_lbp = crop_patch(lbp, inside_patch_y,  inside_patch_x)
        outside_patch_lbp = crop_patch(lbp, outside_patch_y, outside_patch_x)

        # make sure the inside patch is darker than outside (swap if needed)
        mean_brightness_inside = float(np.mean(inside_patch_luminance))
        mean_brightness_outside = float(np.mean(outside_patch_luminance))

        if mean_brightness_outside <= mean_brightness_inside:
            # swap so the darker patch is always treated as the shadow side
            inside_patch_luminance, outside_patch_luminance = outside_patch_luminance, inside_patch_luminance
            inside_patch_lbp, outside_patch_lbp = outside_patch_lbp, inside_patch_lbp
            (inside_patch_x, inside_patch_y), (outside_patch_x, outside_patch_y) =\
                (outside_patch_x, outside_patch_y), (inside_patch_x, inside_patch_y)

        # compare LBP histograms with chi-squared distance (lower = more similar texture)
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
        lbp_hist_inside, _ = np.histogram(inside_patch_lbp.ravel(), bins=256, range=(0, 256), density=True)
        lbp_hist_outside, _ = np.histogram(outside_patch_lbp.ravel(), bins=256, range=(0, 256), density=True)
        chi2_distance = 0.5 * np.sum(
            ((lbp_hist_inside - lbp_hist_outside) ** 2) / (lbp_hist_inside + lbp_hist_outside + 1e-9)
        )

        chi2_list.append(float(chi2_distance))
        inside_top_left_x = inside_patch_x - half_patch
        inside_top_left_y = inside_patch_y - half_patch
        outside_top_left_x = outside_patch_x - half_patch
        outside_top_left_y = outside_patch_y - half_patch
        patch_pairs.append(((inside_top_left_x, inside_top_left_y), (outside_top_left_x, outside_top_left_y), patch_size))

    return chi2_list, patch_pairs

# ----------------------------------------------------------
# FEATURE EXTRACTION
# ----------------------------------------------------------
def extract_features(
    img_bgr,
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
    ml_features = {}

    # make sure mask is uint8
    mask = mask.astype(np.uint8)

    # get the shadow boundary from the mask
    # this is needed for the boundary direction consistency features
    boundary = boundary_from_mask(mask)

    # boundary geometry consistency
    '''
    use gradient directions gx and gy to see how consistent the shadow edge direction is
    - if all edge directions are aligned, clear and natural boundary
    - if edge directions point everywhere, messy or fake
    '''
    ml_features.update(angular_variance_min(gx, gy, boundary))

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
    ml_features["chi2_count"] = int(d_arr.size)
    ml_features["chi2_mean"] = float(d_arr.mean()) if d_arr.size else 0.0
    ml_features["chi2_std"] = float(d_arr.std()) if d_arr.size else 0.0
    ml_features["chi2_p25"] = percentile(d_arr, 25) if d_arr.size else 0.0
    ml_features["chi2_p50"] = percentile(d_arr, 50) if d_arr.size else 0.0
    ml_features["chi2_p75"] = percentile(d_arr, 75) if d_arr.size else 0.0

    # MULTI-SCALE PATCH ENTROPY FEATURES
    '''
    using multiple sizes instead of only one patch size; small patches are for more 
    local details, medium or large patches are for more overall texture consistency
    '''
    for patch_size in [24, 40]:
        try:
            # only using one patch size
            entropy_feats = patch_entropy(
                lbp.astype(np.uint8), 
                mask.astype(np.uint8), 
                patch_size=patch_size,
                prefix=f"patch_entropy_s{patch_size}"
            )
        except Exception:
            # use default zero values instead of crashing
            entropy_feats = {
                f"patch_entropy_s{patch_size}_mean": 0.0,
                f"patch_entropy_s{patch_size}_std": 0.0,
                f"patch_entropy_s{patch_size}_max_dev": 0.0,
                f"patch_entropy_s{patch_size}_in_mean": 0.0,
                f"patch_entropy_s{patch_size}_out_mean": 0.0,
                f"patch_entropy_s{patch_size}_in_out_gap": 0.0,
                f"patch_entropy_s{patch_size}_outlier_frac": 0.0,      
            }
        ml_features.update(entropy_feats)

    # MULTI-SCALE PATCH LUMINANCE (light intensity) STD FEATURES
    '''
    using luminance standard deviation to measure how much local brightness varies
    in a patch

    idea:
    - real shadows keep some texture / variation from the surface underneath
    - fake shadows can make regions look too smooth or too flat
    - compare patch luminance variation at different patch sizes
    '''
    for patch_size in [24, 40]:
        try:
            lumstd_feats = patch_luminance_std(
                L8.astype(np.uint8),
                mask.astype(np.uint8),
                patch_size=patch_size,
                prefix=f"patch_lumstd_s{patch_size}"
            )
        except Exception:
            # use zeros if something goes wrong
            lumstd_feats = {
                f"patch_lumstd_s{patch_size}_mean": 0.0,
                f"patch_lumstd_s{patch_size}_std": 0.0,
                f"patch_lumstd_s{patch_size}_max_dev": 0.0,
                f"patch_lumstd_s{patch_size}_in_mean": 0.0,
                f"patch_lumstd_s{patch_size}_out_mean": 0.0,
                f"patch_lumstd_s{patch_size}_in_out_gap": 0.0,
                f"patch_lumstd_s{patch_size}_outlier_frac": 0.0,
            }

        ml_features.update(lumstd_feats)

    return ml_features

# ----------------------------------------------------------
# TAMPER SCORE CALCULATION & FEATURE EXTRACTION
# ----------------------------------------------------------
def calculate_texture_tamper_score(chi2_list, features=None):
    '''
    rule-based tamper score (0.0 = likely real, 1.0 = likely tampered)
    based on two signals that match the module's two analysis methods:

    1. cross-boundary texture — chi-squared distance (60%):
       for each shadow, we compare LBP texture histograms from a patch just inside
       the shadow boundary vs a patch just outside it
       - real: same surface on both sides → histograms match → low chi2
       - fake: shadow placed on a different surface → histograms differ → high chi2

    2. LBP texture — entropy gap (40%):
       we compare texture complexity (LBP entropy) inside vs outside the shadow
       using overlapping patches across the whole image
       - real: similar complexity on both sides (same material, just darker)
       - fake: complexity differs because the shadow was added over different content
    '''
    if features is None:
        features = {}

    def linear_score(x, low, high):
        # returns 0.0 at x <= low, 1.0 at x >= high, linear in between
        if x <= low:
            return 0.0
        if x >= high:
            return 1.0
        return (x - low) / (high - low)

    # ----------------------------------
    # signal 1: cross-boundary texture (chi2)
    # ----------------------------------
    if len(chi2_list) == 0:
        cross_boundary_score = 0.0  # no shadows found — nothing suspicious to report
    else:
        chi2_arr = np.asarray(chi2_list, dtype=np.float32)

        # average texture mismatch across all shadow regions
        # real: mean < 0.15 (textures match well on the same surface)
        # suspicious: mean > 0.40 (textures clearly come from different surfaces)
        mean_chi2 = float(np.mean(chi2_arr))
        score_mean = linear_score(mean_chi2, 0.15, 0.40)

        # fraction of shadow regions where chi2 is high (individual texture mismatch)
        # real: < 10% of regions are suspicious
        # tampered: > 50% of regions are suspicious
        pct_suspicious = float(np.mean(chi2_arr > 0.35))
        score_pct = linear_score(pct_suspicious, 0.10, 0.50)

        cross_boundary_score = 0.60 * score_mean + 0.40 * score_pct

    # ----------------------------------
    # signal 2: LBP texture (entropy gap)
    # ----------------------------------
    # entropy measures how much texture detail a patch has (derived from the LBP map)
    # gap = difference in average entropy between patches inside vs outside the shadow
    # real: gap is small — same surface on both sides, same texture complexity
    # suspicious: gap is large — shadow was placed over a region with very different detail
    # gap <= 0.15: similar complexity on both sides → not suspicious
    # gap >= 0.60: very different complexity → suspicious
    entropy_gap = float(features.get("patch_entropy_s40_in_out_gap", 0.0))
    lbp_score = linear_score(entropy_gap, 0.15, 0.60)

    tamper_score = 0.60 * cross_boundary_score + 0.40 * lbp_score

    return max(0.0, min(1.0, tamper_score))


# ----------------------------------------------------------
# VISUALIZATION HELPERS
# ----------------------------------------------------------
def visualize_texture_analysis(img, mask, L8, lbp, chi2_list, patch_pairs, max_pairs_vis=200, patches_for_vis=None):
    '''
    visualize the texture analysis results with maps
    '''
    # outline of the mask (shadow boundaries)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_outline = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, k)

    # make LBP texture map 
    # lbp = lbp_map(L8)
    lbp_vis = cv2.normalize(lbp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # outline the shadow areas (using mask boundaries)
    lbp_color = cv2.applyColorMap(lbp_vis, cv2.COLORMAP_BONE)
    lbp_color[mask_outline > 0] = (255, 0, 0)

    # show shadow mask
    mask_overlay = img.copy()
    mask_overlay[mask == 255] = (0, 0, 255)  # red mask overlay
    mask_overlay = cv2.addWeighted(img, 0.7, mask_overlay, 0.3, 0)

    cv2.namedWindow("Shadow Mask", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Shadow Mask", 800, 600)
    cv2.imshow("Shadow Mask", mask)

    cv2.namedWindow("Overlay", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Overlay", 800, 600)
    cv2.imshow("Overlay", mask_overlay)

    cv2.waitKey(1)

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
        cv2.rectangle(overlay_pairs, (x_in, y_in), (x_in + s, y_in + s), (0, 0, 255), 2)
        # green box = outside shadow (lit area)
        cv2.rectangle(overlay_pairs, (x_out, y_out), (x_out + s, y_out + s), (0, 255, 0), 2)
        # label pairs
        cv2.putText(overlay_pairs, str(i), (x_in, max(0, y_in - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1, cv2.LINE_AA)
        cv2.putText(overlay_pairs, str(i), (x_out, max(0, y_out - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1, cv2.LINE_AA)                

    overlay_pairs[mask_outline > 0] = (255, 255, 255)
    cv2.namedWindow("Comparison Patches (red=inside shadow, green=outside shadow)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Comparison Patches (red=inside shadow, green=outside shadow)", 1000, 750)
    cv2.imshow("Comparison Patches (red=inside shadow, green=outside shadow)", overlay_pairs)

    # chi-squared histogram (similarity scores)
    if len(chi2_list) > 0:
        plt.figure(figsize=(8,5))
        plt.hist(chi2_list, bins=40)
        plt.title("Texture Similarity Scores - LBP chi-squared distances")
        plt.xlabel("Chi-squared distance (lower = more similar)")
        plt.ylabel("Number of shadow regions")
        plt.tight_layout()
        plt.show()

    # draw boxes around patch pairs that look suspiciously similar
    if patches_for_vis is not None and len(patches_for_vis) >= 2:
        clone_overlay = img.copy()

        for i in range(len(patches_for_vis)):
            for j in range(i + 1, len(patches_for_vis)):
                similarity = float(np.dot(patches_for_vis[i]["vec"], patches_for_vis[j]["vec"]))
                if similarity >= 0.95:
                    # get the top-left corner of each patch
                    r_i, c_i = patches_for_vis[i]["row"], patches_for_vis[i]["col"]
                    r_j, c_j = patches_for_vis[j]["row"], patches_for_vis[j]["col"]
                    # red box around each suspicious patch
                    clone_patch_size = 40
                    cv2.rectangle(clone_overlay, (c_i, r_i), (c_i + clone_patch_size, r_i + clone_patch_size), (0, 0, 255), 2)
                    cv2.rectangle(clone_overlay, (c_j, r_j), (c_j + clone_patch_size, r_j + clone_patch_size), (0, 0, 255), 2)
                    # orange line connecting the two similar patches
                    # cv2.line(clone_overlay, (c_i + 12, r_i + 12), (c_j + 12, r_j + 12), (0, 165, 255), 1)            

        cv2.namedWindow("Clone Stamp Detection (red = suspicious pairs)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Clone Stamp Detection (red = suspicious pairs)", 1000, 750)
        clone_overlay[mask_outline > 0] = (255, 255, 255)
        cv2.imshow("Clone Stamp Detection (red = suspicious pairs)", clone_overlay)

    # patch entropy heatmap
    # build a blank heatmap the same size as the image
    height_img, width_img = L8.shape[:2]
    patch_size_ent = 32
    entropy_heatmap = np.zeros((height_img, width_img), dtype=np.float32)

    for y in range(0, height_img - patch_size_ent + 1, patch_size_ent):
        for x in range(0, width_img - patch_size_ent + 1, patch_size_ent):
            patch = lbp[y:y + patch_size_ent, x:x + patch_size_ent].ravel()
            hist = np.bincount(patch, minlength=256).astype(np.float32)
            small_value = 1e-12
            hist /= hist.sum() + small_value
            entropy = -float(np.sum(hist * np.log(hist + small_value)))
            # fill the whole patch area with its entropy value
            entropy_heatmap[y:y + patch_size_ent, x:x + patch_size_ent] = entropy

    # normalize to 0-255 so we can colorize it (brighter = more texture detail)
    entropy_vis = cv2.normalize(entropy_heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    entropy_colored = cv2.applyColorMap(entropy_vis, cv2.COLORMAP_JET)

    # blend heatmap on top of original image so you can see where patches fall
    entropy_overlay = cv2.addWeighted(img, 0.5, entropy_colored, 0.5, 0)
    
    entropy_overlay[mask_outline > 0] = (255, 255, 255)

    cv2.namedWindow("Patch Entropy Heatmap (brighter = more texture detail)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Patch Entropy Heatmap (brighter = more texture detail)", 1000, 750)
    cv2.imshow("Patch Entropy Heatmap (brighter = more texture detail)", entropy_overlay)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ----------------------------------------------------------
# CHOOSING THE IMAGE - MAIN ANALYSIS
# ----------------------------------------------------------
def analyze_texture(image_input, visualize=True, compute_tamper_score=True, max_pairs_vis=200):
    '''
    analyze texture across shadow boundaries using LBP and return chi-square similarities
    - highlight the pairs of patches (one pair per shadow) that are compared
    - return raw chi-square distances for ML feature engineering
    '''
    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input))
    else:
        img = image_input

    print(f"Analyzing image: {image_input if isinstance(image_input, str) else 'provided as array'}")

    assert img is not None, f"Cannot read image: {image_input}"

    mask = final_shadow_mask(img)
    # mask should align with the original image
    assert mask.shape[:2] == img.shape[:2], "mask must align with original image"

    # convert intensity to 8-bit
    # create texture inputs from the original image, not the preprocessed image
    # L8_mask = np.uint8(np.clip(L, 0, 255))
    _, _, I0 = bgr_to_hsi_linear(img)
    L8_text = np.uint8(np.clip(I0*255, 0, 255))

    # calculate gradients to find outward direction of shadow
    gx = cv2.Sobel(L8_text, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L8_text, cv2.CV_32F, 0, 1, ksize=3)

    # make LBP texture map 
    lbp = lbp_map(L8_text)

    # EXACTLY one pair per component
    chi2_list, patch_pairs = pairs_one_per_component(
        mask_u8=mask,
        L8=L8_text,
        lbp=lbp,
        gx=gx,
        gy=gy,
        min_area=800,
        patch_size=25,
        in_offset=6,
        out_offset=9
    )

    # run clone detection and save patch positions for visualization
    # clone_feats, patches_for_vis = clone_detection(L8_text.astype(np.uint8), patch_size=40, max_patches=100, similar_thresh=0.95)
    patches_for_vis = None

    # extract features for ML (independent of rule-based scoring)
    features = extract_features(
        img,
        mask=mask,
        lbp=lbp,
        L8=L8_text, 
        gx=gx,
        gy=gy,
        chi2_list=chi2_list,
    )

    # IMPORTANT: disable tamper score until feature analysis and extraction is reliable
    # calculate tamper score (for rule-based path only)
    tamper_score = None
    if compute_tamper_score:
        tamper_score = calculate_texture_tamper_score(
            chi2_list=chi2_list, 
            features=features
        )
        print(f"\n{'='*60}")
        print(f"TEXTURE TAMPER SCORE: {tamper_score:.3f}")
        print(f"{'='*60}")

    if visualize:
        visualize_texture_analysis(
            img=img,
            mask=mask,
            L8=L8_text,
            lbp=lbp,
            chi2_list=chi2_list,
            patch_pairs=patch_pairs,
            max_pairs_vis=max_pairs_vis,
            patches_for_vis=patches_for_vis
        )

    # return results
    result = {
        "mask": mask, 
        "lbp": lbp,
        "L8": L8_text, 
        "chi2_distances": chi2_list,
        "patch_pairs": patch_pairs,
        "features": features,
    }

    # IMPORTANT: disable tamper score 
    # only include tamper_score if computed
    if tamper_score is not None:
        result["tamper_score"] = tamper_score

    return result

if __name__ == "__main__":
    result = analyze_texture("data/images/93-edited.jpg", visualize=True)
    print("\nExtracted features:", result["features"])
