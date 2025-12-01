# Third Feature: 
# - analyzing shadows through penumbra hardness (edge width)
# - direction consistency (shadow orientation)

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from typing import Dict, List, Tuple, Optional

from shadow_mask import final_shadow_mask, bgr_to_hsi_linear

# ----------------------------------------------------------
# VISUALIZE FILTERING - only for testing purposes
# ----------------------------------------------------------
def visualize_shadow_filtering(img, mask, min_shadow_area=300, min_perimeter=30):
    '''
    diagnostics: show which shadow regions are kept vs filtered as noise
    
    Use this to tune min_shadow_area and min_perimeter parameters
    
    function arguments:
    - img: Original BGR image
    - mask: Shadow mask from final_shadow_mask
    - min_shadow_area: Minimum area threshold (pixels²)
    - min_perimeter: Minimum perimeter threshold (pixels)
    
    show:
    - green: Shadow regions that WILL be analyzed (area >= threshold)
    - red: Shadow regions that will be SKIPPED as noise (area < threshold)
    '''
    mask_u8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # create visualization
    vis = img.copy()
    
    kept_count = 0
    skipped_count = 0
    kept_areas = []
    skipped_areas = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, closed=True)
        
        if area < min_shadow_area or perimeter < min_perimeter:
            # draw in RED - will be skipped
            cv2.drawContours(vis, [contour], -1, (0, 0, 255), 2)
            skipped_count += 1
            skipped_areas.append(area)
        else:
            # draw in GREEN - will be analyzed
            cv2.drawContours(vis, [contour], -1, (0, 255, 0), 2)
            kept_count += 1
            kept_areas.append(area)
    
    # CONSOLE OUTPUT OF STATISTICS
    # print(f"\n{'='*60}")
    # print(f"SHADOW REGION FILTERING DIAGNOSTICS")
    # print(f"{'='*60}")
    # print(f"Total regions found: {len(contours)}")
    # print(f"  Kept (GREEN):   {kept_count} regions")
    # if kept_areas:
    #     print(f"    Area range: {min(kept_areas):.0f} - {max(kept_areas):.0f} px-squared")
    # print(f"  Skipped (RED):  {skipped_count} regions")
    # if skipped_areas:
    #     print(f"    Area range: {min(skipped_areas):.0f} - {max(skipped_areas):.0f} px-squared")
    # print(f"\nCurrent thresholds:")
    # print(f"  min_shadow_area = {min_shadow_area} px-squared")
    # print(f"  min_perimeter   = {min_perimeter} px")
    # print(f"\nAdjustment tips:")
    # if kept_count == 0:
    #     print(f"  No regions kept, decrease min_shadow_area (try {min_shadow_area//2})")
    # elif skipped_count == 0:
    #     print(f"  No regions filtered, image may have noise. Increase min_shadow_area")
    # else:
    #     print(f"  Reasonable thresholds")
    # print(f"{'='*60}\n")
    
    # display
    cv2.namedWindow("Shadow Filtering (GREEN=keep, RED=skip)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Shadow Filtering (GREEN=keep, RED=skip)", 1000, 750)
    cv2.imshow("Shadow Filtering (GREEN=keep, RED=skip)", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ----------------------------------------------------------
# FEATURE EXTRACTION HELPERS
# ----------------------------------------------------------
def mean_std(arr):
    # calculate mean and std dev
    arr = np.asarray(arr).ravel()
    arr = arr[np.isfinite(arr)]
    # edge case
    if arr.size == 0:
        return 0.0, 0.0
    return float(np.mean(arr)), float(np.std(arr))

def percentile(arr, p):
    # calculate percentile
    arr = np.asarray(arr).ravel()
    arr = arr[np.isfinite(arr)]
    # edge case
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, p))

# ----------------------------------------------------------
# PENUMBRA HARDNESS
# ----------------------------------------------------------
def estimate_edge_width_contrast(profile, k_side=5, min_delta=5.0):
    '''
    measure how hard or soft a shadow edge is

    idea:
    - a shadow's edge relates to its penumbra, and typically a true shadow has a softer edge
        - hard shadow: sharp edge, instant transition (width is about 1-2 pixels)
        - soft shadow: gradual fade, slow transition (width is about 10-20 pixels) ** MAY NEED TO CHANGE VALUES FOR THIS **
    
    how we measured it:
    - look at a line of pixels crossing the shadow edge and find the following:
        - how bright is the shadow side (inside value)
        - how bright is the lit side (outside value)
        - how many pixels does it take to go from 10% to 90% of that transition
    
    function's arguments:
    - profile: array of intensity values crossing shadow edge
        ex: [50, 52, 55, 60, 70, 85, 100, 105, 110]
            |shadow|   |transition|     |lit|
    - k_side: how many pixels to average for inside/outside (default set to 5)
    - min_delta: minimum brightness difference to be a real edge (default set to 5)

    return:
    - width_px: how wide the transition is in pixels
    - contrast: how much darker is the shadow (intensity difference)
    '''
    profile = np.asarray(profile, dtype=np.float32)
    n = profile.size

    # need enough pixels to measure
    if n < 2 * k_side + 3:
        return 0.0, 0.0
    
    # measure average brightness on each side
    inside = float(np.mean(profile[:k_side]))   # first k pixels (shadow side)
    outside = float(np.mean(profile[-k_side:]))     # last k pixels (lit side)

    # profile should go from dark to bright
    # if it goes backwards (bright to dark), flip it
    if inside > outside:
        profile = profile[::-1]
        inside, outside = outside, inside

    # how much darker is the shadow
    delta = outside - inside

    # if there's barely any difference, this shouldn't be counted as a real shadow edge
    if abs(delta) < min_delta:
        return 0.0, 0.0
    
    # find the 10% to 90% transition points
    # this is where the width of the penumbra is measured
    low_val = inside + 0.1 * delta    # 10% of the way from dark to light
    high_val = inside + 0.9 * delta     # 90% of the way from dark to light

    def first_crossing(target):
        # find where the profile first crosses a target value
        for i in range(n-1):
            if profile[i] <= target <= profile[i+1]:
                # using linear interpolation for sub-pixel accuracy
                if abs(profile[i+1] - profile[i]) < 1e-6:
                    return float(i)
                t = (target - profile[i]) / (profile[i+1] - profile[i])
                return i + t
        return -1.0
    
    i_low = first_crossing(low_val)
    i_high = first_crossing(high_val)

    # if both crossing points are not found, count it as 0.0 (invalid measurement)
    if i_low < 0 or i_high < 0 or i_high <= i_low:
        return 0.0, 0.0
    
    # width is the distance between 10% to 90% points
    width_px = float(i_high - i_low)
    contrast = float(delta)

    return width_px, contrast

# ----------------------------------------------------------
# SAMPLE PROFILES ALONG SHADOW EDGES
# ----------------------------------------------------------
def sample_profiles_along_contour(
    contour,
    gray_img,
    mask_bin,
    step=4,
    half_len=10,
    num_samples=21      
):
    '''
    sample many profiles perpendicular to the shadow boundary
    - you have shadow boundary from the mask
    - at many points along that boundary, draw a line perpendicular to it
    - each line goes from inside the shadow -> across the edge -> into the lit area
    - measure the intensity values along each line
    - analyze each line to measure the penumbra width

    function arguments:
    - contour: shadow boundary from findContours
    - gray_img: grayscale intensity image
    - mask_bin: binary shadow mask (255=shadow, 0=lit)
    - step: sample every n points along the contour (8 = every 8th point)
    - half_len: how far to extend the line on each side of the boundary (pixels)
    - num_samples: how many intensity samples to take along each line

    return:
    - widths: list of measured edge widths (one per sample location)
    - contrasts: list of measured contrasts (one per sample location)
    - center_ys: list of y-coords where samples are (to visualize)
    - debug_points: list of (x,y) points where samples are (to visualize)
    '''
    contour = contour.squeeze(1)    # shape is (n,2) where each row is [x,y]
    n_points = len(contour)

    widths, contrasts, center_ys, debug_points = [], [], [], []

    if n_points < 3:
        return widths, contrasts, center_ys, debug_points
    
    h, w = gray_img.shape

    # calculate perpendicular directions at each point
    # for each point on the shadow boundary, find which direction is perpendicular to the boundary (right angle)
    normals = []
    for i in range(n_points):
        # get the point before and after this one
        p_prev = contour[(i-1) % n_points]
        p_next = contour[(i+1) % n_points]

        # tangent is the direction along the boundary
        tangent = p_next.astype(np.float32) - p_prev.astype(np.float32)
        norm = np.linalg.norm(tangent)

        if norm < 1e-3: 
            normals.append(np.array([0.0, 0.0], dtype=np.float32))
            continue

        tangent /= norm

        # normal is perpendicular to the tangent (rotate 90 degrees)
        # if tangent is [dx, dy], normal is [-dy, dx]
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
        normals.append(normal)

    sampled_count = 0
    rejected_no_crossing = 0
    rejected_invalid_measurement = 0

    # sample profiles at regular steps
    for idx in range(0, n_points, step):
        sampled_count += 1
        p = contour[idx].astype(np.float32)     # current boundary point [x,y]
        nrm = normals[idx]

        if np.linalg.norm(nrm) < 1e-3:
            continue

        # create a line perpendicular to the boundary
        # line goes from (p - half_len*normal) to (p + half_len*normal)
        ts = np.linspace(-half_len, half_len, num_samples).astype(np.float32)
        # x-coords along the line
        xs = p[0] + ts * nrm[0]
        # y-coords along the line
        ys = p[1] + ts * nrm[1]

        # convert to integer pixel coords (and clip to image bounds)
        xs_i = np.clip(np.round(xs).astype(int), 0, w-1)
        ys_i = np.clip(np.round(ys).astype(int), 0, h-1)

        # extract intensity values along this line
        profile = gray_img[ys_i, xs_i].astype(np.float32)

        # verify this line acrosses the shadow boundary
        # skip lines that are entirely in shadow or entirely in lit area
        center_idx = num_samples // 2
        center_in_shadow = mask_bin[ys_i[center_idx], xs_i[center_idx]] > 0

        if center_in_shadow:
            # if center is in shadow, make sure some part is lit
            if np.all(mask_bin[ys_i, xs_i] > 0):
                # skip if entire line is in shadow
                rejected_no_crossing += 1
                continue    
        else:
            # if center is lit, make sure some part is shadow
            if np.all(mask_bin[ys_i, xs_i] == 0):
                # skip if entire line is in lit
                rejected_no_crossing += 1
                continue
        
        # measure the penumbra width for this profile
        width_px, contrast = estimate_edge_width_contrast(profile)

        # skip if the measurement failed
        if width_px <= 0 or contrast <= 0:
            rejected_invalid_measurement += 1
            continue
        
        # store the results
        widths.append(width_px)
        contrasts.append(contrast)
        center_ys.append(float(p[1]))
        debug_points.append((int(p[0]), int(p[1])))

    # CONSOLE DEBUGGING
    # if sampled_count > 0:
    #     print(f"    [DEBUG] Sampled {sampled_count} locations (every {step} pixels)")
    #     print(f"    [DEBUG] Rejected {rejected_no_crossing} (no boundary crossing)")
    #     print(f"    [DEBUG] Rejected {rejected_invalid_measurement} (invalid measurement)")
    #     print(f"    [DEBUG] Kept {len(widths)} valid measurements")

    return widths, contrasts, center_ys, debug_points

# ----------------------------------------------------------
# DIRECTION CONSISTENCY ANALYSIS
# ----------------------------------------------------------
def estimate_shadow_direction(contour, min_points=20, min_elongation=2.0):
    '''
    find the dominant orientation of a single shadow section

    we get an axis orientation only from this (no arrow direction):
    - angles are in [0, 180) 

    function arguments:
    - contour: opencv contour (n x 1 x 2 array of boundary points)
    - min_points: minimum amount of points for a reliable analysis
    - min_elongation: ratio of longest/shortest axis

    return:
    - orientation: axis orientation in degrees [0,180), or none if invalid
    '''
    points = contour.reshape(-1, 2).astype(np.float32)
    if points.shape[0] < min_points:
        return None

    M = cv2.moments(contour)
    if abs(M["m00"]) < 1e-6:
        return None
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    center = np.array([cx, cy], dtype=np.float32)

    vecs = points - center
    dists = np.linalg.norm(vecs, axis=1)
    if dists.size == 0:
        return None

    max_r = float(np.max(dists))
    median_r = float(np.median(dists))

    # radial elongation: 1 => round; >2 => one strong "tail"
    elongation = max_r / max(median_r, 1e-3)
    if elongation < min_elongation:
        return None

    idx_far = int(np.argmax(dists))
    vx, vy = vecs[idx_far]

    # math coords (x right, y up) -> flip vy
    angle_rad = np.arctan2(-vy, vx)
    angle_deg = np.degrees(angle_rad) % 360.0

    return float(angle_deg)

def circular_stats(angles_deg):
    '''
    calculate mean angle and circular std dev for shadow orientations

    process:
    - convert each angle to a unit vector
        - 0 deg -> vector [1,0]
        - 90 deg -> vector [0,1]
        - 180 deg -> vector [-1,0]
    - average all the vectors
    - convert back to angle

    function arguments:
    - angles_deg: list of angles in degrees [0, 360]

    return:
    - mean_angle_deg: avg direction (0 to 360)
    - std_angle_deg: spread of directions (0 = all aligned, 180 = random)
        - if std_angle_deg is large, that means very inconsistent
    '''
    if len(angles_deg) == 0:
        return 0.0, 0.0
    
    # convert degrees to radians for calculating
    angles_rad = np.deg2rad(angles_deg)

    # convert each angle to a unit vector and average them
    # circular mean
    sin_sum = np.mean(np.sin(angles_rad))
    cos_sum = np.mean(np.cos(angles_rad))

    # convert average vector back to angle
    mean_angle_rad = np.arctan2(sin_sum, cos_sum)
    
    # calculate mean resultant length (R)
    # R = 1 means all vectors point exactly the same way
    # R = 0 means vectors cancel out (random directions)
    R = np.sqrt(sin_sum**2 + cos_sum**2)
    
    # circular standard deviation
    # when R is close to 1 (aligned), std is small
    # when R is close to 0 (scattered), std is large
    std_angle_rad = np.sqrt(-2 * np.log(R + 1e-9))
    
    # convert back to degrees
    mean_angle_deg = float(np.degrees(mean_angle_rad))
    std_angle_deg = float(np.degrees(std_angle_rad))
    
    # normalize mean to [0,360]
    if mean_angle_deg < 0:
        mean_angle_deg += 360.0
    
    return mean_angle_deg, std_angle_deg

def axial_circular_stats(angles_deg):
    '''
    circular mean and std dev for axial data (0° and 180° are equivalent).

    angles_deg: list of orientations in degrees [0,180)

    returns:
    - mean_angle_deg: mean orientation in [0,180)
    - std_angle_deg: circular std deviation in degrees
    '''
    if len(angles_deg) == 0:
        return 0.0, 0.0

    angles_rad = np.deg2rad(angles_deg)

    # double angles so 0° and 180° map to same direction
    doubled = 2.0 * angles_rad
    sin_sum = np.mean(np.sin(doubled))
    cos_sum = np.mean(np.cos(doubled))

    mean_doubled = np.arctan2(sin_sum, cos_sum)
    R = np.sqrt(sin_sum**2 + cos_sum**2)

    std_doubled = np.sqrt(-2.0 * np.log(R + 1e-9))

    mean_rad = mean_doubled / 2.0
    std_rad = std_doubled / 2.0

    mean_deg = np.degrees(mean_rad)
    std_deg = np.degrees(std_rad)

    if mean_deg < 0:
        mean_deg += 180.0

    return float(mean_deg), float(std_deg)

def enforce_global_direction(angle_deg, global_dir_deg):
    '''
    given a full direction angle_deg and a desired global direction,
    flip by 180° if it's more than 90° away from the global direction

    both inputs are in degrees
    '''
    a = angle_deg % 360.0
    g = global_dir_deg % 360.0

    # smallest signed difference in [-180, 180]
    diff = ((a - g + 180.0) % 360.0) - 180.0

    if abs(diff) > 90.0:
        a = (a + 180.0) % 360.0

    return a

# ----------------------------------------------------------
# FEATURE EXTRACTION
# ----------------------------------------------------------
def extract_features(widths, contrasts, directions):
    '''
    extract ML features from depth analysis based on penumbra hardness (width) and direction consistency

    penumbra width: 
    - mean/median: typical penumbra width (small = hard shadows; nearby light source) (large = soft shadows; distant/diffuse light)
    - std dev: how consistent the shadows are (low std = all shadows have similar hardness) (high std = shadows vary a lot)

    direction:
    - num_shadows: how many elongated shadows are found (0-2 is not good enough for reliability; 3-5 is good; 6+ is best)
    - mean_dir: average direction all shadows point toward (tells us general direction of where the light source is)
    - std_dir: how much directions vary (low std [<15] = all aligned, most likely single light source) (high std [>30] = scattered, might be multiple)

    function arguments:
    - widths: list of penumbra width measurements
    - contrasts: list of contrast measurements
    - directions: list of shadow direction angles (in degrees)

    return:
    - dict of features ready for analysis or ML
    '''
    ml_features = {}

    widths_arr = np.asarray(widths, dtype=np.float32)
    contrasts_arr = np.asarray(contrasts, dtype=np.float32)

    # penumbra width stats
    if widths_arr.size > 0:
        ml_features["edge_width_mean"] = float(np.mean(widths_arr))
        ml_features["edge_width_median"] = float(np.median(widths_arr))
        ml_features["edge_width_std"] = float(np.std(widths_arr))
        ml_features["edge_width_min"] = float(np.min(widths_arr))
        ml_features["edge_width_max"] = float(np.max(widths_arr))
        ml_features["edge_width_range"] = float(np.max(widths_arr) - np.min(widths_arr))
        ml_features["edge_width_p25"] = percentile(widths_arr, 25)
        ml_features["edge_width_p75"] = percentile(widths_arr, 75)    

        # Coefficient of variation (normalized std)
        if ml_features["edge_width_mean"] > 1e-3:
            ml_features["edge_width_cv"] = ml_features["edge_width_std"] / ml_features["edge_width_mean"]
        else:
            ml_features["edge_width_cv"] = 0.0
    else:
        ml_features["edge_width_mean"] = 0.0
        ml_features["edge_width_median"] = 0.0
        ml_features["edge_width_std"] = 0.0
        ml_features["edge_width_min"] = 0.0
        ml_features["edge_width_max"] = 0.0
        ml_features["edge_width_range"] = 0.0
        ml_features["edge_width_p25"] = 0.0
        ml_features["edge_width_p75"] = 0.0
        ml_features["edge_width_cv"] = 0.0

    # contrast statistics
    if contrasts_arr.size > 0:
        ml_features["edge_contrast_mean"] = float(np.mean(contrasts_arr))
        ml_features["edge_contrast_median"] = float(np.median(contrasts_arr))
        ml_features["edge_contrast_std"] = float(np.std(contrasts_arr))
        ml_features["edge_contrast_min"] = float(np.min(contrasts_arr))
        ml_features["edge_contrast_max"] = float(np.max(contrasts_arr))
    else:
        ml_features["edge_contrast_mean"] = 0.0
        ml_features["edge_contrast_median"] = 0.0
        ml_features["edge_contrast_std"] = 0.0
        ml_features["edge_contrast_min"] = 0.0
        ml_features["edge_contrast_max"] = 0.0

    # count of measurements
    ml_features["num_measurements"] = int(widths_arr.size)

    # direction stats
    if len(directions) > 0:
        mean_dir, std_dir = axial_circular_stats(directions)
        ml_features["num_dir_samples"] = int(len(directions))
        ml_features["dir_mean_deg"] = mean_dir
        ml_features["dir_std_deg"] = std_dir
    else:
        ml_features["num_dir_samples"] = 0
        ml_features["dir_mean_deg"] = 0.0
        ml_features["dir_std_deg"] = 0.0

    return ml_features

# ----------------------------------------------------------
# TAMPER SCORE
# ----------------------------------------------------------
def calculate_depth_tamper_score(ml_features):
    '''
    tamper score based on both penumbra hardness and direction consistency

    score components:
    - penumbra variability (50% weight)
        - measures if shadow hardness on shadow boundaries is consistent
        - high std = possibly mixing different lighting conditions (i.e., copy-pasting a shadow from one photo to another)
    - direction consistency (50% weight)
        - measures if all shadows point the same way
        - high std = possibly multiple/inconsistent light sources

    why this approach:
    - real photos typically have consistent hardness and aligned directions
    - fake photos may fail with at least one of these
    '''
    # penumbra hardness variability
    width_std = float(ml_features.get("edge_width_std", 0.0))
    width_cv = float(ml_features.get("edge_width_cv", 0.0))
    width_mean = float(ml_features.get("edge_width_mean", 1.0))
    width_median = float(ml_features.get("edge_width_median", 1.0))
    width_max = float(ml_features.get("edge_width_max", 1.0))
    
    # standard deviation for hardness variability
    if width_std <= 5.0:
        score_std = 0.0  # normal variation
    elif width_std >= 10.0:
        score_std = 1.0  # extreme variation
    else:
        # linear interpolation between 5 and 10
        score_std = (width_std - 5.0) / 5.0
    
    # coefficient of variation for hardnessvariability
    if width_cv <= 0.8:
        score_cv = 0.0  # normal
    elif width_cv >= 1.5:
        score_cv = 1.0  # extreme
    else:
        # linear interpolation
        score_cv = (width_cv - 0.8) / 0.7
    
    # range check (max vs median ratio)
    # real: max usually < 3x median
    ratio = width_max / max(width_median, 0.1)
    if ratio <= 3.0:
        score_range = 0.0
    elif ratio >= 6.0:
        score_range = 1.0
    else:
        score_range = (ratio - 3.0) / 3.0

    # combine components
    penumbra_score = (
        0.50 * score_std +      # primary indicator
        0.30 * score_cv +       # scale-independent check
        0.20 * score_range      # extreme outlier check
    )
    
    # direction consistency
    dir_std = float(ml_features.get("dir_std_deg", 0.0))
    num_dir = int(ml_features.get("num_dir_samples", 0))

    # check if there are enough direction samples
    if num_dir < 2:
        # not enough elongated shadows to analyze direction
        # fall back to penumbra only score
        direction_score = 0.0
        direction_weight = 0.0
        penumbra_weight = 1.0
    else:
        # use direction data
        direction_weight = 0.5
        penumbra_weight = 0.5

        # direction std dev
        # real: std dev typically < 15 deg (all shadows aligned)
        # fake: std dev > 30 deg (possibly inconsistent light sources)
        if dir_std <= 15.0:
            direction_score = 0.0
        elif dir_std >= 40.0:
            direction_score = 1.0
        else:
            # linear interpolation (if there are unknown values and you need to estimate) between 15 and 40 deg
            direction_score = (dir_std - 15.0) / 25.0

        # if very inconsistent, add penalty
        if dir_std > 60.0:
            direction_score = 1.0
    
    # combined score for penumbra analysis and direction analysis
    tamper_score = (
        penumbra_weight * penumbra_score + 
        direction_weight * direction_score
    )

    return max(0.0, min(1.0, tamper_score))    

# ----------------------------------------------------------
# VISUALIZATION
# ----------------------------------------------------------
def visualize_depth_analysis(img, mask, debug_points, widths, contrasts, direction_debug_info, all_directions, max_points_vis=2000):
    '''
    show what was measured so far:
    - shadow mask overlay
    - sample points (where did we measure)
    - histogram of edge widths (distribution of penumbra hardness)
    - histogram of contrasts (distribution of shadow darkness)
    - shadow directions with arrows
        - green arrows=direction of each elongated shadow
        - arrow length proportional to shadow size
    '''
    mask_u8 = mask.astype(np.uint8)
    
    # create boundary outline for visualization
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_outline = cv2.morphologyEx(mask_u8, cv2.MORPH_GRADIENT, kernel)

    # shadow overlay (red)
    mask_overlay = img.copy()
    mask_overlay[mask_u8 == 255] = (0, 0, 255)
    mask_overlay = cv2.addWeighted(img, 0.7, mask_overlay, 0.3, 0)

    # show mask
    cv2.namedWindow("Shadow Mask", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Shadow Mask", 800, 600)
    cv2.imshow("Shadow Mask", mask_u8)

    # show overlay with shadow regions highlighted
    cv2.namedWindow("Shadow Overlay", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Shadow Overlay", 800, 600)
    cv2.imshow("Shadow Overlay", mask_overlay)

    # show where we sampled (green dots)
    pts_overlay = mask_overlay.copy()
    for i, (x, y) in enumerate(debug_points[:max_points_vis]):
        cv2.circle(pts_overlay, (x, y), 3, (0, 255, 0), -1)
    
    # direction arrows
    arrows_overlay = mask_overlay.copy()
    if len(direction_debug_info) > 0 and len(all_directions) > 0:
        # use axial stats to get the main line of shadows
        '''
        NOTICE: some images have arrow visuals that seem "flipped" (180 deg off) since drawing the arrows only chooses one 
        global sign along that axis (ex: if the direction is 210 deg, the axis is the same as 30 deg)
        - both are valid mathematically; the alg has no other info to know which matches the sun
        - the 180 deg ambiguity doesn't change the analysis (only affects visuals)
        '''
        axes = [(d % 180.0) for d in all_directions]
        global_axis_deg, axis_std = axial_circular_stats(axes)

        # we align arrow directions to this axis
        # print(f"[VIS] Global shadow axis: {global_axis_deg:.1f}, std={axis_std:.1f}")

        for info in direction_debug_info:
            cx, cy = info['center']
            local_dir = info['angle']  # 0–360°
            area = info['area']

            # flip local_dir by 180° if it's >90° away from axis
            angle_deg = enforce_global_direction(local_dir, global_axis_deg)

            # arrow length
            arrow_len = min(int(np.sqrt(area) / 3), 80)
            arrow_len = max(arrow_len, 30)

            angle_rad = np.deg2rad(angle_deg)
            dx = int(arrow_len * np.cos(angle_rad))
            dy = int(-arrow_len * np.sin(angle_rad))  # flip y for image coords

            end_x = cx + dx
            end_y = cy + dy

            cv2.arrowedLine(
                arrows_overlay, (cx, cy), (end_x, end_y),
                (255, 0, 0), 2, tipLength=0.3
            )
            cv2.circle(arrows_overlay, (cx, cy), 5, (0, 255, 255), -1)

            label = f"{int(angle_deg)}"
            cv2.putText(
                arrows_overlay, label, (cx + 10, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA
            )

    cv2.namedWindow("Penumbra Sample Points (green dots)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Penumbra Sample Points (green dots)", 800, 600)
    cv2.imshow("Penumbra Sample Points (green dots)", pts_overlay)

    cv2.namedWindow("Directions (blue arrows)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Directions (blue arrows)", 800, 600)
    cv2.imshow("Directions (blue arrows)", arrows_overlay)

    cv2.waitKey(1)

    # histograms
    widths_arr = np.asarray(widths, dtype=np.float32)
    contrasts_arr = np.asarray(contrasts, dtype=np.float32)

    if widths_arr.size > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # width histogram
        axes[0].hist(widths_arr, bins=30, edgecolor='black', alpha=0.7)
        axes[0].axvline(np.mean(widths_arr), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(widths_arr):.2f}px')
        axes[0].axvline(np.median(widths_arr), color='blue', linestyle='--', linewidth=2,
                       label=f'Median: {np.median(widths_arr):.2f}px')
        axes[0].set_xlabel('Penumbra Width (pixels)', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].set_title('Distribution of Shadow Edge Widths', fontsize=14)
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # contrast histogram
        axes[1].hist(contrasts_arr, bins=30, edgecolor='black', alpha=0.7)
        axes[1].axvline(np.mean(contrasts_arr), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(contrasts_arr):.1f}')
        axes[1].axvline(np.median(contrasts_arr), color='blue', linestyle='--', linewidth=2,
                       label=f'Median: {np.median(contrasts_arr):.1f}')
        axes[1].set_xlabel('Shadow Contrast (intensity units)', fontsize=12)
        axes[1].set_ylabel('Count', fontsize=12)
        axes[1].set_title('Distribution of Shadow Darkness', fontsize=14)
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def analyze_depth(image_input, visualize=True, sample_step=4, compute_tamper_score=True, min_shadow_area=300, min_perimeter=30):
    # load image
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
    else:
        img = image_input

    assert img is not None, f"Cannot read image: {image_input}"

    # shadow mask
    mask = final_shadow_mask(img)
    assert mask.shape[:2] == img.shape[:2], "mask must align with original image"

    # convert to grayscale
    _, _, I0 = bgr_to_hsi_linear(img)
    gray_img = np.uint8(np.clip(I0 * 255, 0, 255))

    # find shadow boundaries
    mask_u8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(f"Found {len(contours)} shadow regions")

    # sample profiles and measure penumbra
    all_widths = []
    all_contrasts = []
    all_center_ys = []
    debug_points_all = []

    # use configurable filtering parameters
    MIN_SHADOW_AREA_PENUMBRA = min_shadow_area
    MIN_PERIMETER_PENUMBRA = min_perimeter

    # more permissive parameters for direction-only regions
    # MIN_SHADOW_AREA_DIR = max(50, 0.2 * min_shadow_area)
    # MIN_PERIMETER_DIR   = max(20, 0.5 * min_perimeter)
    MIN_SHADOW_AREA_DIR = min_shadow_area
    MIN_PERIMETER_DIR   = min_perimeter

    skipped_small = 0
    measured_regions = 0

    for i, contour in enumerate(contours):
        # calculate area and perimeter
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, closed=True)

        # skip small regions that may likely be noise
        if area < MIN_SHADOW_AREA_PENUMBRA or perimeter < MIN_PERIMETER_PENUMBRA:
            skipped_small += 1
            continue 

        widths, contrasts, center_ys, debug_points = sample_profiles_along_contour(
            contour,
            gray_img=gray_img,
            mask_bin=mask_u8,
            step=sample_step,          # sample every 4 pixels along boundary
            half_len=10,     # extend +-10 pixels from boundary
            num_samples=21   # 21 intensity samples per profile
        )
        all_widths.extend(widths)
        all_contrasts.extend(contrasts)
        all_center_ys.extend(center_ys)
        debug_points_all.extend(debug_points)
        
        if widths:
            measured_regions += 1
            # print(f"  Region {measured_regions}: area={area:.0f}px-squared, {len(widths)} measurements")
    
    # if skipped_small > 0:
    #     print(f"    (Skipped {skipped_small} small regions as noise)")

    # collect shadow directions for direction consistency check
    all_directions = []
    direction_debug_info = []   # for visualization

    # loop through the same contours used for penumbra
    for i, contour in enumerate(contours):
        # use same filtering criteria as penumbra
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, closed=True)

        # direction loop
        if area < MIN_SHADOW_AREA_DIR or perimeter < MIN_PERIMETER_DIR:
            # skip small regions that may be noise
            continue

        # estimate direction of this shadow (relax min_points and min_elongation to catch poles)
        angle_deg = estimate_shadow_direction(
            contour, 
            min_points=20,  # need at least 20 boundary points
            min_elongation=2.0  # only measure elongated shadows
        )

        if angle_deg is not None:
            # valid direction found
            all_directions.append(angle_deg)

            # store debug info for visualization
            M = cv2.moments(contour)
            if abs(M["m00"]) > 1e-6:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                direction_debug_info.append({
                    'center': (cx, cy),
                    'angle': angle_deg,
                    'area': area
                })            

    # CONSOLE print direction summary
    if len(all_directions) > 0:
        # axes (0–180) for consistency
        axes = [(d % 180.0) for d in all_directions]
        mean_axis, std_axis = axial_circular_stats(axes)

        # align all directions to that axis for a signed mean (optional, for info)
        aligned_dirs = [enforce_global_direction(d, mean_axis) for d in all_directions]
        mean_dir, dir_spread = circular_stats(aligned_dirs)

        # print(f"  Found {len(all_directions)} elongated shadows")
        # print(f"  Mean axis: {mean_axis:.1f} deg (0≡180), axis std: {std_axis:.1f} deg")
        # print(f"  Mean direction (aligned): {mean_dir:.1f} deg, spread: {dir_spread:.1f} deg")
        
    #     if std_axis < 15:
    #         print(f"  → Very consistent (aligned shadows)")
    #     elif std_axis < 30:
    #         print(f"  → Moderately consistent")
    #     else:
    #         print(f"  → High variation (possibly different light sources)")
    # else:
    #     print(f"  No elongated shadows found (all too circular)")
    #     print(f"  Direction consistency cannot be assessed")

    # calculate stats for penumbra width
    # print(f"\n{'='*60}")
    # print(f"PENUMBRA HARDNESS ANALYSIS")
    # print(f"{'='*60}")
    # print(f"Total measurements: {len(all_widths)}")
    
    # if len(all_widths) > 0:
    #     print(f"\nEdge Width (Penumbra Hardness):")
    #     print(f"  Mean:   {np.mean(all_widths):.2f} pixels")
    #     print(f"  Median: {np.median(all_widths):.2f} pixels")
    #     print(f"  Std:    {np.std(all_widths):.2f} pixels")
    #     print(f"  Range:  {np.min(all_widths):.2f} - {np.max(all_widths):.2f} pixels")
        
    #     print(f"\nShadow Contrast:")
    #     print(f"  Mean:   {np.mean(all_contrasts):.1f} intensity units")
    #     print(f"  Median: {np.median(all_contrasts):.1f} intensity units")
    #     print(f"  Range:  {np.min(all_contrasts):.1f} - {np.max(all_contrasts):.1f}")
    # else:
    #     print("\nWARNING: No valid penumbra measurements found")
    #     print("Possibilities:")
    #     print("  - Shadow edges are too small/fragmented")
    #     print("  - Shadows are too hard (width < min threshold)")
    #     print("  - Not enough contrast at boundaries")

    features = extract_features(all_widths, all_contrasts, all_directions)

    # calculate tamper score
    tamper_score = None
    if compute_tamper_score and len(all_widths) > 0:
        tamper_score = calculate_depth_tamper_score(features)
        print(f"\n{'='*60}")
        print(f"PENUMBRA TAMPER SCORE: {tamper_score:.3f}")
        print(f"{'='*60}")

    # visualize
    if visualize and len(all_widths) > 0:
        visualize_depth_analysis(
            img,
            mask_u8,
            debug_points_all,
            all_widths,
            all_contrasts,
            direction_debug_info,
            all_directions,
            max_points_vis=2000
        )

    # return results
    result = {
        "mask": mask_u8,
        "gray_img": gray_img,
        "widths": all_widths,
        "contrasts": all_contrasts,
        "center_ys": all_center_ys,
        "features": features
    }
    
    if tamper_score is not None:
        result["tamper_score"] = tamper_score

    return result

# ----------------------------------------------------------
# TEST
# ----------------------------------------------------------
if __name__ == "__main__":
    # testing visuals before analyzing
    img = cv2.imread("data/images/20-edited.jpg")
    mask = final_shadow_mask(img)

    # visualize_shadow_filtering(
    #     img, 
    #     mask,
    #     min_shadow_area=300,  # adjust these based on visualization
    #     min_perimeter=30
    # )       

    result = analyze_depth(
        "data/images/20-edited.jpg",
        visualize=True,
        sample_step=4,
        compute_tamper_score=True,
        min_shadow_area=300,
        min_perimeter=30
    )

    print("\n" + "="*60)
    print("EXTRACTED DEPTH FEATURES")
    print("="*60)
    for key, value in result["features"].items():
        print(f"{key:30s}: {value}")
