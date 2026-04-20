# Third Feature:
# - analyzing shadows through penumbra hardness (edge width)
# - elongation consistency (shadow shape)

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from pathlib import Path

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
def estimate_edge_width_contrast(profile, side_window=4, min_contrast=5.0, smooth_size=5):
    '''
    estimate how soft or hard a shadow edge is from a 1D intensity profile

    idea:
    - a shadow's edge relates to its penumbra, and typically a true shadow has a softer edge
        - hard shadow: sharp edge, instant transition (width is about 1-2 pixels)
        - soft shadow: gradual fade, slow transition (width is about 10-20 pixels) ** MAY NEED TO CHANGE VALUES FOR THIS **
    
    how we measured it:
    - we take a line of pixel values that crosses a shadow boundary
    - the shadow side should be darker
    - the lit side should be brighter
    - the penumbra width is how long it takes to go from 10% to 90% of that brightness change
    
    return:
    - width_px: estimated penumbra width/how wide the transition is in pixels
    - contrast: brightness difference across edge/how much darker is the shadow
    '''
    profile = np.asarray(profile, dtype=np.float32)
    num_values = profile.size

    # need enough pixels to measure
    if num_values < 2 * side_window + 5:
        return 0.0, 0.0
    
    # smooth the profile a little bit so tiny texture changes do not dominate
    if smooth_size >= 3:
        smoothed_profile = cv2.GaussianBlur(
            profile.reshape(1, -1), (1, smooth_size), 0
        ).ravel()
    else:
        smoothed_profile = profile.copy()

    # the real boundary should be somewhere near the center of the sampled line
    center_index = num_values // 2

    # take the 1D gradient of the profile
    # this tells us where the biggest brightness change happens
    gradient = np.diff(smoothed_profile)

    # only search near the middle so we do not accidentally measure some other texture edge
    search_left = max(0, center_index - num_values // 4)
    search_right = min(len(gradient), center_index + num_values // 4)

    if search_right <= search_left:
        return 0.0, 0.0

    middle_gradient = gradient[search_left:search_right]

    # find the strongest edge near the center
    edge_index = search_left + int(np.argmax(np.abs(middle_gradient)))

    # estimate average brightness on the left side of the edge
    left_start = max(0, edge_index - side_window + 1)
    left_end = edge_index + 1

    # estimate average brightness on the right side of the edge
    right_start = edge_index + 1
    right_end = min(num_values, edge_index + 1 + side_window)

    if left_end <= left_start or right_end <= right_start:
        return 0.0, 0.0

    left_value = float(np.mean(smoothed_profile[left_start:left_end]))
    right_value = float(np.mean(smoothed_profile[right_start:right_end]))

    # force the profile to go from dark -> bright
    # if not, flip it so the rest of the math stays simple
    if left_value <= right_value:
        shadow_value = left_value
        lit_value = right_value
    else:
        shadow_value = right_value
        lit_value = left_value
        smoothed_profile = smoothed_profile[::-1]

    contrast = lit_value - shadow_value

    # skip weak edges that are probably not useful
    if contrast < min_contrast:
        return 0.0, 0.0

    # define the 10% and 90% brightness levels
    low_value = shadow_value + 0.1 * contrast
    high_value = shadow_value + 0.9 * contrast

    def first_crossing(target_value):
        '''
        find where the profile first crosses a target brightness level
        using linear interpolation for a smoother estimate
        '''
        for i in range(num_values - 1):
            a = smoothed_profile[i]
            b = smoothed_profile[i + 1]

            if (a <= target_value <= b) or (a >= target_value >= b):
                if abs(b - a) < 1e-6:
                    return float(i)

                t = (target_value - a) / (b - a)
                return float(i + t)

        return -1.0

    low_index = first_crossing(low_value)
    high_index = first_crossing(high_value)

    if low_index < 0 or high_index < 0 or high_index <= low_index:
        return 0.0, 0.0

    width_px = float(high_index - low_index)

    return width_px, float(contrast)

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
        boundary_point = contour[idx].astype(np.float32)     # current boundary point [x,y]
        normal_vector = normals[idx]

        if np.linalg.norm(normal_vector) < 1e-3:
            continue

        # create a line perpendicular to the boundary
        # line goes from (boundary_point - half_len*normal) to (boundary_point + half_len*normal)
        sample_offsets = np.linspace(-half_len, half_len, num_samples).astype(np.float32)
        # x-coords along the line
        x_coords = boundary_point[0] + sample_offsets * normal_vector[0]
        # y-coords along the line
        y_coords = boundary_point[1] + sample_offsets * normal_vector[1]

        # convert to integer pixel coords (and clip to image bounds)
        x_pixels = np.clip(np.round(x_coords).astype(int), 0, w-1)
        y_pixels = np.clip(np.round(y_coords).astype(int), 0, h-1)

        # extract intensity values along this line
        profile = gray_img[y_pixels, x_pixels].astype(np.float32)

        # check whether this sampled line crosses a shadow boundary
        # we want some of the line in shadow and some of it outside shadow
        line_mask = (mask_bin[y_pixels, x_pixels] > 0).astype(np.uint8)

        # skip if the whole line is only shadow or only non-shadow
        if np.all(line_mask == 0) or np.all(line_mask == 1):
            rejected_no_crossing += 1
            continue

        # also make sure the boundary is near the center of the profile
        # this helps avoid measuring unrelated edges
        center_index = num_samples // 2
        center_window = 3

        middle_part = line_mask[
            max(0, center_index - center_window) : min(num_samples, center_index + center_window + 1)
        ]

        if np.all(middle_part == 0) or np.all(middle_part == 1):
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
        center_ys.append(float(boundary_point[1]))
        debug_points.append((int(boundary_point[0]), int(boundary_point[1])))

    # CONSOLE DEBUGGING
    # if sampled_count > 0:
    #     print(f"[DEBUG] Sampled {sampled_count} locations (every {step} pixels)")
    #     print(f"[DEBUG] Rejected {rejected_no_crossing} (no boundary crossing)")
    #     print(f"[DEBUG] Rejected {rejected_invalid_measurement} (invalid measurement)")
    #     print(f"[DEBUG] Kept {len(widths)} valid measurements")

    return widths, contrasts, center_ys, debug_points

# ----------------------------------------------------------
# ELONGATION AND ORIENTATION CONSISTENCY ANALYSIS
# ----------------------------------------------------------
def axial_circular_stats(angles_deg):
    '''
    compute mean and std for axial angle data, where 0° and 180° are the same axis

    regular circular stats would treat 0° and 179° as nearly opposite, but for
    shadow orientations they are nearly identical (same shadow axis, just measured
    from the other end). the double-angle trick maps both to the same point on the
    unit circle before averaging.

    function arguments:
    - angles_deg: list of angles in degrees (any range, not just [0, 180))

    returns:
    - (mean_deg, std_deg): mean axis direction in [0, 180) and circular std in degrees
    '''
    if len(angles_deg) == 0:
        return 0.0, 0.0

    # normalize to [0, 180) — the fundamental domain for axial data
    angles = np.array(angles_deg, dtype=np.float64) % 180.0

    # double each angle so that 0 and 180 degrees map to the same unit-circle point
    doubled = np.deg2rad(2.0 * angles)

    sin_mean = np.mean(np.sin(doubled))
    cos_mean = np.mean(np.cos(doubled))

    # circular mean (in the doubled space), then halve back to get the axis mean
    mean_doubled_deg = np.degrees(np.arctan2(sin_mean, cos_mean))
    mean_deg = (mean_doubled_deg / 2.0) % 180.0

    # R is the length of the mean resultant vector — 1.0 = perfectly concentrated,
    # 0.0 = uniformly spread; use it to get a circular std
    R = np.sqrt(sin_mean ** 2 + cos_mean ** 2)
    circ_std_rad = np.sqrt(-2.0 * np.log(max(R, 1e-9)))
    # halve again: we doubled the angles, so the spread is also doubled
    std_deg = np.degrees(circ_std_rad / 2.0)

    return float(mean_deg), float(std_deg)


def estimate_shadow_elongation(contour, min_points=5):
    '''
    estimate the elongation ratio of a shadow using its minimum area
    bounding rectangle

    idea:
    - fit the tightest possible rectangle around the shadow contour
    - elongation = major axis / minor axis
    - all shadows in a real scene share the same sun elevation angle,
      so their elongation ratios should be consistent across the image
    - high variance in elongation = inconsistent light source = suspicious

    function arguments:
    - contour: shadow boundary from findContours
    - min_points: minimum contour points needed to attempt the fit

    returns:
    - elongation ratio (float >= 1.0), or None if contour is too small
    '''
    points = contour.reshape(-1, 2).astype(np.float32)
    if points.shape[0] < min_points:
        return None

    # fit the minimum area bounding rectangle around the contour
    # returns: center, (width, height), rotation angle
    _, (w, h), _ = cv2.minAreaRect(contour)

    # make sure major >= minor so ratio is always >= 1.0
    major = max(w, h)
    minor = min(w, h)

    # skip degenerate contours with no real size
    if minor < 1.0:
        return None

    return float(major / minor)

def collect_elongation_ratios(contours, min_shadow_area, min_perimeter, min_elongation=1.5):
    '''
    collect elongation ratios from all shadow contours that pass
    size and shape filters

    function arguments:
    - contours: list of shadow contours from findContours
    - min_shadow_area: skip contours smaller than this (pixels squared)
    - min_perimeter: skip contours with perimeter shorter than this
    - min_elongation: minimum ratio to include — shadows below this are
      too compact to have a meaningful elongation axis (nearly square),
      and including them would just add noise to the consistency check

    returns:
    - ratios: list of elongation ratios, one per qualifying shadow
    '''
    ratios = []

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, closed=True)

        if area < min_shadow_area or perimeter < min_perimeter:
            continue

        # skip very irregular shapes like leaves or fragmented noise
        # solidity = actual area / convex hull area
        # low solidity means the shape is messy and the bounding box is unreliable
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area < 1.0:
            continue
        if (area / hull_area) < 0.5:
            continue

        ratio = estimate_shadow_elongation(contour)
        if ratio is None:
            continue

        # skip compact shadows since they contribute no meaningful elongation signal
        if ratio < min_elongation:
            continue

        ratios.append(ratio)

    return ratios


def collect_orientation_angles(contours, min_shadow_area, min_perimeter, min_elongation=1.5):
    '''
    collect the long-axis orientation of each elongated shadow using minAreaRect

    idea:
    - every shadow in a real scene is cast by the same sun, so all long axes
      should point in roughly the same direction
    - a composited shadow from a different photo may have a different axis angle,
      which shows up as high circular std across the image

    this only uses shadows that are elongated enough (major/minor >= min_elongation)
    so compact blobs do not pollute the axis estimate

    function arguments:
    - contours: list of shadow contours from findContours
    - min_shadow_area: skip contours smaller than this (pixels squared)
    - min_perimeter: skip contours shorter than this
    - min_elongation: minimum major/minor ratio — below this the axis is unreliable

    returns:
    - angles: list of long-axis angles in [0, 180), one per qualifying shadow
    '''
    angles = []

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, closed=True)

        if area < min_shadow_area or perimeter < min_perimeter:
            continue

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area < 1.0:
            continue
        if (area / hull_area) < 0.5:
            continue

        points = contour.reshape(-1, 2).astype(np.float32)
        if points.shape[0] < 5:
            continue

        _, (w, h), angle = cv2.minAreaRect(contour)

        major = max(w, h)
        minor = min(w, h)
        if minor < 1.0:
            continue

        # skip shadows too compact to have a meaningful axis
        if (major / minor) < min_elongation:
            continue

        # minAreaRect reports the angle of the 'width' side relative to horizontal
        # if h > w the long axis is the height side, so rotate 90° to get the long axis
        if h > w:
            angle = angle + 90.0

        # normalize to [0, 180) — the axial fundamental domain
        angle = angle % 180.0

        angles.append(float(angle))

    return angles


# ----------------------------------------------------------
# FEATURE EXTRACTION
# ----------------------------------------------------------
def extract_features(widths, contrasts, elongation_ratios, orientation_angles):
    '''
    turn raw measurements into a small set of ML features

    goal:
    - keep only a few features
    - prefer stable features over too many detailed ones
    - this is better for a small dataset

    function arguments:
    - widths: list of penumbra edge width measurements
    - contrasts: list of shadow contrast measurements
    - elongation_ratios: list of per-shadow major/minor axis ratios
    - orientation_angles: list of per-shadow long-axis angles in [0, 180)
    '''
    ml_features = {}

    width_values = np.asarray(widths, dtype=np.float32)
    contrast_values = np.asarray(contrasts, dtype=np.float32)

    # -----------------------------
    # penumbra width features
    # -----------------------------
    if width_values.size > 0:
        width_mean = float(np.mean(width_values))
        width_std = float(np.std(width_values))
        width_median = float(np.median(width_values))

        ml_features["edge_width_mean"] = width_mean
        ml_features["edge_width_std"] = width_std
        ml_features["edge_width_median"] = width_median

        # coefficient of variation = std / mean
        # normalizes variability so images with wider penumbrae are not
        # unfairly penalized for having larger absolute spread
        if width_mean > 1e-3:
            ml_features["edge_width_cv"] = width_std / width_mean
        else:
            ml_features["edge_width_cv"] = 0.0
    else:
        ml_features["edge_width_mean"] = 0.0
        ml_features["edge_width_std"] = 0.0
        ml_features["edge_width_median"] = 0.0
        ml_features["edge_width_cv"] = 0.0

    # -----------------------------
    # shadow contrast feature
    # -----------------------------
    if contrast_values.size > 0:
        ml_features["edge_contrast_mean"] = float(np.mean(contrast_values))
    else:
        ml_features["edge_contrast_mean"] = 0.0

    # how many valid penumbra measurements we got
    ml_features["num_measurements"] = int(width_values.size)

    # -----------------------------
    # elongation consistency features
    # -----------------------------
    # elongation ratio = major axis / minor axis of each shadow's bounding box
    # all shadows in a real scene share the same sun elevation angle, so their
    # ratios should cluster together; high spread means inconsistent light sources
    if len(elongation_ratios) >= 2:
        ratio_arr = np.asarray(elongation_ratios, dtype=np.float32)
        ratio_mean = float(np.mean(ratio_arr))
        ratio_std  = float(np.std(ratio_arr))

        ml_features["elongation_mean"] = ratio_mean
        ml_features["elongation_std"] = ratio_std
        # CV normalizes std by mean so images with very elongated shadows
        # are not unfairly penalized for larger raw variance
        ml_features["elongation_cv"] = ratio_std / ratio_mean if ratio_mean > 1e-3 else 0.0
        ml_features["elongation_samples"] = int(ratio_arr.size)
    else:
        ml_features["elongation_mean"] = 0.0
        ml_features["elongation_std"] = 0.0
        ml_features["elongation_cv"] = 0.0
        ml_features["elongation_samples"] = 0

    # -----------------------------
    # orientation consistency features
    # -----------------------------
    # long-axis angle of each shadow's bounding rectangle in [0, 180)
    # all real shadows share the same sun azimuth, so their axes should cluster;
    # a composited shadow from a different photo will have a different axis angle
    if len(orientation_angles) >= 2:
        _, angle_std = axial_circular_stats(orientation_angles)
        ml_features["orientation_std_deg"]  = angle_std
        ml_features["orientation_samples"]  = int(len(orientation_angles))
    else:
        ml_features["orientation_std_deg"]  = 0.0
        ml_features["orientation_samples"]  = 0

    return ml_features

# ----------------------------------------------------------
# TAMPER SCORE
# ----------------------------------------------------------
def calculate_depth_tamper_score(ml_features):
    '''
    tamper score based on penumbra hardness and elongation consistency

    score components:
    - penumbra score (50% when shape features available, 100% otherwise):
        - std dev (65%): how much edge softness varies across all shadow boundaries
        - CV (35%): normalized variability, catches cases where std is inflated
          by large absolute widths
    - elongation score (25% when available):
        - CV of per-shadow major/minor axis ratios; real scenes have consistent
          elongation because the sun elevation angle is fixed
    - orientation score (25% when available):
        - circular std of per-shadow long-axis angles; real shadows share the same
          sun azimuth so their axes cluster tightly; a composited shadow from a
          different photo has a noticeably different axis

    weights fall back gracefully:
    - both shape features available: penumbra 50% + elongation 25% + orientation 25%
    - neither available (< 2 elongated shadows): penumbra 100%

    real photos have consistent penumbra, elongation, and orientation from one light
    source; composited photos mix shadows with different edge softness and axes
    '''
    # ---------------------------
    # penumbra hardness score
    # ---------------------------
    width_std = float(ml_features.get("edge_width_std", 0.0))
    width_cv  = float(ml_features.get("edge_width_cv",  0.0))

    # std score: 0 at std<=5px, 1 at std>=10px
    if width_std <= 5.0:
        score_std = 0.0
    elif width_std >= 10.0:
        score_std = 1.0
    else:
        score_std = (width_std - 5.0) / 5.0

    # CV score: 0 at cv<=0.8, 1 at cv>=1.5
    if width_cv <= 0.8:
        score_cv = 0.0
    elif width_cv >= 1.5:
        score_cv = 1.0
    else:
        score_cv = (width_cv - 0.8) / 0.7

    penumbra_score = 0.65 * score_std + 0.35 * score_cv

    # ---------------------------
    # elongation consistency score
    # ---------------------------
    elongation_cv = float(ml_features.get("elongation_cv", 0.0))
    elongation_samples = int(ml_features.get("elongation_samples", 0))

    if elongation_samples < 2:
        # not enough elongated shadows — fall back to penumbra only
        return max(0.0, min(1.0, penumbra_score))

    # CV (coefficient of variation) = std / mean of the elongation ratios across all shadows
    # it measures how spread out the ratios are relative to their average:
    #   CV <= 0.3 → ratios are tightly clustered → consistent elongation → score 0.0 (not suspicious)
    #   CV >= 0.6 → ratios are widely spread → inconsistent elongation → score 1.0 (suspicious)
    #   between 0.3 and 0.6 → linearly interpolated
    # variation is expected (lamp post vs fire hydrant cast different length shadows),
    # but extreme spread suggests shadows from different sun elevations were mixed together
    if elongation_cv <= 0.3:
        elongation_score = 0.0
    elif elongation_cv >= 0.6:
        elongation_score = 1.0
    else:
        elongation_score = (elongation_cv - 0.3) / 0.3

    # ---------------------------
    # orientation consistency score
    # ---------------------------
    orientation_std = float(ml_features.get("orientation_std_deg", 0.0))
    orientation_samples = int(ml_features.get("orientation_samples",   0))

    if orientation_samples < 2:
        tamper_score = 0.60 * penumbra_score + 0.40 * elongation_score
    else:
        # circular std of long-axis angles measures how much the shadow axes vary in direction:
        #   std <= 15° → axes point roughly the same way → same sun azimuth → score 0.0 (not suspicious)
        #   std >= 30° → axes point in clearly different directions → score 1.0 (suspicious)
        #   between 15° and 30° → linearly interpolated
        # a composited shadow from a different photo will have a noticeably different axis angle
        if orientation_std <= 15.0:
            orientation_score = 0.0
        elif orientation_std >= 30.0:
            orientation_score = 1.0
        else:
            orientation_score = (orientation_std - 15.0) / 15.0

        tamper_score = (0.50 * penumbra_score
                        + 0.25 * elongation_score
                        + 0.25 * orientation_score)

    return max(0.0, min(1.0, tamper_score))    

# ----------------------------------------------------------
# VISUALIZATION
# ----------------------------------------------------------
def visualize_depth_analysis(img, mask, debug_points, widths, contrasts,
                             shadow_contours=None, min_shadow_area=300, min_perimeter=30,
                             max_points_vis=2000):
    '''
    show what was measured so far:
    - shadow mask overlay
    - sample points (where penumbra was measured, shown as green dots)
    - elongation bounding boxes (cyan rectangles) and long-axis lines (magenta)
    - histogram of edge widths (distribution of penumbra hardness)
    - histogram of contrasts (distribution of shadow darkness)
    '''
    mask_u8 = mask.astype(np.uint8)

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

    # show where penumbra was sampled (green dots)
    pts_overlay = mask_overlay.copy()
    for i, (x, y) in enumerate(debug_points[:max_points_vis]):
        cv2.circle(pts_overlay, (x, y), 3, (0, 255, 0), -1)

    cv2.namedWindow("Penumbra Sample Points (green dots)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Penumbra Sample Points (green dots)", 800, 600)
    cv2.imshow("Penumbra Sample Points (green dots)", pts_overlay)

    # elongation visualization: bounding rectangles + long-axis lines
    #
    # how to interpret this window:
    # cyan box — the tightest rectangle fitted around each qualifying shadow;
    #   its proportions show the elongation ratio (long/short side)
    # magenta line — the long-axis direction of that rectangle; this is the angle
    #   used for orientation consistency
    #
    # what to look for:
    # REAL image   — magenta lines should point in roughly the same direction
    #   across all shadows (same sun, same axis for every shadow)
    # TAMPERED image — one or more magenta lines point in a noticeably different
    #   direction, indicating a shadow from a different light source
    #   was composited in; the cyan box will also look disproportionate
    #   if the shadow was stretched differently than the others
    if shadow_contours is not None:
        elong_overlay = img.copy()

        for contour in shadow_contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, closed=True)
            if area < min_shadow_area or perimeter < min_perimeter:
                continue

            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area < 1.0 or (area / hull_area) < 0.5:
                continue

            points = contour.reshape(-1, 2).astype(np.float32)
            if points.shape[0] < 5:
                continue

            rect = cv2.minAreaRect(contour)
            (cx, cy), (w, h), angle = rect

            major = max(w, h)
            minor = min(w, h)
            if minor < 1.0 or (major / minor) < 1.5:
                continue

            # draw the rotated bounding rectangle in cyan
            box = np.int32(cv2.boxPoints(rect))
            cv2.polylines(elong_overlay, [box], isClosed=True, color=(255, 255, 0), thickness=2)

            # compute the long-axis angle (same normalization as collect_orientation_angles)
            if h > w:
                angle = angle + 90.0
            angle = angle % 180.0
            theta = np.deg2rad(angle)

            # draw a magenta line through the center along the long axis
            half_len = int(major / 2)
            dx = int(np.cos(theta) * half_len)
            dy = int(np.sin(theta) * half_len)
            pt1 = (int(cx) - dx, int(cy) - dy)
            pt2 = (int(cx) + dx, int(cy) + dy)
            cv2.line(elong_overlay, pt1, pt2, color=(255, 0, 255), thickness=2)

        cv2.namedWindow("Elongation: boxes (cyan) + axes (magenta)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Elongation: boxes (cyan) + axes (magenta)", 800, 600)
        cv2.imshow("Elongation: boxes (cyan) + axes (magenta)", elong_overlay)

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
    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input))
    else:
        img = image_input

    assert img is not None, f"Cannot read image: {image_input}"

    # shadow mask
    mask = final_shadow_mask(img)
    assert mask.shape[:2] == img.shape[:2], "mask must align with original image"

    # clean the mask to reduce noise before finding contours:
    # open removes tiny specks, close connects nearby shadow pixels
    cleaned_mask = mask.astype(np.uint8).copy()
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, small_kernel)
    medium_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, medium_kernel)

    # convert to grayscale
    _, _, I0 = bgr_to_hsi_linear(img)
    gray_img = np.uint8(np.clip(I0 * 255, 0, 255))

    # find shadow boundaries from cleaned mask
    mask_u8 = mask.astype(np.uint8)
    shadow_contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # sample profiles and measure penumbra
    all_widths = []
    all_contrasts = []
    all_center_ys = []
    debug_points_all = []

    for contour in shadow_contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, closed=True)

        # skip small regions that are likely noise
        if area < min_shadow_area or perimeter < min_perimeter:
            continue

        widths, contrasts, center_ys, debug_points = sample_profiles_along_contour(
            contour,
            gray_img=gray_img,
            mask_bin=mask_u8,
            step=sample_step,
            half_len=10,
            num_samples=21
        )
        all_widths.extend(widths)
        all_contrasts.extend(contrasts)
        all_center_ys.extend(center_ys)
        debug_points_all.extend(debug_points)

    # collect elongation ratios and orientation angles from the same contours
    elongation_ratios = collect_elongation_ratios(
        shadow_contours,
        min_shadow_area=min_shadow_area,
        min_perimeter=min_perimeter
    )
    orientation_angles = collect_orientation_angles(
        shadow_contours,
        min_shadow_area=min_shadow_area,
        min_perimeter=min_perimeter
    )

    features = extract_features(all_widths, all_contrasts, elongation_ratios, orientation_angles)

    # calculate tamper score
    tamper_score = None
    if compute_tamper_score and len(all_widths) > 0:
        tamper_score = calculate_depth_tamper_score(features)
        print(f"\n{'='*60}")
        print(f"DEPTH TAMPER SCORE: {tamper_score:.3f}")
        print(f"{'='*60}")

    # visualize
    if visualize and len(all_widths) > 0:
        visualize_depth_analysis(
            img,
            mask_u8,
            debug_points_all,
            all_widths,
            all_contrasts,
            shadow_contours=shadow_contours,
            min_shadow_area=min_shadow_area,
            min_perimeter=min_perimeter,
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
    test_path = "data/images/2-edited.jpg"

    img = cv2.imread(test_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {test_path}")
    
    mask = final_shadow_mask(img)     

    result = analyze_depth(
        test_path,
        visualize=True,
        sample_step=4,
        compute_tamper_score=True,
        min_shadow_area=300,
        min_perimeter=30
    )

    print("\nExtracted features:", result["features"])
