# Third Feature - depth through correspoinding points of shadows and objects intersecting at a single point
# Checks if shadows are consistent with their matching objects and created by the same light source to verify authenticity

import cv2
import numpy as np
import matplotlib.pyplot as plt
import io, contextlib
from typing import Tuple, List, Optional

# use same shadow mask as texture.py
from shadow_mask import final_shadow_mask

def mask_from_texture(img_bgr, suppress_prints=True, **kwargs): 
    # call texture.py's make_shadow_mask function to get the shadow mask
    # binary mask where 255=shadow region, 0=not shadow
    if suppress_prints:
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink):
            out = final_shadow_mask(img_bgr, **kwargs)
    else:
        out = final_shadow_mask(img_bgr, **kwargs)

    if isinstance(out, tuple):
        mask = out[0]
    else:
        mask = out
    return mask

# ----------------------------------------------------------
# HELPER FUNCTIONS FOR VISUALS
# ----------------------------------------------------------
def overlay_red_mask(img_bgr, mask):
    mask_overlay = np.zeros_like(img_bgr)
    mask_overlay[:, :, 2] = mask    # gives red channel
    return cv2.addWeighted(img_bgr, 0.7, mask_overlay, 0.3, 0)

def shadow_bounds_from_mask(mask):
    # canny edge on the shadow mask to show the boundaries
    return cv2.Canny(mask, 50, 150)

def draw_points(img, pts, color, radius=2):
    out = img.copy()
    for (x, y) in pts:
        cv2.circle(out, (int(x), int(y)), radius, color, -1, lineType=cv2.LINE_AA)
    return out

def draw_pairs(img, pairs):
    out = img.copy()
    for (s, o) in pairs:
        cv2.circle(out, s, 2, (0, 0, 255), -1, lineType=cv2.LINE_AA)  # red shadow
        cv2.circle(out, o, 2, (0, 255, 0), -1, lineType=cv2.LINE_AA)  # green object
        cv2.line(out, s, o, (255, 200, 0), 1, lineType=cv2.LINE_AA)
    return out

def draw_lines_only(img, pairs):
    out = img.copy()
    for (s, o) in pairs:
        cv2.line(out, s, o, (0, 0, 0), 2, lineType=cv2.LINE_AA)
    return out

def draw_convergence(img, pairs, intersections, estimated_light):
    # original image size
    out = img.copy()
    # all pair lines
    for (s, o) in pairs:
        cv2.line(out, s, o, (200, 200, 200), 1, lineType=cv2.LINE_AA)
    # intersections area
    for (x, y) in intersections:
        cv2.circle(out, (int(round(x)), int(round(y))), 2, (0, 0, 255), -1, lineType=cv2.LINE_AA)
    # estimated light
    if estimated_light is not None:
        cv2.drawMarker(out, (int(estimated_light[0]), int(estimated_light[1])),
                      (255, 0, 255), cv2.MARKER_STAR, 20, 2)
    return out

def draw_convergence_measure(img, intersections, estimated_light, avg_distance):
    # original image size
    out = img.copy()
    
    cx, cy = int(estimated_light[0]), int(estimated_light[1])
    for (x, y) in intersections:
        cv2.circle(img, (int(round(x)), int(round(y))), 2, (0, 0, 255), -1, lineType=cv2.LINE_AA)
    # show mean radius as a circle to visualize “spread”
    cv2.circle(img, (cx, cy), int(avg_distance), (0, 0, 0), 2, lineType=cv2.LINE_AA)
    cv2.drawMarker(img, (cx, cy), (255, 0, 255), cv2.MARKER_CROSS, 20, 2)
    cv2.putText(img, f"avg spread - {avg_distance:.1f}px", (10, 30),
               cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 0, 0), 1)
    return img

# class to show visualizations
class DebugVisualizer:
    def __init__(self):
        self.images = []
    
    def add_image(self, name, img):
        self.images.append((name, img))
    
    def show_all(self):
        for name, img in self.images:
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            cv2.imshow(name, img)
            # cv2.waitKey(0)
            # cv2.destroyWindow(name)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def find_corners(image, max_corners=50):
    '''
    find corner points in the image (i.e., corners of boxes, tips of cones, etc.)
    these are geometrically significant points we want to use
    '''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Shi-Tomasi corner detection (good quality corners)
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_corners,
        qualityLevel=0.01,
        minDistance=10,
        blockSize=7
    )
    
    if corners is None:
        return []
    
    # convert to list of (x, y) tuples
    corner_points = []
    for corner in corners:
        x, y = corner.ravel()
        corner_points.append((int(x), int(y)))
    
    return corner_points

# ----------------------------------------------------------
# DETECT SHADOW-OBJECT PAIRS
# ----------------------------------------------------------
def find_shadow_object_pairs(img_bgr, shadow_mask):
    '''
    based on "Can people detect errors in shadows and reflections?" by Nightingale et al. and "Photo Forensics" by Hany Farid
    - premise: lines connecting shadow edge point and corresponding point of its object should meet at a single light source (line convergence constraint)
    - repeating this for many pairs -> if all lines intersect at one point, the image is likely to be real 
    return:
    - list of coords ((shadow_edge_x, shadow_edge_y), (object_edge_x, object_edge_y)) pairs
    '''
    # # find shadow edges
    # shadow_edges = cv2.Canny(shadow_mask, 50, 150)

    # # find edges in the original image to find the objects
    # gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # object_edges = cv2.Canny(gray, 50, 150)

    # # get size of image
    # h, w = shadow_mask.shape

    # # compute gradient of shadow mask to get direction of shadow edge
    # gx = cv2.Sobel(shadow_mask, cv2.CV_32F, 1, 0, ksize=3)
    # gy = cv2.Sobel(shadow_mask, cv2.CV_32F, 0, 1, ksize=3)

    # # compute inward normals (from shadow edge toward the object)
    # # sobel gradient points where pixel intensity changes fast -> from dark to light
    # # flipping the sign reverses the direction so from shadow edge back toward the object that casts it
    # eps = 1e-6  # prevent division by 0
    # nx = -gx / (np.sqrt(gx*gx + gy*gy) + eps)
    # ny= -gy / (np.sqrt(gx*gx + gy*gy) + eps)

    # # get all shadow edge pixel coords
    # shadow_y_coords, shadow_x_coords = np.where(shadow_edges > 0)

    # # subsample to avoid processing every pixel
    # step_size = max(1, int(0.002 * (h + w)))
    # shadow_points = list(zip(shadow_x_coords[::step_size], shadow_y_coords[::step_size]))

    # matched_pairs = []
    # max_search_distance = int(0.5 * max(h, w))

    # # find the corresponding object edge for each shadow edge point
    # for (shadow_x, shadow_y) in shadow_points:
    #     # get normal direction of this point
    #     dir_x = nx[shadow_y, shadow_x]
    #     dir_y = ny[shadow_y, shadow_x]

    #     # skip if normal direction is unreliable
    #     if dir_x**2 + dir_y**2 < 0.1:
    #         continue

    #     # normalize the direction vector
    #     magnitude = np.sqrt(dir_x**2 + dir_y**2) + eps
    #     dir_x = dir_x / magnitude
    #     dir_y = dir_y / magnitude

    #     # go along this direction to look for an object edge
    #     current_x = float(shadow_x)
    #     current_y = float(shadow_y)
    #     found_object = None

    #     for step in range(max_search_distance):
    #         current_x += dir_x
    #         current_y += dir_y

    #         # convert it to integer coords
    #         check_x = int(round(current_x))
    #         check_y = int(round(current_y))

    #         # check if you're outside the image
    #         if check_x < 0 or check_x >= w or check_y < 0 or check_y >= h:
    #             break

    #         # check if you hit an object edge
    #         if object_edges[check_y, check_x] > 0:
    #             found_object = (check_x, check_y)
    #             break
        
    #     # if match is found, save the pair
    #     if found_object is not None:
    #         shadow_point = (int(shadow_x), int(shadow_y))
    #         object_point = found_object
    #         matched_pairs.append((shadow_point, object_point))

    # print(f"Found {len(matched_pairs)} shadow-object pairs")
    # return matched_pairs
    all_corners = find_corners(img_bgr, max_corners=100)
    
    # separate corners into shadow corners and object corners
    shadow_corners = []
    object_corners = []
    
    for (x, y) in all_corners:
        # check if this corner is in the shadow
        if shadow_mask[y, x] > 0:
            shadow_corners.append((x, y))
        else:
            object_corners.append((x, y))
    
    print(f"Found {len(shadow_corners)} shadow corners and {len(object_corners)} object corners")
    
    if len(shadow_corners) == 0 or len(object_corners) == 0:
        print("Not enough corners found!")
        return []
    
    # calculate shadow gradient to find search direction
    gradient_y = cv2.Sobel(shadow_mask, cv2.CV_32F, 0, 1, ksize=3)
    gradient_x = cv2.Sobel(shadow_mask, cv2.CV_32F, 1, 0, ksize=3)
    
    # calculate normal direction (from shadow toward object)
    epsilon = 0.000001
    normal_x = -gradient_x / (np.sqrt(gradient_x**2 + gradient_y**2) + epsilon)
    normal_y = -gradient_y / (np.sqrt(gradient_x**2 + gradient_y**2) + epsilon)
    
    matched_pairs = []
    
    # for each shadow corner, find the best matching object corner
    for (shadow_x, shadow_y) in shadow_corners:
        # get the direction to search (perpendicular to shadow edge)
        dir_x = normal_x[shadow_y, shadow_x]
        dir_y = normal_y[shadow_y, shadow_x]
        
        # skip if direction is unreliable
        if dir_x**2 + dir_y**2 < 0.1:
            continue
        
        # normalize direction
        magnitude = np.sqrt(dir_x**2 + dir_y**2) + epsilon
        dir_x = dir_x / magnitude
        dir_y = dir_y / magnitude
        
        # find the closest object corner along this direction
        best_match = None
        best_score = float('inf')
        
        for (obj_x, obj_y) in object_corners:
            # vector from shadow corner to object corner
            vec_x = obj_x - shadow_x
            vec_y = obj_y - shadow_y
            distance = np.sqrt(vec_x**2 + vec_y**2)
            
            if distance < 10:  # too close, probably same feature
                continue
            
            # normalize the vector
            vec_x = vec_x / distance
            vec_y = vec_y / distance
            
            # check if this object corner is roughly in the right direction
            # (dot product with normal direction)
            alignment = vec_x * dir_x + vec_y * dir_y
            
            # we want alignment > 0.3 (roughly in the right direction)
            if alignment > 0.3:
                # score combines distance and alignment
                # prefer closer matches that are better aligned
                score = distance * (1.0 - alignment)
                
                if score < best_score:
                    best_score = score
                    best_match = (obj_x, obj_y)
        
        # if we found a good match, save it
        if best_match is not None:
            matched_pairs.append(((shadow_x, shadow_y), best_match))
    
    print(f"Matched {len(matched_pairs)} shadow-object corner pairs")
    
    # limit to about 10-15 best pairs (highest quality)
    if len(matched_pairs) > 15:
        # sort by line length (shorter is often better)
        matched_pairs.sort(key=lambda p: np.hypot(p[1][0]-p[0][0], p[1][1]-p[0][1]))
        matched_pairs = matched_pairs[:15]
        print(f"Using top 15 pairs for analysis")
    
    return matched_pairs

# ----------------------------------------------------------
# HELPER FUNCTIONS FOR LINES
# ----------------------------------------------------------
def calculate_line_equation(point1, point2):
    # get equation of line that passes through points
    x1, y1 = point1
    x2, y2 = point2

    # line equation from two points
    a = y2 - y1
    b = x1 - x2
    c = x2*y1 - x1*y2

    # normalize
    length = np.sqrt(a**2 + b**2)
    if length > 0:
        a = a/length
        b = b/length
        c = c/length

    return np.array([a, b, c])

def find_line_intersection(line1, line2):
    # find where two lines intersect with cross product
    intersection = np.cross(line1, line2)

    # if the z-component is too small, lines are parallel
    if abs(intersection[2]) < 0.0000000001:
        return None
    
    # convert from coords
    x = intersection[0] / intersection[2]
    y = intersection[1] / intersection[2]

    return (x, y)

def estimate_light_source(pairs, image_shape):
    '''
    find where light source is (the point where all lines intersect)
    -- useful for when location of light source is unknown in image
    '''
    if len(pairs) < 2:
        return None, 0.0, []
    
    # calculate line equations for all pairs
    lines = []
    for shadow_pt, object_pt in pairs:
        line = calculate_line_equation(shadow_pt, object_pt)
        lines.append(line)

    # find all pairwise intersections of these lines
    all_intersections = []
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            intersection = find_line_intersection(lines[i], lines[j])

            if intersection is not None:
                x, y = intersection
                h, w = image_shape[:2]

                # only keep intersections that are reasonably close to the image
                # light source could be outside the fram
                if x > -w and x < 2*w and y > -h and y < 2*h:
                    all_intersections.append(intersection)
    
    if len(all_intersections) == 0:
        return None, 0.0, []
    
    # estimated light source is the median of all intersections
    x_coords = [pt[0] for pt in all_intersections]
    y_coords = [pt[1] for pt in all_intersections]
    estimated_light = (np.median(x_coords), np.median(y_coords))

    # calculate how clustered the intersections are
    # tightly clustered = good (consistent lighting)
    # spread out = bad (possibly fake)
    distances = []
    for pt in all_intersections:
        dist = np.sqrt((pt[0] - estimated_light[0])**2 + (pt[1] - estimated_light[1])**2)
        distances.append(dist)

    avg_distance = np.mean(distances)

    # convert average distance to consistency score (0-1)
    # closer = higher score
    if avg_distance < 50:
        consistency_score = 1.0
    elif avg_distance < 150:
        # linear falloff
        consistency_score = 1.0 - (avg_distance - 50) / 100.0
    else:
        consistency_score = max(0.0, 1.0 - avg_distance / 300.0)
    
    print(f"Light source estimated at ({estimated_light[0]:.0f}, {estimated_light[1]:.0f})")
    print(f"Consistency score: {consistency_score:.2f}")
    
    return estimated_light, consistency_score, all_intersections

def ransac_find_best_light_source(pairs, iterations=500, threshold=5.0):
    '''
    ALTERNATIVE: could use RANSAC to find light source
    - helps with outliers by randomly sampling pairs and finding which light source has most "votes"
    '''
    print("Using RANSAC...")

    if len(pairs) < 3:
        return None, [], 0.0

    best_light_source = None
    best_inlier_indices = []
    best_inlier_ratio = 0.0
    
    # try many random samples
    for iteration in range(iterations):
        # randomly pick 2 pairs
        random_indices = np.random.choice(len(pairs), 2, replace=False)
        
        # calculate the lines for these pairs
        line1 = calculate_line_equation(pairs[random_indices[0]][0], 
                                        pairs[random_indices[0]][1])
        line2 = calculate_line_equation(pairs[random_indices[1]][0], 
                                        pairs[random_indices[1]][1])
        
        # find where they intersect (hypothesis for light source)
        light_hypothesis = find_line_intersection(line1, line2)
        
        if light_hypothesis is None:
            continue  # lines were parallel
        
        # count how many other pairs "agree" with this light source
        inliers = []
        for idx, (shadow_pt, object_pt) in enumerate(pairs):
            line = calculate_line_equation(shadow_pt, object_pt)
            a, b, c = line
            light_x, light_y = light_hypothesis
            
            # calculate distance from light source to this line
            distance = abs(a*light_x + b*light_y + c)
            
            # inlier if it's close enough
            if distance < threshold:
                inliers.append(idx)
        
        # check if this is better than previous finding
        inlier_ratio = len(inliers) / len(pairs)
        if inlier_ratio > best_inlier_ratio:
            best_inlier_ratio = inlier_ratio
            best_light_source = light_hypothesis
            best_inlier_indices = inliers
    
    print(f"RANSAC found {len(best_inlier_indices)} inliers ({best_inlier_ratio*100:.1f}%)")
    return best_light_source, best_inlier_indices, best_inlier_ratio

# ----------------------------------------------------------
# VISUALIZATION
# ----------------------------------------------------------
def draw_points(img, pts, color, radius=4):
    out = img.copy()
    for (x, y) in pts:
        cv2.circle(out, (int(x), int(y)), radius, color, -1, lineType=cv2.LINE_AA)
    return out

def draw_results(img, shadow_mask, pairs, light_source, consistency_score, inliers=None):
    output = img.copy()
    h, w = output.shape[:2]

    # overlay shadow regions in red
    mask_overlay = np.zeros_like(output)
    mask_overlay[:, :, 0] = shadow_mask  # blue channel
    output = cv2.addWeighted(img, 0.7, mask_overlay, 0.4, 0)

    # draw lines from shadow to object
    for idx, (shadow_pt, object_pt) in enumerate(pairs):
        if inliers is not None:
            # color code: green if it's consistent, red if not
            if idx in inliers:
                color = (0, 255, 0)  # green = good
            else:
                color = (0, 0, 255)  # red = outlier
        else:
            color = (0, 255, 0)
        
        cv2.line(output, shadow_pt, object_pt, color, 2)
    
    # mark the light source with a star
    if light_source is not None:
        light_x = int(light_source[0])
        light_y = int(light_source[1])
        cv2.drawMarker(output, (light_x, light_y), (0, 0, 255),
                     markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
    
    # add verdict text
    if consistency_score > 0.7:
        verdict = "AUTHENTIC"
        text_color = (0, 255, 0)
    elif consistency_score > 0.4:
        verdict = "SUSPICIOUS"
        text_color = (0, 165, 255)
    else:
        verdict = "LIKELY FAKE"
        text_color = (0, 0, 255)
        
    return output

def calculate_depth_tamper_score(consistency_score, shadow_coverage, num_pairs):
    '''
    calculate final tamper score based on shadow analysis
    - invert consistency score since high score = more tampered
    '''
    if num_pairs < 2 or shadow_coverage < 0.01:
        return 0.5
    
    # invert the consistency score
    # high consistency = low tamper score (possibly real)
    # low consistency = high tamper score (possibly fake)
    tamper_score = 1.0 - consistency_score

    return tamper_score

def analyze_depth(image_path, use_ransac=False, show_debug=False):    
    # load the image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(image_path)

    if show_debug:
        viz = DebugVisualizer()
    else:
        viz = None
    
    print(f"Analyzing: {image_path}")
    
    # original image
    if viz: 
        viz.add_image("original img", img)

    # detect shadows
    shadow_mask = final_shadow_mask(img)
    shadow_pixels = int(np.sum(shadow_mask > 0))
    total_pixels = img.shape[0] * img.shape[1]
    shadow_coverage = shadow_pixels / total_pixels
    
    # red mask
    red_overlay = overlay_red_mask(img, shadow_mask)
    if viz: 
        viz.add_image("red mask", red_overlay)

    if shadow_coverage < 0.01:
        print("\nNot enough shadows in this image to analyze!")
        if viz:
            viz.show_all()
        return
    
    # show shadow boundaries
    shadow_bounds = shadow_bounds_from_mask(shadow_mask)
    if viz: 
        viz.add_image("shadow bounds", shadow_bounds)

    # find shadow-object pairs
    pairs = find_shadow_object_pairs(img, shadow_mask)
    if len(pairs) < 2:
        print("\nCouldn't find enough shadow-object pairs to analyze!")
        return
    
    # array of paired points
    if viz:
        viz.add_image("array of paired points", draw_pairs(img, pairs))

    # lines connected shadow point to corresponding object point
    if viz:
        viz.add_image("lines connecting shadow point to corresponding object point",
                         draw_lines_only(img, pairs))

    # estimate light source
    if use_ransac:
        # use RANSAC for robustness
        light_source, inliers, score = ransac_find_best_light_source(pairs)
        # still calculate intersections for reference
        _, consistency_score, intersections = estimate_light_source(pairs, img.shape)
    else:
        # use simple method
        light_source, consistency_score, intersections = estimate_light_source(pairs, img.shape)
        inliers = None

    # lines converge to a point
    if viz:
        viz.add_image("lines converge to a point", draw_convergence(img, pairs, intersections, light_source))

    # measure convergence (average distance of intersections)
    dists = [np.hypot(x - light_source[0], y - light_source[1]) for (x, y) in intersections]
    if dists:
        avg_dist = float(np.mean(dists)) 
    else:
        return 0.0
    
    # measure convergence of lines
    if viz: viz.add_image("measure convergence of lines",
                          draw_convergence_measure(img, intersections, light_source, avg_dist))
   
    # final score
    score = consistency_score
    
    # visualize results
    #result_image = draw_results(img, shadow_mask, pairs, light_source, score, inliers)

    # calculate final tamper score
    tamper_score = calculate_depth_tamper_score(score, shadow_coverage, len(pairs))
    
    # tamper score
    print(f"tamper score = {tamper_score:.2f}")

    # show the results
    # cv2.imshow("Shadow Depth Result", result_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    if viz:
        viz.show_all()
    # print summary
    print("\n========================================")
    print("ANALYSIS COMPLETE")
    print("========================================")
    print(f"Shadow coverage: {shadow_coverage*100:.1f}%")
    print(f"Pairs analyzed: {len(pairs)}")
    if light_source:
        print(f"Light source: ({light_source[0]:.0f}, {light_source[1]:.0f})")
    print(f"Final score: {score:.2f}")
    
    if score > 0.7:
        print("Verdict: Image appears AUTHENTIC")
    elif score > 0.4:
        print("Verdict: Image is SUSPICIOUS - may be manipulated")
    else:
        print("Verdict: Image is LIKELY FAKE - inconsistent lighting")
    print("========================================\n")


# Run the analysis
if __name__ == "__main__":
    result = analyze_depth("data/images/5.jpg", show_debug=True)
