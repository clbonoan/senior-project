from flask import Flask, request, jsonify, render_template

import os
import base64
import cv2 as cv
import numpy as np
import joblib
import pandas as pd

from texture import analyze_texture
from lighting import analyze_lighting
from depth import analyze_depth

#app = FastAPI()
app = Flask(__name__)

# uploaded files go in uploads
# UPLOAD_DIR = os.path.join("static", "uploads")

# load ML model (module models and stacked model)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# os.makedirs(UPLOAD_DIR, exist_ok=True)

# load saved stacked pipeline models
texture_bundle = joblib.load(os.path.join(MODELS_DIR, "texture_model.joblib"))
lighting_bundle = joblib.load(os.path.join(MODELS_DIR, "lighting_model.joblib"))
depth_bundle = joblib.load(os.path.join(MODELS_DIR, "depth_model.joblib"))
stack_bundle = joblib.load(os.path.join(MODELS_DIR, "stack_model.joblib"))

texture_model = texture_bundle["model"]
lighting_model = lighting_bundle["model"]
depth_model = depth_bundle["model"]
stack_model = stack_bundle["model"]

texture_feature_cols = texture_bundle["feature_cols"]
lighting_feature_cols = lighting_bundle["feature_cols"]
depth_feature_cols = depth_bundle["feature_cols"]
stack_feature_cols = stack_bundle["feature_cols"]

# texture_probability_flip = texture_bundle.get("probability_flip", False)
# lighting_probability_flip = lighting_bundle.get("probability_flip", False)
# depth_probability_flip = depth_bundle.get("probability_flip", False)

print("Loaded texture features:", texture_feature_cols)
print("Loaded lighting features:", lighting_feature_cols)
print("Loaded depth features:", depth_feature_cols)
print("Loaded stacked features:", stack_feature_cols)

# routes to html pages
# when website is visited, it first calls index.html 
@app.route("/", methods=["GET"])
def upload_page():
    return render_template("index.html")

@app.route("/about", methods=["GET"])
def about():
    return render_template("about.html")

@app.route("/info", methods=["GET"])
def info():
    return render_template("info.html")

@app.route("/repo", methods=["GET"])
def repo():
    return render_template("repo.html")

# helpers
def _resize_for_web(img, max_w=900):
    h, w = img.shape[:2]
    if w > max_w:
        scale = max_w / w
        img = cv.resize(img, (max_w, int(h * scale)), interpolation=cv.INTER_AREA)
    return img

def _encode_b64(bgr):
    _, buf = cv.imencode(".png", bgr)
    return base64.b64encode(buf).decode()

def make_texture_vizzes(img, result):
    lbp = result.get("lbp")
    mask = result.get("mask")
    if lbp is None or mask is None:
        return None, None

    # viz 1: shadow regions overlaid on original (shows what was detected)
    overlay = img.copy()
    shadow_px = mask > 127
    overlay[shadow_px] = (
        overlay[shadow_px].astype(np.float32) * 0.5
        + np.array([0, 0, 200], dtype=np.float32) * 0.5
    ).astype(np.uint8)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(overlay, contours, -1, (0, 0, 255), 2)
    viz1 = _encode_b64(_resize_for_web(overlay))

    # viz 2: LBP texture map with shadow boundary (shows the texture pattern evidence)
    lbp_color = cv.applyColorMap(lbp, cv.COLORMAP_BONE)
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    outline = cv.morphologyEx(mask, cv.MORPH_GRADIENT, k)
    lbp_color[outline > 0] = (0, 0, 255)
    viz2 = _encode_b64(_resize_for_web(lbp_color))

    return viz1, viz2

def make_lighting_vizzes(img, result):
    mask = result.get("mask")
    shadow_log_ratios = result.get("shadow_log_ratios", {})
    shadow_labels = result.get("shadow_labels")
    if mask is None or shadow_labels is None:
        return None, None

    num_labels = int(shadow_labels.max()) + 1

    # viz 1: each shadow component a distinct color (shows how many shadows and where)
    overlay1 = img.copy()
    for i in range(1, num_labels):
        hue = int((i * 137) % 180)
        color_bgr = cv.cvtColor(
            np.array([[[hue, 200, 220]]], dtype=np.uint8), cv.COLOR_HSV2BGR
        )[0, 0].tolist()
        pixels = shadow_labels == i
        overlay1[pixels] = (
            overlay1[pixels].astype(np.float32) * 0.4
            + np.array(color_bgr, dtype=np.float32) * 0.6
        ).astype(np.uint8)
    viz1 = _encode_b64(_resize_for_web(overlay1))

    # viz 2: brightness ratio heatmap (same-colored = consistent light source = real)
    viz2 = None
    if shadow_log_ratios:
        ratios = list(shadow_log_ratios.values())
        r_min, r_max = min(ratios), max(ratios)
        r_range = r_max - r_min if r_max != r_min else 1.0
        overlay2 = img.copy()
        for label_idx, ratio in shadow_log_ratios.items():
            norm = int(((ratio - r_min) / r_range) * 255)
            color_bgr = cv.applyColorMap(
                np.array([[[norm]]], dtype=np.uint8), cv.COLORMAP_VIRIDIS
            )[0, 0].tolist()
            pixels = shadow_labels == label_idx
            overlay2[pixels] = (
                overlay2[pixels].astype(np.float32) * 0.3
                + np.array(color_bgr, dtype=np.float32) * 0.7
            ).astype(np.uint8)
        viz2 = _encode_b64(_resize_for_web(overlay2))

    return viz1, viz2

def make_depth_vizzes(img, result):
    mask = result.get("mask")
    gray = result.get("gray_img")
    debug_points = result.get("debug_points", [])
    shadow_contours = result.get("shadow_contours", [])
    if mask is None:
        return None, None

    MIN_AREA = 300
    MIN_PERIMETER = 30
    MIN_ELONGATION = 1.5

    # viz 1: penumbra sampling points on shadow overlay
    # green dots = locations along shadow boundaries where a perpendicular brightness
    # profile was sampled to measure the penumbra width (gradual vs. sharp transition)
    base = cv.cvtColor(gray, cv.COLOR_GRAY2BGR) if gray is not None else img.copy()
    base[mask > 0] = (
        base[mask > 0].astype(np.float32) * 0.6
        + np.array([0, 0, 180], dtype=np.float32) * 0.4
    ).astype(np.uint8)
    for (x, y) in debug_points[:2000]:
        cv.circle(base, (x, y), 3, (0, 255, 0), -1)
    viz1 = _encode_b64(_resize_for_web(base))

    # viz 2: elongation bounding boxes on original image
    # cyan box = tightest rectangle fitted around each shadow (box proportions = elongation ratio)
    # magenta line = long axis of that box (consistent direction across shadows = real)
    overlay2 = img.copy()
    for cnt in shadow_contours:
        area = cv.contourArea(cnt)
        perimeter = cv.arcLength(cnt, closed=True)
        if area < MIN_AREA or perimeter < MIN_PERIMETER:
            continue
        hull = cv.convexHull(cnt)
        if cv.contourArea(hull) < 1.0 or (area / cv.contourArea(hull)) < 0.5:
            continue
        if cnt.reshape(-1, 2).shape[0] < 5:
            continue
        rect = cv.minAreaRect(cnt)
        (cx, cy), (w_r, h_r), angle = rect
        major, minor = max(w_r, h_r), min(w_r, h_r)
        if minor < 1.0 or (major / minor) < MIN_ELONGATION:
            continue
        box = np.int32(cv.boxPoints(rect))
        cv.polylines(overlay2, [box], isClosed=True, color=(255, 255, 0), thickness=2)
        angle_norm = (angle + 90.0) % 180.0 if h_r > w_r else angle % 180.0
        theta = np.deg2rad(angle_norm)
        half_len = int(major / 2)
        dx, dy = int(np.cos(theta) * half_len), int(np.sin(theta) * half_len)
        cv.line(overlay2, (int(cx) - dx, int(cy) - dy), (int(cx) + dx, int(cy) + dy), (255, 0, 255), 2)
    viz2 = _encode_b64(_resize_for_web(overlay2))

    return viz1, viz2

def extract_tamper_score(result) -> float | None:
    if isinstance(result, dict) and "tamper_score" in result:
        try:
            return float(result["tamper_score"])
        except (TypeError, ValueError):
            return None
    return None

def build_feature_df(features_dict, feature_cols):
    # build a single row DataFrame in the same column order from training;
    # any missing or invalid values are replaced with 0.0
    row = []
    for col in feature_cols:
        value = features_dict.get(col, 0.0)
        try:
            row.append(float(value))
        except (TypeError, ValueError):
            row.append(0.0)

    return pd.DataFrame([row], columns=feature_cols)

def get_class1_probability(model, feature_df):
    # return probability for class 1 (tampered). assume labels are [0,1]
    proba = model.predict_proba(feature_df)[0]
    classes = model.named_steps["model"].classes_

    if classes is not None:
        classes_list = list(classes)
        if 1 in classes_list:
            idx = classes_list.index(1)
            return float(proba[idx])
        
    return float(proba[np.argmax(proba)])

# API
@app.route("/process/", methods=["POST"])
def process_image():
    # validate upload
    if "file" not in request.files:
        return jsonify(error="No file uploaded"), 400
    
    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify(error="Empty filename / no file selected."), 400
    
    # model selector (ml or dl)
    model = request.form.get("model", "ml")

    # read bytes -> opencv image
    contents = file.read()
    np_img = np.frombuffer(contents, np.uint8)
    img = cv.imdecode(np_img, cv.IMREAD_COLOR)

    if img is None:
        return jsonify(error="Could not read uploaded image."), 400
    
    '''
    run the rule-based analysis; scores are based on handcrafted tamper score
    calculations
    '''
    scores = {}

    # initialize so viz functions receive empty dicts on failure
    texture_result = {}
    lighting_result = {}
    depth_result = {}
    texture_features = {}
    lighting_features = {}
    depth_features = {}

    # texture
    try:
        texture_result = analyze_texture(img, visualize=False, compute_tamper_score=True)
        scores["texture"] = extract_tamper_score(texture_result)
        texture_features = texture_result.get("features", {})
    except Exception as e:
        print("error in texture analysis:", e)
        scores["texture"] = None

    # lighting
    try:
        lighting_result = analyze_lighting(img, show_debug=False, compute_tamper_score=True)
        scores["lighting"] = extract_tamper_score(lighting_result)
        lighting_features = lighting_result.get("features", {})
    except Exception as e:
        print("error in lighting analysis:", e)
        scores["lighting"] = None

    # depth
    try:
        depth_result = analyze_depth(
            img,
            visualize=False,
            compute_tamper_score=True,
            sample_step=4,
            min_shadow_area=300,
            min_perimeter=30,
        )
        scores["depth"] = extract_tamper_score(depth_result)
        depth_features = depth_result.get("features", {})
    except Exception as e:
        print("error in depth analysis:", e)
        scores["depth"] = None

    # generate evidence visualizations (always built; frontend decides what to show)
    try:
        tex_viz1, tex_viz2 = make_texture_vizzes(img, texture_result)
    except Exception as e:
        print("error in texture viz:", e)
        tex_viz1, tex_viz2 = None, None

    try:
        light_viz1, light_viz2 = make_lighting_vizzes(img, lighting_result)
    except Exception as e:
        print("error in lighting viz:", e)
        light_viz1, light_viz2 = None, None

    try:
        depth_viz1, depth_viz2 = make_depth_vizzes(img, depth_result)
    except Exception as e:
        print("error in depth viz:", e)
        depth_viz1, depth_viz2 = None, None

    evidence_images = {
        "texture": {"shadow_overlay": tex_viz1, "lbp_map": tex_viz2},
        "lighting": {"component_overlay": light_viz1, "ratio_heatmap": light_viz2},
        "depth": {"contour_overlay": depth_viz1, "orientation_overlay": depth_viz2},
    }

    # rule-based threshold vote
    THRESHOLD_TAMPER = 0.65
    THRESHOLD_REAL = 0.45

    votes = {}
    num_ones = 0
    num_zeros = 0

    for feature_name, score in scores.items():
        if score is None:
            votes[feature_name] = None
            continue

        if score >= THRESHOLD_TAMPER:
            votes[feature_name] = 1     # tampered
            num_ones += 1
        elif score <= THRESHOLD_REAL:
            votes[feature_name] = 0     # real
            num_zeros += 1
        else:
            votes[feature_name] = None  # uncertainty zone -> take out from vote

    # majority/consensus vote
    if num_ones == 0 and num_zeros == 0:
        final_vote = None
        vote_reason = "all_uncertain"   # every module was in the uncertainty zone
    elif num_ones > num_zeros:
        final_vote = 1
        vote_reason = "majority"
    elif num_zeros > num_ones:
        final_vote = 0
        vote_reason = "majority"
    else:
        final_vote = None   # tied vote -> inconclusive, not defaulting to Real
        vote_reason = "tie"
    
    # optional: overall continuous score through mean of scores
    valid_scores = [s for s in scores.values() if isinstance(s, (int, float))]
    overall_rule_based = float(np.mean(valid_scores)) if valid_scores else None
 
    # stacked ML prediction only when model == "ml"
    ml_prediction = None
    ml_probability = None
    ml_module_probabilities = {
        "texture": None,
        "lighting": None,
        "depth": None,
    }

    '''
    build a texture, lighting, and depth row, get one probability from each, and
    feed those values (3 probabilities) into the stacked model
    '''
    if model == "ml":
        try:
            # build single row DataFrame for each module
            texture_df_ml = build_feature_df(texture_features, texture_feature_cols)
            lighting_df_ml = build_feature_df(lighting_features, lighting_feature_cols)
            depth_df_ml = build_feature_df(depth_features, depth_feature_cols)

            # module probabilities
            texture_prob = get_class1_probability(texture_model, texture_df_ml)
            lighting_prob = get_class1_probability(lighting_model, lighting_df_ml)
            depth_prob = get_class1_probability(depth_model, depth_df_ml)

            # # apply saved flipped setting so higher probability aligns with "more likely tampered"
            # if texture_probability_flip:
            #     texture_prob = 1.0 - texture_prob
            # if lighting_probability_flip:
            #     lighting_prob = 1.0 - lighting_prob
            # if depth_probability_flip:
            #     depth_prob = 1.0 - depth_prob
            
            ml_module_probabilities["texture"] = float(texture_prob)
            ml_module_probabilities["lighting"] = float(lighting_prob)
            ml_module_probabilities["depth"] = float(depth_prob)

            # stacked model input
            stack_input = pd.DataFrame([{
                "texture_prob": texture_prob,
                "lighting_prob": lighting_prob,
                "depth_prob": depth_prob,
            }], columns=stack_feature_cols)

            # final stacked prediction
            ml_prediction = int(stack_model.predict(stack_input)[0])
            ml_probability = get_class1_probability(stack_model, stack_input)

        except Exception as e:
            print("Error during stacked ML prediction:", e)
            ml_prediction = None
            ml_probability = None
            ml_module_probabilities = {
                "texture": None,
                "lighting": None,
                "depth": None,
            }

    return jsonify(
        # rule-based output
        threshold = THRESHOLD_TAMPER,
        rule_based_scores = scores,
        rule_based_votes = votes,
        overall_rule_based_score = overall_rule_based,
        final_rule_based_vote = final_vote,     # 0=real, 1=tampered, None=inconclusive
        final_vote_reason = vote_reason,        # "majority" | "tie" | "all_uncertain"

        # stacked ML output
        ml_prediction =  ml_prediction,             # 0=real, 1=tampered, None=error
        ml_probability_tampered =  ml_probability,  # probability of class 1
        ml_module_probabilities = ml_module_probabilities,

        # evidence visualizations (base64 PNG strings)
        evidence_images = evidence_images,
    ), 200

if __name__ == "__main__":
    app.run(debug=True)