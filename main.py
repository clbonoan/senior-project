from flask import Flask, request, jsonify, render_template

import os
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
UPLOAD_DIR = os.path.join("static", "uploads")

# load ML model (module models and stacked model)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(UPLOAD_DIR, exist_ok=True)

#old ML pipeline
# ml_bundle = joblib.load(ML_MODEL_PATH)
# ml_model = ml_bundle["model"]
# ml_feature_cols = ml_bundle["feature_cols"]
# print("Loaded ML feature columns:", ml_feature_cols)

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

texture_probability_flip = texture_bundle.get("probability_flip", False)
lighting_probability_flip = lighting_bundle.get("probability_flip", False)
depth_probability_flip = depth_bundle.get("probability_flip", False)

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
    classes = getattr(model, "classes_", None)

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

    # make sure these exist even if analysis fails
    texture_features = {}
    lighting_features = {}
    depth_features = {}

    # texture
    try:
        texture_result = analyze_texture(img, visualize=False, compute_tamper_score=True)
        #print("Result from texture:", texture_result)      # debug
        scores["texture"] = extract_tamper_score(texture_result)
        texture_features = texture_result.get("features", {})
    except Exception as e:
        print("error in texture analysis:", e)
        scores["texture"] = None

    # lighting
    try:
        lighting_result = analyze_lighting(img, show_debug=False, compute_tamper_score=True)
        #print("Result from lighting:", lighting_result)       # debug
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
        #print("Result from depth:", depth_result)       # debug
        scores["depth"] = extract_tamper_score(depth_result)
        depth_features = depth_result.get("features", {})
    except Exception as e:
        print("error in depth analysis:", e)
        scores["depth"] = None

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
        final_vote = None   # no valid scores; all features are uncertain
    elif num_ones > num_zeros:
        final_vote = 1
    elif num_zeros > num_ones:
        final_vote = 0
    else:
        # require majority to call tampered
        final_vote = 0  # tie, so assume real
    
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

            # apply saved flipped setting so higher probability aligns with "more likely tampered"
            if texture_probability_flip:
                texture_prob = 1.0 - texture_prob
            if lighting_probability_flip:
                lighting_prob = 1.0 - lighting_prob
            if depth_probability_flip:
                depth_prob = 1.0 - depth_prob
            
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
        final_rule_based_vote = final_vote,     # 0=real, 1=tampered, None=unknown

        # stacked ML output
        ml_prediction =  ml_prediction,             # 0=real, 1=tampered, None=error
        ml_probability_tampered =  ml_probability,  # probability of class 1
        ml_module_probabilities = ml_module_probabilities,
        # DEBUGGING
        #"ml_features_used": combined_ml_features,
    ), 200

if __name__ == "__main__":
    app.run(debug=True)