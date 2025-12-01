from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import JSONResponse

import shutil
import os
import cv2 as cv
import numpy as np 
import joblib
import pandas as pd

from texture import analyze_texture
from lighting import analyze_lighting
from depth import analyze_depth

app = FastAPI()

#to load images and style.css 
app.mount("/static", StaticFiles(directory="static"), name="static")
#connects fastapi to .htmlfile
templates = Jinja2Templates(directory="templates")

#uploaded files go in uploads
UPLOAD_DIR = "static/uploads"
#processed files go in processed
OUTPUT_DIR = "static/processed"

# load ML model (scaled logistic regression pipeline)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ML_MODEL_PATH = os.path.join(BASE_DIR, "data", "logreg-scaled.joblib")
ml_bundle = joblib.load(ML_MODEL_PATH)
ml_model = ml_bundle["model"]
ml_feature_cols = ml_bundle["feature_cols"]
print("Loaded ML feature columns:", ml_feature_cols)

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

#when website is visited, it calls index.html 
@app.get("/", response_class=HTMLResponse)
async def upload_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})

@app.get("/info", response_class=HTMLResponse)
async def info(request: Request):
    return templates.TemplateResponse("info.html", {"request": request})

@app.get("/repo", response_class=HTMLResponse)
async def repo(request: Request):
    return templates.TemplateResponse("repo.html", {"request": request})

def extract_tamper_score(result) -> float | None:
    if isinstance(result, dict) and "tamper_score" in result:
        try:
            return float(result["tamper_score"])
        except (TypeError, ValueError):
            return None
    return None

@app.post("/process/")
async def process_image(
    file: UploadFile = File(...), 
    model: str = Form("ml")
):
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    img = cv.imdecode(np_img, cv.IMREAD_COLOR)

    if img is None:
        return JSONResponse({"error": "Could not read uploaded image."}, status_Code = 400)
    
    # run all the rule-based parts of the analyzers
    scores: dict[str, float | None] = {}

    # make sure these exist even if analysis fails
    texture_features: dict[str, float] = {}
    lighting_features: dict[str, float] = {}
    depth_features: dict[str, float] = {}

    # texture
    try:
        texture_result = analyze_texture(img, visualize=False, compute_tamper_score=True)
        print("Result from texture:", texture_result)
        scores["texture"] = extract_tamper_score(texture_result)
        texture_features = texture_result.get("features", {})
    except Exception as e:
        print("error in texture analysis:", e)
        scores["texture"] = None

    # lighting
    try:
        lighting_result = analyze_lighting(img, show_debug=False, compute_tamper_score=True)
        print("Result from lighting:", lighting_result)
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
        print("Result from depth:", depth_result)
        scores["depth"] = extract_tamper_score(depth_result)
        depth_features = depth_result.get("features", {})
    except Exception as e:
        print("error in depth analysis:", e)
        scores["depth"] = None

    # rule-based threshold vote
    THRESHOLD_TAMPER = 0.65
    THRESHOLD_REAL = 0.45

    votes: dict[str, int | None] = {}
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
 
    # ML Prediction only when model == "ml"
    ml_prediction: int | None = None
    ml_probability: float | None = None

    if model == "ml":
        # ML feature response
        combined_ml_features: dict[str, float] = {}

        def add_prefixed(prefix: str, feats: dict):
            for k, v in feats.items():
                key = f"{prefix}_{k}"
                try:
                    combined_ml_features[key] = float(v)
                except (TypeError, ValueError):
                    combined_ml_features[key] = 0.0
        
        add_prefixed("texture", texture_features)
        add_prefixed("lighting", lighting_features)
        add_prefixed("depth", depth_features)

        # build DataFrame in the same order of ml_feature_cols
        feature_row = [combined_ml_features.get(col, 0.0) for col in ml_feature_cols]
        feature_df = pd.DataFrame([feature_row], columns=ml_feature_cols)

        try:
            # predict class label
            ml_prediction = int(ml_model.predict(feature_df)[0])
            
            # predict probability if possible
            if hasattr(ml_model, "predict_proba"):
                proba = ml_model.predict_proba(feature_df)[0]
                classes = getattr(ml_model, "classes_", None)
                
                if classes is not None:
                    # find the index of class "1" (tampered) if it exists
                    classes_list = list(classes)
                    if 1 in classes_list:
                        idx = classes_list.index(1)
                        ml_probability = float(proba[idx])
                    else:
                        # if class 1 doesn't exist, just use max probability
                        ml_probability = float(proba[np.argmax(proba)])
                else:
                    # no classes_ attribute; just take max probability
                    ml_probability = float(proba[np.argmax(proba)])
            else:
                print("Model has no predict_proba; probability not available.")
        except Exception as e:
            print("Error during ML prediction:", e)


    return JSONResponse({
        "threshold": THRESHOLD_TAMPER,
        "rule_based_scores": scores,
        "rule_based_votes": votes,
        "overall_rule_based_score": overall_rule_based,
        "final_rule_based_vote": final_vote,     # 0=real, 1=tampered, None=unknown

        # ML output
        "ml_prediction": ml_prediction,             # 0=real, 1=tampered, None=error
        "ml_probability_tampered": ml_probability,  # probability of class 1
        "ml_feature_cols": ml_feature_cols,         # list of all feature names
        # DEBUGGING
        #"ml_features_used": combined_ml_features,
    })
