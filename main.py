from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import shutil
import os
from fastapi.responses import JSONResponse
import cv2 as cv
import numpy as np

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
async def process_image(file: UploadFile = File(...)):
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    img = cv.imdecode(np_img, cv.IMREAD_COLOR)

    if img is None:
        return JSONResponse({"error": "Could not read uploaded image."}, status_Code = 400)
    
    # run all the rule-based parts of the analyzers
    scores: dict[str, float | None] = {}

    # texture
    try:
        texture_result = analyze_texture(img, visualize=False, compute_tamper_score=True)
        print("Result from texture:", texture_result)
        scores["texture"] = extract_tamper_score(texture_result)
    except Exception as e:
        print("error in texture analysis:", e)
        scores["texture"] = None

    # lighting
    try:
        lighting_result = analyze_lighting(img, show_debug=False, compute_tamper_score=True)
        print("Result from lighting:", lighting_result)
        scores["lighting"] = extract_tamper_score(lighting_result)
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
    except Exception as e:
        print("error in depth analysis:", e)
        scores["depth"] = None

    # threshold vote
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
            votes[feature_name] = None  # uncertain zone -> take out from vote

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
 
    return JSONResponse({
        "threshold": THRESHOLD_TAMPER,
        "rule_based_scores": scores,
        "rule_based_votes": votes,
        "overall_rule_based_score": overall_rule_based,
        "final_rule_based_vote": final_vote     # 0=real, 1=tampered, None=unknown
    })
