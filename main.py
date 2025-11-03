from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import shutil
import os
from fastapi.responses import JSONResponse
from texture import analyze_image

 

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

@app.post("/process/")
async def process_image(file: UploadFile = File(...)):
    contents = await file.read()

    np_img = np.frombuffer(contents, np.uint8)

    img = cv.imdecode(np_img, cv.IMREAD_COLOR)

    if img is None:
        return JSONResponse({"error": "Could not read uploaded image."}, status_Code = 400)
 
    result = analyze_image(img, visualize=False)
    
    print("Result returned from texture1:", result)

    score = 0.0
    if isinstance(result, dict) and "tamper_score" in result:
        score = result["tamper_score"]

    return JSONResponse({"tamper_score": score})
