# Detecting Image Tampering Through Shadow Features in Urban Infrastructure
This is a full-stack web application that detects image manipulation by analyzing shadow 
inconsistencies in ground-level images using rule-base algorithms and machine learning.

**Project Timeline:** Fall 2025 - Spring 2026
**California State University, Bakersfield - Computer Science Senior Design**

## Project Overview
This system analyzes three shadow features (texture, lighting, and depth) to determine
whether an image has been tampered with. Users can upload images through a web 
interface and receive tamper detection results from multiple analysis approaches:

- **Rule-base analysis:** manual threshold-based detection with explainable features
- **Machine Learning model:** traditional ML using manually engineered features

## Features
- Interactive web interface for image upload and analysis output
- Real-time tamper detection and scoring
- Dual analysis modes:
    - Manual feature engineering (rule-based + ML) with interpretable results
    - Automated feature learning (DL) for enhanced accuracy
- Shadow detection and analysis across three shadow features: texture, lighting, and depth
- Comparative results showing probability scores and threshold-based classifications

## Technologies Used
**Backend:**
- Python (3.13.7)
- FastAPI
- OpenCV and scikit-image (image processing)
- scikit-learn (machine learning)
- NumPy and SciPy (numerical computations)
- pandas (data analysis)

**Frontend:**
- HTML/CSS/JavaScript

**Tools:**
- Photoshop (dataset creation)

## Prerequisites
- Python 3.12+
- pip
- Git
- **Virtual environment is used**

## Dependencies
- matplotlib (for visualization purposes during testing)
- opencv-python
- scikit-image
- pandas
- scikit-learn
- uvicorn
- fastapi
- numpy
- scipy


## Installation
1. Clone the repository:
```bash
git clone https://github.com/clbonoan/senior-project
cd senior-project
```

2. Start virtual environment:
**Windows**
```bash
C:\venvs\global\Scripts\Activate.ps1
```
**Mac**
```bash
source ~venvs/global/bin/activate
```
3. Install required dependencies:
```bash
pip install [dependencies]
```

4. Set up data directory:
```bash
cd data/images
```

> [!NOTE]
> Place your images in 'data/images/' (this directory is not tracked in Git and does not have images used for train/test)

## Usage

### Feature Extraction
Extract shadow feature measurements (from texture, lighting, depth modules) from images:
```bash
python scripts/extract_csv.py --images data/images --labels data/labels.csv --features texture,lighting,depth --out data/features.csv
```

**What this does:**
- Reads all iamges from 'data/images/'
- Uses 'data/labels.csv' for ground truth labels
- Extracts shadow features using detection algorithms
- Saves measurements to 'data/features.csv'

### Running the Web Application
Start the FastAPI backend and enable auto-reload as project files change:
```bash
uvicorn main:app --reload
```
Access the web interface, open your browser, and navigate to http://127.0.0.1:8000

## Project Structure
```
├── data/
│   ├── images/                 # Image dataset (not tracked)
│   ├── labels.csv              # Ground truth labels
│   └── features.csv            # Extracted features
├── scripts/
│   └── extract_csv.py          # Feature extraction script
├── static/                     # Front-end assets
│   ├── app.js                  # JavaScript application logic
│   ├── favicon.png             # Website favicon
│   └── style.css               # CSS stylesheet
├── templates/                  # Front-end design
│   ├── about.html              # About page
│   ├── index.html              # Index page
│   ├── info.html               # Information page
│   └── repo.html               # Repository page
├── depth.py                    # Depth analysis module
├── lighting.py                 # Lighting analysis module
├── main.py                     # FastAPI implementation
├── ml_step1_load.py            # Load/import feature.csv
├── ml_step2_prepare.py         # Preprocess and split data for train/test
├── ml_step3_logreg-scaled.py   # Logistic Regression ML model with feature scaling
├── ml_step3_logreg.py          # Logistic Regression ML model unscaled
├── ml_step3_train_rf.py        # Random Forest classifier model
├── shadow_mask.py              # Shadow detection module
└── texture.py                  # Texture analysis module
```

## Methodology

### Shadow Feature Analysis

1. **Texture:** analyzes shadow texture patterns and consistency
2. **Lighting:** evaluates light intensity ratios across shadows
3. **Depth:** assesses shadow parts (penumbra) and direction

### Detection Approaches

**Rule-Based + Machine Learning Path:**
- Uses manually engineering features from the three shadow analysis modules
- Rule-based system applies threshold-based classification
- ML model learns weights for the same features
- Provides interpretable results with decision boundaries

## Dataset

- **Current size:** 68 ground-level images
- **Composition:** mix of authentic and tampered images
- **Creation**: tampered images created using Photoshop
- **Future expansion:** planned dataset growth in Spring 2026 semester

## Current Status (Fall 2025)

**Completed:**
- Shadow detection and analysis algorithms for all three features
- Rule-based classification system
- Traditional ML model training and inference
- Full-stack web application with FastAPI backend
- User interface with two analysis modes
- Initial dataset of 68 images

**In Progress (Spring 2026):**
- Algorithm optimization and refactoring
- Dataset expansion
- Suspicious region visualization on output images
- Accuracy improvements for shadow detection
