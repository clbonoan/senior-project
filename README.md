# Detecting Image Tampering Through Shadow Features in Urban Infrastructure
This is a full-stack web application that detects image manipulation by analyzing shadow inconsistencies in ground-level images using rule-base algorithms and machine learning.

**Project Timeline:** Fall 2025 - Spring 2026
**California State University, Bakersfield - Computer Science Senior Design**

***

## Project Overview
This system analyzes three shadow features (texture, lighting, and depth) to determine whether an image has been tampered with. Users can upload images through a web interface and receive tamper detection results from two analysis approaches built on the same engineered feature set:

- **Rule-Based Analysis:** applies manually defined thresholds to extract features for fully interpretable, deterministic classification
- **Machine Learning Pipeline:** uses the same features to train module models and a stacking ensemble to learn and detect patterns

***

## Features
- Interactive web interface for image upload and analysis output
- Real-time tamper detection and scoring
- Dual analysis modes:
    - Manual feature engineering and threshold scores (rule-based) with interpretable results
    - Learned patterns based on feature selection (ML) to predict outcomes
- Shadow detection and analysis across three shadow features: texture, lighting, and depth
- Comparative results showing probability scores and classifications

*** 

## Technologies Used
**Backend:**
- Python (3.13.7)
- Flask
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
- flask
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

> NOTE:
> Place your images in 'data/images/' (this directory is not tracked in Git and does not have images used for train/test)

***

## Usage Instructions

### Feature Extraction (Step 1)
Process all images and create feature datasets for each module (texture, lighting, and depth). Each file contains the following:
- extracted features
- labels (real vs tampered)
- optional rule-based scores (for comparison and rule-based path use)
```bash
python scripts/build_feature_csvs.py
```

### Train Module Models (Step 2)
Run:
```bash
python scripts/train_module_models.py
```

This trains a classifier for each module:
- texture model -> predicts texture_prob
- lighting model -> predicts lighting_prob
- depth model -> predicts depth_prob

The output file `data/stacking_features.csv` contains the probabilities and labels.

### Train Final Stacking Model (Step 3)
Run:
```bash
python scripts/train_stacking_model.py
```

This trains the final model using the three probabilities:
**[texture_prob, lighting_prob, depth_prob] -> final prediction**

Outputs:
- Final accuracy
- Confusion matrix
- Classification report
- `data/stacking_results_rf.csv` (predictions)

### Extra Usage Notes
**Model Selection**
You can change the classifier in both scripts by editing:
```bash
MODEL_TYPE = "rf"
```
Options:
- "rf" -> Random Forest (recommended)
- "logreg" -> Logistic Regression
- "svm" -> Support Vector Machine

**Labels**
Labels must be in data/labels.csv with:
```bash
filename,label
image1.jpg,0
image1-edited.jpg,1
```

### Running the Web Application
Start the Flask backend (do a standard run):
```bash
flask --app main run
```
OR run in debug mode (auto-reload):
```bash
flask --app main run --debug
```
Access the web interface, open your browser, and navigate to http://127.0.0.1:5000

***

## Project Structure
```
├── data/
│   ├── images/                     # Image dataset (not tracked)
│   ├── labels.csv                  # Ground truth labels
│   ├── features.csv                # Extracted features
|   ├── texture_features.csv        # Texture-based features
|   ├── lighting_features.csv       # Lighting-based features
|   ├── depth_features.csv          # Depth-based features
|   ├── stacking_features.csv       # Features used for stacking
|   ├── stacking_features_oof.csv   # Out-of-fold stacking features
|   ├── stacking_results_logreg.csv # Logistic Regression results
|   ├── stacking_results_rf.csv     # Random Forest results
|   └── stacking_results_svm.csv    # SVM results
├── models/
|   ├── texture_model.joblib        # Trained texture model
|   ├── lighting_model.joblib       # Trained lighting model
|   ├── depth_model.joblib          # Trained depth model
|   ├── stack_model.joblib          # Final stacking model
|   └── model_debug.json            # Debug/evaluation metadata
├── scripts/
|   ├── build_feature_csvs.py       # Build all feature CSVs for each module
│   ├── extract_csv.py              # Feature extraction script
|   ├── correlation_analysis.py     # Feature correlation analysis
|   ├── train_module_models.py      # Train individual module models
|   ├── train_stacking_model.py     # Train stacking/ensemble model
|   ├── save_models.py              # Save trained models to /models
|   └── checker.py                  # Debugging/validation script
├── static/                         # Front-end assets
│   ├── app.js                      # JavaScript application logic
│   ├── favicon.png                 # Website favicon
│   └── style.css                   # CSS stylesheet
├── templates/                      # Front-end design
│   ├── about.html                  # About page
│   ├── index.html                  # Index/Home page
│   ├── info.html                   # Information page
│   └── repo.html                   # Repository page
├── texture.py                      # Texture analysis module
├── lighting.py                     # Lighting analysis module
├── depth.py                        # Depth analysis module
├── main.py                         # Flask implementation/backend
├── shadow_mask.py                  # Shadow detection module
└── README.md
```

***

## Methodology

### Shadow Feature Analysis (to be updated)

1. **Texture:** analyzes shadow texture patterns and consistency
2. **Lighting:** evaluates light intensity ratios across shadows
3. **Depth:** assesses shadow parts (penumbra) and direction

### Detection Approaches

**Dual-Path Analysis: Rule-Based + Machine Learning Path:**
- Uses the same set of manually engineered features from texture, lighting, and depth analysis
- Implements two parallel approaches:

    **Rule-Based Path**
    - Applies threshold-based logic for classification
    - Fully interpretable and easy to debug
    - Serves as a transparent baseline

    **Machine Learning Path**
    - Trains module-specific models on the engineered features
    - Combines outputs using a stacking ensemble model
    - Learns patterns beyond fixed thresholds
- This dual approach enables:
    - Direct comparison between heuristic and learned methods
    - Designed to evaluate how far engineered heuristics can go compared to learned models on the same feature space
- Rule-based system applies threshold-based classification
- ML model learns weights for the same features
- Provides interpretable results with decision boundaries

## Dataset

- **Current size:** 169 ground-level images
- **Composition:** mix of authentic and tampered images
- **Creation**: tampered images created using Photoshop
- **Future expansion:** planned dataset growth in Spring 2026 semester

***

## Current Status (Fall 2025)

**Completed:**
- Shadow detection and analysis algorithms for all three features
- Rule-based threshold system
- Traditional ML model training and inference
- Full-stack web application with Flask backend
- User interface with two analysis modes
- Initial dataset of 169 images

**In Progress (Spring 2026):**
- Algorithm optimization and refactoring
- Dataset expansion
- Suspicious region visualization on output images
- Accuracy improvements for shadow detection
