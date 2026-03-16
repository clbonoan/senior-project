# training a classifier for each module (texture, lighting, depth)
# each module learns based on features, what is probability(image is tampered)


import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# choose classifier by changing to "logreg" (Logistic Regression),
# "rf" (Random Forest), or "svm" (Support Vector Machine):
MODEL_TYPE = "rf"

# go to project's root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# data directory where csv files are stored
DATA_DIR = PROJECT_ROOT / "data"

# paths to the generated feature csv files
texture_csv = DATA_DIR / "texture_features.csv"
lighting_csv = DATA_DIR / "lighting_features.csv"
depth_csv = DATA_DIR / "depth_features.csv"

# load feature files into dataframes
texture_df = pd.read_csv(texture_csv)
lighting_df = pd.read_csv(lighting_csv)
depth_df = pd.read_csv(depth_csv)

# PREPARE FEATURES FOR MACHINE LEARNING
# remove columns that shouldn't be used as ML inputs; separate the label column
def prepare_features(df, rule_col, error_col):
    # X = the feature matrix (what the model learns from)
    # drop image_id, label, rule_score_*, and *_error
    X = df.drop(columns=["image_id", "label", rule_col, error_col], errors="ignore").copy()
    
    # y = ground truth labels (0=real, 1=tampered)
    y = df["label"].copy()

    # force all remaining values to numeric values
    X = X.apply(pd.to_numeric, errors="coerce")

    # replace inf / -inf with NaN 
    X = X.replace([np.inf, -np.inf], np.nan)

    return X, y

# extract ML features from each module dataset
X_texture, y = prepare_features(texture_df, "rule_score_texture", "texture_error")
X_lighting, _ = prepare_features(lighting_df, "rule_score_lighting", "lighting_error")
X_depth, _ = prepare_features(depth_df, "rule_score_depth", "depth_error")

print("\nFeature shapes:")
print("Texture:", X_texture.shape)
print("Lighting:", X_lighting.shape)
print("Depth:", X_depth.shape)

print("\nMissing values after cleanup:")
print("Texture:", int(X_texture.isna().sum().sum()))
print("Lighting:", int(X_lighting.isna().sum().sum()))
print("Depth:", int(X_depth.isna().sum().sum()))

# TRAIN/TEST SPLIT
# split: 80% training, 20% testing
# keep real/fake ratio balanced in both sets with stratify=y
X_train_tex, X_test_tex, y_train, y_test = train_test_split(
    X_texture, y, test_size=0.2, random_state=42, stratify=y
)

# lighting and depth datasets use the same rows as texture split
X_train_light = X_lighting.loc[X_train_tex.index]
X_test_light = X_lighting.loc[X_test_tex.index]

X_train_depth = X_depth.loc[X_train_tex.index]
X_test_depth = X_depth.loc[X_test_tex.index]

# MODEL (CLASSIFIER) BUILDER BASED ON MODEL_TYPE
def make_model(model_type):
    # step 1: fill missing values with column median
    base_steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]

    if model_type == "logreg":
        # logistic regression benefits from scaling
        return Pipeline(base_steps + [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, random_state=42))
        ])

    elif model_type == "svm":
        # SVM also benefits from scaling
        return Pipeline(base_steps + [
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", probability=True, random_state=42))
        ])

    elif model_type == "rf":
        # random forest usually does not need scaling
        return Pipeline(base_steps + [
            ("model", RandomForestClassifier(
                n_estimators=200,
                random_state=42
            ))
        ])

    else:
        raise ValueError(f"Unknown MODEL_TYPE: {model_type}")

# CREATE MODELS
texture_model = make_model(MODEL_TYPE)
lighting_model = make_model(MODEL_TYPE)
depth_model = make_model(MODEL_TYPE)

print(f"\nTraining models using {MODEL_TYPE}")

# TRAIN MODELS
texture_model.fit(X_train_tex, y_train)
lighting_model.fit(X_train_light, y_train)
depth_model.fit(X_train_depth, y_train)

# EVALUATE MODULE PERFORMANCE
print("\nModule accuracies:")

# test each model on the test dataset
for name, model, X_test_module in [
    ("Texture", texture_model, X_test_tex),
    ("Lighting", lighting_model, X_test_light),
    ("Depth", depth_model, X_test_depth),
]:
    # predict class labels
    preds = model.predict(X_test_module)
    # calculate accuracy
    acc = accuracy_score(y_test, preds)
    print(f"{name}: {acc:.3f}")

# GENERATE PROBABILITIES FOR STACKING
# predict_proba for probability of each class
# [;, 1] = probability of the tampered class (class 1)
texture_prob = texture_model.predict_proba(X_texture)[:, 1]
lighting_prob = lighting_model.predict_proba(X_lighting)[:, 1]
depth_prob = depth_model.predict_proba(X_depth)[:, 1]

# BUILD STACKING DATASET
# stacking - combines the predictions of several individual models to create
# a new, more robust model (meta-model)
stack_df = pd.DataFrame({
    "image_id": texture_df["image_id"],
    "label": texture_df["label"],
    "texture_prob": texture_prob,
    "lighting_prob": lighting_prob,
    "depth_prob": depth_prob
})

stack_path = DATA_DIR / "stacking_features.csv"
stack_df.to_csv(stack_path, index=False)

print("\nSaved stacking features:")
print(stack_path)