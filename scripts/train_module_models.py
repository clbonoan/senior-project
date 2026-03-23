# training a classifier for each module (texture, lighting, depth)
# each module learns based on features, what is probability(image is tampered)

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
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

# DEBUG
print("\nFeature shapes:")
print("Texture:", X_texture.shape)
print("Lighting:", X_lighting.shape)
print("Depth:", X_depth.shape)

print("\nMissing values after cleanup:")
print("Texture:", int(X_texture.isna().sum().sum()))
print("Lighting:", int(X_lighting.isna().sum().sum()))
print("Depth:", int(X_depth.isna().sum().sum()))

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

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

texture_oof = np.zeros(len(y))
lighting_oof = np.zeros(len(y))
depth_oof = np.zeros(len(y))

texture_fold_acc = []
lighting_fold_acc = []
depth_fold_acc = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_texture, y), start=1):
    X_train_tex, X_val_tex = X_texture.iloc[train_idx], X_texture.iloc[val_idx]
    X_train_light, X_val_light = X_lighting.iloc[train_idx], X_lighting.iloc[val_idx]
    X_train_depth, X_val_depth = X_depth.iloc[train_idx], X_depth.iloc[val_idx]

    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    texture_model = make_model(MODEL_TYPE)
    lighting_model = make_model(MODEL_TYPE)
    depth_model = make_model(MODEL_TYPE)

    texture_model.fit(X_train_tex, y_train)
    lighting_model.fit(X_train_light, y_train)
    depth_model.fit(X_train_depth, y_train)

    # raw sklearn probabilities for class 1 (class 1 = tampered)
    texture_prob_raw = texture_model.predict_proba(X_val_tex)[:, 1]
    lighting_prob_raw = lighting_model.predict_proba(X_val_light)[:, 1]
    depth_prob_raw = depth_model.predict_proba(X_val_depth)[:, 1]

    # flip values here so higher probability means more likely tampered
    texture_oof[val_idx] = 1 - texture_prob_raw
    lighting_oof[val_idx] = 1 - lighting_prob_raw
    depth_oof[val_idx] = 1 - depth_prob_raw

    # module predictions
    texture_pred = texture_model.predict(X_val_tex)
    lighting_pred = lighting_model.predict(X_val_light)
    depth_pred = depth_model.predict(X_val_depth)

    # module accuracies per fold
    texture_fold_acc.append(accuracy_score(y_val, texture_pred))
    lighting_fold_acc.append(accuracy_score(y_val, lighting_pred))
    depth_fold_acc.append(accuracy_score(y_val, depth_pred))

print(f"\n{MODEL_TYPE} OOF module accuracies")
print(f"Texture mean acc:  {np.mean(texture_fold_acc):.3f}")
print(f"Lighting mean acc: {np.mean(lighting_fold_acc):.3f}")
print(f"Depth mean acc:    {np.mean(depth_fold_acc):.3f}")

stack_df = pd.DataFrame({
    "image_id": texture_df["image_id"],
    "label": y,
    "texture_prob": texture_oof,
    "lighting_prob": lighting_oof,
    "depth_prob": depth_oof
})

# check labels (flipped)
print("\nClass mapping:", texture_model.named_steps["model"].classes_)
print(texture_df[["image_id", "label"]].head(10))
print(texture_df["label"].value_counts())

print("\nProbability means by label:")
print(stack_df.groupby("label")[["texture_prob", "lighting_prob", "depth_prob"]].mean())

print("\nProbability medians by label:")
print(stack_df.groupby("label")[["texture_prob", "lighting_prob", "depth_prob"]].median())

stack_path = DATA_DIR / "stacking_features_oof.csv"
stack_df.to_csv(stack_path, index=False)

print("\nSaved OOF stacking features:")
print(stack_path)