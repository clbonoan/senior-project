# save the trained models for frontend/backend inference
# train module models and the final stacked model, then save them as joblib files

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from feature_selection import select_texture_features

# choose classifier type
# module models use full lighting/depth features and a selected texture feature subset
# stacked model uses only 3 inputs: texture_prob, lighting_prob, depth_prob
MODULE_MODEL_TYPE = "rf"
STACK_MODEL_TYPE = "logreg"

# start from root directory for paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

texture_csv = DATA_DIR / "texture_features.csv"
lighting_csv = DATA_DIR / "lighting_features.csv"
depth_csv = DATA_DIR / "depth_features.csv"


def prepare_features(df, rule_col, error_col):
    '''
    build X and y from a module feature CSV,
    drop non-ML columns and keep only numeric values
    '''
    # remove image_id, label, rule-based score, error column, and count columns
    # count columns (num_measurements, elongation_samples, orientation_samples) are
    # bookkeeping — they reflect image complexity, not whether the image is tampered
    count_cols = ["num_measurements", "elongation_samples", "orientation_samples", "num_components_usable"]
    X = df.drop(columns=["image_id", "label", rule_col, error_col] + count_cols, errors="ignore").copy()
    y = df["label"].copy()

    # force numeric values only
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    return X, y


def make_model(model_type):
    '''
    build the sklearn pipeline used for training.
    '''
    base_steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]

    if model_type == "logreg":
        return Pipeline(base_steps + [
            ("scaler", RobustScaler()),
            ("model", LogisticRegression(solver="liblinear", max_iter=5000, random_state=42))
        ])

    elif model_type == "svm":
        return Pipeline(base_steps + [
            ("scaler", RobustScaler()),
            ("model", SVC(kernel="rbf", probability=True, random_state=42))
        ])

    elif model_type == "rf":
        return Pipeline(base_steps + [
            ("model", RandomForestClassifier(
                n_estimators=200,
                random_state=42
            ))
        ])

    raise ValueError(f"Unknown model type: {model_type}")


def positive_class_proba(model, X, positive_label=1):
    '''
    get probability of class 1 (tampered), even if class order changes
    '''
    classes = model.named_steps["model"].classes_
    pos_idx = np.where(classes == positive_label)[0][0]
    return model.predict_proba(X)[:, pos_idx]


def save_bundle(path, model, feature_cols, model_type):
    '''
    save model and metadata needed for inference
    '''
    bundle = {
        "model": model,
        "feature_cols": list(feature_cols),
        "model_type": model_type,
        "positive_label": 1,   # class 1 = tampered
    }
    joblib.dump(bundle, path)


def main():
    # load CSV files
    texture_df = pd.read_csv(texture_csv)
    lighting_df = pd.read_csv(lighting_csv)
    depth_df = pd.read_csv(depth_csv)

    # prepare features
    X_texture, y = prepare_features(texture_df, "rule_score_texture", "texture_error")
    # select one fixed texture feature subset using the full dataset
    X_texture_selected_np, selected_texture_names = select_texture_features(
        X_texture.to_numpy(),
        list(X_texture.columns),
        y.to_numpy(),
        k=20
    )
    # convert back to DataFrame so column names stay attached
    X_texture_selected = pd.DataFrame(
        X_texture_selected_np,
        columns=selected_texture_names,
        index=X_texture.index
    )
    X_lighting, y_lighting = prepare_features(lighting_df, "rule_score_lighting", "lighting_error")
    X_depth, y_depth = prepare_features(depth_df, "rule_score_depth", "depth_error")

    # check labels match across module CSVs
    if not y.equals(y_lighting) or not y.equals(y_depth):
        raise ValueError("Label columns do not match across texture, lighting, and depth CSV files.")

    # check image rows match across module CSVs
    if not texture_df["image_id"].equals(lighting_df["image_id"]) or not texture_df["image_id"].equals(depth_df["image_id"]):
        raise ValueError("image_id rows do not match across module CSV files.")

    print("\nFeature shapes:")
    print("Texture:", X_texture.shape)
    print("\nSelected texture features for deployment:")
    print(f"Texture selected shape: {X_texture_selected.shape}")
    # print(selected_texture_names)
    print("Lighting:", X_lighting.shape)
    print("Depth:", X_depth.shape)

    print("\nMissing values after cleanup:")
    print("Texture:", int(X_texture.isna().sum().sum()))
    print("Lighting:", int(X_lighting.isna().sum().sum()))
    print("Depth:", int(X_depth.isna().sum().sum()))

    # train full-data module models
    texture_model = make_model(MODULE_MODEL_TYPE)
    lighting_model = make_model(MODULE_MODEL_TYPE)
    depth_model = make_model(MODULE_MODEL_TYPE)

    print(f"\nTraining full-data module models using: {MODULE_MODEL_TYPE}")

    # texture_model.fit(X_texture, y)
    texture_model.fit(X_texture_selected, y)
    lighting_model.fit(X_lighting, y)
    depth_model.fit(X_depth, y)

    # class-1 probabilities from module models (class 1 = tampered)
    # texture_prob = positive_class_proba(texture_model, X_texture, positive_label=1)
    texture_prob = positive_class_proba(texture_model, X_texture_selected, positive_label=1)
    lighting_prob = positive_class_proba(lighting_model, X_lighting, positive_label=1)
    depth_prob = positive_class_proba(depth_model, X_depth, positive_label=1)

    # build stacked training input
    X_stack = pd.DataFrame({
        "texture_prob": texture_prob,
        "lighting_prob": lighting_prob,
        "depth_prob": depth_prob,
    })

    print("\nStacking feature means by label:")
    stack_df_debug = X_stack.copy()
    stack_df_debug["label"] = y.values
    print(stack_df_debug.groupby("label")[["texture_prob", "lighting_prob", "depth_prob"]].mean())

    # train final stacked model
    stack_model = make_model(STACK_MODEL_TYPE)

    print(f"\nTraining final stacked model using: {STACK_MODEL_TYPE}")
    stack_model.fit(X_stack, y)

    # save all bundles
    texture_path = MODELS_DIR / "texture_model.joblib"
    lighting_path = MODELS_DIR / "lighting_model.joblib"
    depth_path = MODELS_DIR / "depth_model.joblib"
    stack_path = MODELS_DIR / "stack_model.joblib"

    save_bundle(
        texture_path,
        texture_model,
        X_texture_selected.columns,
        MODULE_MODEL_TYPE,
    )

    save_bundle(
        lighting_path,
        lighting_model,
        X_lighting.columns,
        MODULE_MODEL_TYPE,
    )

    save_bundle(
        depth_path,
        depth_model,
        X_depth.columns,
        MODULE_MODEL_TYPE,
    )

    save_bundle(
        stack_path,
        stack_model,
        X_stack.columns,
        STACK_MODEL_TYPE,
    )

    print("\nSaved model files:")
    print(texture_path)
    print(lighting_path)
    print(depth_path)
    print(stack_path)

    # quick full-data debug predictions
    stack_prob = positive_class_proba(stack_model, X_stack, positive_label=1)
    stack_pred = (stack_prob >= 0.5).astype(int)

    debug = {
        "module_model_type": MODULE_MODEL_TYPE,
        "stack_model_type": STACK_MODEL_TYPE,
        # "texture_num_features": int(X_texture.shape[1]),
        "texture_raw_num_features": int(X_texture.shape[1]),
        "texture_selected_num_features": int(X_texture_selected.shape[1]),
        "selected_texture_features": list(selected_texture_names),
        "lighting_num_features": int(X_lighting.shape[1]),
        "depth_num_features": int(X_depth.shape[1]),
        "stack_num_features": int(X_stack.shape[1]),
        "num_samples": int(len(y)),
        "texture_prob_mean": float(np.mean(texture_prob)),
        "lighting_prob_mean": float(np.mean(lighting_prob)),
        "depth_prob_mean": float(np.mean(depth_prob)),
        "stack_prob_mean": float(np.mean(stack_prob)),
        "stack_pred_positive_rate": float(np.mean(stack_pred)),
    }

    debug_path = MODELS_DIR / "model_debug.json"
    with open(debug_path, "w", encoding="utf-8") as f:
        json.dump(debug, f, indent=2)

    print("\nSaved manifest/debug:")
    print(debug_path)


if __name__ == "__main__":
    main()