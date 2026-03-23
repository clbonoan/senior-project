# save the trained models for frontend/backend inference
# train module models and the final stacked model, then save them as joblib files

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# choose classifier type for both module models and stacked model
# options: "logreg", "rf", "svm"
MODEL_TYPE = "rf"

# start from root of directory for paths
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
    drops non-ML columns and keeps only numeric values
    '''
    # remove image_id, label, rule-based score, error percentage
    X = df.drop(columns=["image_id", "label", rule_col, error_col], errors="ignore").copy()
    y = df["label"].copy()

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    return X, y


def make_model(model_type):
    # build the sklearn pipeline used for training
    base_steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]

    if model_type == "logreg":
        return Pipeline(base_steps + [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, random_state=42))
        ])

    elif model_type == "svm":
        return Pipeline(base_steps + [
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", probability=True, random_state=42))
        ])

    elif model_type == "rf":
        return Pipeline(base_steps + [
            ("model", RandomForestClassifier(
                n_estimators=200,
                random_state=42
            ))
        ])

    raise ValueError(f"Unknown MODEL_TYPE: {model_type}")


def save_bundle(path, model, feature_cols, probability_flip=False):
    # save model and metadata needed for inference
    bundle = {
        "model": model,
        "feature_cols": list(feature_cols),
        "probability_flip": bool(probability_flip),
        "model_type": MODEL_TYPE,
    }
    joblib.dump(bundle, path)


def main():
    # load CSV files
    texture_df = pd.read_csv(texture_csv)
    lighting_df = pd.read_csv(lighting_csv)
    depth_df = pd.read_csv(depth_csv)

    # prepare features
    X_texture, y = prepare_features(texture_df, "rule_score_texture", "texture_error")
    X_lighting, y_lighting = prepare_features(lighting_df, "rule_score_lighting", "lighting_error")
    X_depth, y_depth = prepare_features(depth_df, "rule_score_depth", "depth_error")

    # check labels for mismatching
    if not y.equals(y_lighting) or not y.equals(y_depth):
        raise ValueError("Label columns do not match across texture, lighting, and depth CSV files.")

    if not texture_df["image_id"].equals(lighting_df["image_id"]) or not texture_df["image_id"].equals(depth_df["image_id"]):
        raise ValueError("image_id rows do not match across module CSV files.")

    print("\nFeature shapes:")
    print("Texture:", X_texture.shape)
    print("Lighting:", X_lighting.shape)
    print("Depth:", X_depth.shape)

    print("\nMissing values after cleanup:")
    print("Texture:", int(X_texture.isna().sum().sum()))
    print("Lighting:", int(X_lighting.isna().sum().sum()))
    print("Depth:", int(X_depth.isna().sum().sum()))

    # train full-data module models
    texture_model = make_model(MODEL_TYPE)
    lighting_model = make_model(MODEL_TYPE)
    depth_model = make_model(MODEL_TYPE)

    print(f"\nTraining full-data module models using: {MODEL_TYPE}")

    texture_model.fit(X_texture, y)
    lighting_model.fit(X_lighting, y)
    depth_model.fit(X_depth, y)

    # raw class 1 probabilities from module models
    texture_prob = texture_model.predict_proba(X_texture)[:, 1]
    lighting_prob = lighting_model.predict_proba(X_lighting)[:, 1]
    depth_prob = depth_model.predict_proba(X_depth)[:, 1]

    # flip probabilities so higher values consistently mean "more likely tampered"
    # texture_prob = 1.0 - texture_prob_raw
    # lighting_prob = 1.0 - lighting_prob_raw
    # depth_prob = 1.0 - depth_prob_raw

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
    stack_model = make_model(MODEL_TYPE)

    print(f"\nTraining final stacked model using: {MODEL_TYPE}")
    stack_model.fit(X_stack, y)

    # save all bundles
    texture_path = MODELS_DIR / "texture_model.joblib"
    lighting_path = MODELS_DIR / "lighting_model.joblib"
    depth_path = MODELS_DIR / "depth_model.joblib"
    stack_path = MODELS_DIR / "stack_model.joblib"

    save_bundle(
        texture_path,
        texture_model,
        X_texture.columns,
        # do not flip module probabilities in the backend
        probability_flip=False,
    )

    save_bundle(
        lighting_path,
        lighting_model,
        X_lighting.columns,
        probability_flip=False,
    )

    save_bundle(
        depth_path,
        depth_model,
        X_depth.columns,
        probability_flip=False,
    )

    save_bundle(
        stack_path,
        stack_model,
        X_stack.columns,
        # stack model output should already represent class 1 = tampered
        probability_flip=False,
    )

    print("\nSaved model files:")
    print(texture_path)
    print(lighting_path)
    print(depth_path)
    print(stack_path)

    # optional debug file
    debug = {
        "model_type": MODEL_TYPE,
        "texture_num_features": int(X_texture.shape[1]),
        "lighting_num_features": int(X_lighting.shape[1]),
        "depth_num_features": int(X_depth.shape[1]),
        "stack_num_features": int(X_stack.shape[1]),
        "num_samples": int(len(y)),
    }

    print("\n--- DEBUG ---")
    print("Texture:", texture_prob)
    print("Lighting:", lighting_prob)
    print("Depth:", depth_prob)
    print("Final:", ml_probability)
    print("Prediction:", ml_prediction)

    debug_path = MODELS_DIR / "model_debug.json"
    pd.Series(debug).to_json(debug_path, indent=2)

    print("\nSaved manifest/debug:")
    print(debug_path)


if __name__ == "__main__":
    main()