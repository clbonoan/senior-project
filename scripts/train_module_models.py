# training a classifier for each module (texture, lighting, depth)
# each module learns based on features, what is probability(image is tampered)

import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from feature_selection import select_texture_features

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

# TESTING to see if grouped CV is the correct choice, so use one edited image per real image
# texture_csv = DATA_DIR / "texture_features_one_edit_per_base.csv"
# lighting_csv = DATA_DIR / "lighting_features_one_edit_per_base.csv"
# depth_csv = DATA_DIR / "depth_features_one_edit_per_base.csv"

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

def base_image_group(image_id):
    '''
    group related files so variants of the same base image stay in the same fold.
    examples:
      15.jpg -> 15
      15-edited.jpg -> 15
      15.2-edited.jpg -> 15
    '''
    name = Path(str(image_id)).stem
    root = name.split("-")[0].split(".")[0]
    return root

# extract ML features from each module dataset
X_texture, y = prepare_features(texture_df, "rule_score_texture", "texture_error")
X_lighting, _ = prepare_features(lighting_df, "rule_score_lighting", "lighting_error")
X_depth, _ = prepare_features(depth_df, "rule_score_depth", "depth_error")

texture_feature_names = list(X_texture.columns)

# groups for grouped cross-validation
groups = texture_df["image_id"].apply(base_image_group)

# DEBUG
print("\nFeature shapes:")
print("Texture:", X_texture.shape)
print("Lighting:", X_lighting.shape)
print("Depth:", X_depth.shape)

print("\nMissing values after cleanup:")
print("Texture:", int(X_texture.isna().sum().sum()))
print("Lighting:", int(X_lighting.isna().sum().sum()))
print("Depth:", int(X_depth.isna().sum().sum()))

print("\nDistinct groups:", groups.nunique())
print("Sample groups:")
print(pd.DataFrame({
    "image_id": texture_df["image_id"].head(15),
    "group": groups.head(15)
}))

# MODEL (CLASSIFIER) BUILDER BASED ON MODEL_TYPE
def make_model(model_type):
    # step 1: fill missing values with column median
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
        # random forest usually does not need scaling
        return Pipeline(base_steps + [
            ("model", RandomForestClassifier(
                n_estimators=200,
                random_state=42
            ))
        ])
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {model_type}")
    
def positive_class_proba(model, X, positive_label=1):
    classes = model.named_steps["model"].classes_
    pos_idx = np.where(classes == positive_label)[0][0]
    return model.predict_proba(X)[:, pos_idx]

# track how often each texture feature is selected across folds
texture_feature_counter = Counter()

skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

texture_oof = np.zeros(len(y))
lighting_oof = np.zeros(len(y))
depth_oof = np.zeros(len(y))

texture_fold_acc = []
lighting_fold_acc = []
depth_fold_acc = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_texture, y, groups=groups), start=1):
    X_train_tex, X_val_tex = X_texture.iloc[train_idx], X_texture.iloc[val_idx]
    X_train_light, X_val_light = X_lighting.iloc[train_idx], X_lighting.iloc[val_idx]
    X_train_depth, X_val_depth = X_depth.iloc[train_idx], X_depth.iloc[val_idx]

    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # ------------------------------------------
    # texture feature selection on TRAIN fold only
    # ------------------------------------------
    # using the selected 20 features per fold (adapting per fold) out of the 36 in texture.py
    X_train_tex_selected, selected_texture_names = select_texture_features(
        X_train_tex.to_numpy(),
        list(X_train_tex.columns),
        y_train.to_numpy(),
        k=20
    )

    # apply the SAME selected columns to validation fold
    X_val_tex_selected = X_val_tex[selected_texture_names].to_numpy()

    print(f"Fold {fold} selected texture features: {len(selected_texture_names)}")
    print(selected_texture_names)
    
    # count how often each texture feature is selected
    texture_feature_counter.update(selected_texture_names)
    
    train_groups = set(groups.iloc[train_idx])
    val_groups = set(groups.iloc[val_idx])
    overlap = train_groups.intersection(val_groups)
    print(f"\nFold {fold}: train groups = {len(train_groups)}, val groups = {len(val_groups)}, overlap = {len(overlap)}")

    texture_model = make_model(MODEL_TYPE)
    lighting_model = make_model(MODEL_TYPE)
    depth_model = make_model(MODEL_TYPE)

    # texture_model.fit(X_train_tex, y_train)
    texture_model.fit(X_train_tex_selected, y_train)
    lighting_model.fit(X_train_light, y_train)
    depth_model.fit(X_train_depth, y_train)

    # raw sklearn probabilities for class 1 (class 1 = tampered)
    # texture_prob_raw = positive_class_proba(texture_model, X_val_tex, positive_label=1)
    texture_prob_raw = positive_class_proba(texture_model, X_val_tex_selected, positive_label=1)
    lighting_prob_raw = positive_class_proba(lighting_model, X_val_light, positive_label=1)
    depth_prob_raw = positive_class_proba(depth_model, X_val_depth, positive_label=1)

    # store tampered probabilities
    texture_oof[val_idx] = texture_prob_raw
    lighting_oof[val_idx] = lighting_prob_raw
    depth_oof[val_idx] = depth_prob_raw

    # module predictions
    # texture_pred = texture_model.predict(X_val_tex)
    texture_pred = texture_model.predict(X_val_tex_selected)    
    lighting_pred = lighting_model.predict(X_val_light)
    depth_pred = depth_model.predict(X_val_depth)

    print(f"Fold {fold} texture prob mean by true label:")
    print(pd.Series(texture_prob_raw).groupby(y_val.reset_index(drop=True)).mean())

    print(f"Fold {fold} texture confusion counts:")
    print(pd.crosstab(
        y_val.reset_index(drop=True),
        pd.Series(texture_pred, name="pred"),
        rownames=["true"],
        colnames=["pred"],
        dropna=False
    ))

    # CHECKS
    print(f"Fold {fold} lighting confusion counts:")
    print(pd.crosstab(
        y_val.reset_index(drop=True),
        pd.Series(lighting_pred, name="pred"),
        rownames=["true"],
        colnames=["pred"],
        dropna=False
    ))

    print(f"Fold {fold} depth confusion counts:")
    print(pd.crosstab(
        y_val.reset_index(drop=True),
        pd.Series(depth_pred, name="pred"),
        rownames=["true"],
        colnames=["pred"],
        dropna=False
    ))

    # module accuracies per fold
    texture_fold_acc.append(accuracy_score(y_val, texture_pred))
    lighting_fold_acc.append(accuracy_score(y_val, lighting_pred))
    depth_fold_acc.append(accuracy_score(y_val, depth_pred))

    texture_acc = accuracy_score(y_val, texture_pred)
    texture_acc_inv = accuracy_score(y_val, 1 - texture_pred)

    print(f"Texture acc: {texture_acc:.3f}, inverted: {texture_acc_inv:.3f}")

print("\nTexture feature selection frequency across folds:")
for feat, count in texture_feature_counter.most_common():
    print(f"{feat}: {count}/5")

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

print("\nCheck:")
print(stack_df.groupby("label")[["texture_prob"]].mean())

stack_path = DATA_DIR / "stacking_features_oof.csv"
stack_df.to_csv(stack_path, index=False)

print("\nSaved OOF stacking features:")
print(stack_path)