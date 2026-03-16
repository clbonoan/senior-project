# TRAIN FINAL STACKING (FUSION MODEL)
# use texture_prob, lighting_prob, depth_prob to predict label (0=real, 1=tampered)

import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# choose final stacking classifier (logreg, rf, svm)
MODEL_TYPE = "svm"

# go to project's root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

STACKING_CSV = DATA_DIR / "stacking_features.csv"

# load stacking dataset
stack_df = pd.read_csv(STACKING_CSV)

print("\nColumns found:")
print(stack_df.columns.tolist())

# PREPARE X AND y
# X = the 3 learned probabilities from stage 1
X = stack_df[["texture_prob", "lighting_prob", "depth_prob"]].copy()

# y = true labels
y = stack_df["label"].copy()

# TRAIN/TEST SPLIT
# split stacking dataset into 80% train, 20% split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# BUILD MODEL
def make_model(model_type):
    # fill any missing values 
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
    
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {model_type}")

final_model = make_model(MODEL_TYPE)

print(f"\nTraining final stacking model using: {MODEL_TYPE}")

# TRAIN MODEL
final_model.fit(X_train, y_train)

# EVALUATE MODEL
y_pred = final_model.predict(X_test)
y_prob = final_model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)

print("\nFinal Stacking Model Results")
print(f"Accuracy: {acc:.3f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=3))

# OPTIONAL: SAVE PREDICTIONS FOR THE TEST SET
results_df = X_test.copy()
results_df["true_label"] = y_test.values
results_df["pred_label"] = y_pred
results_df["pred_prob"] = y_prob

results_out = DATA_DIR / f"stacking_results_{MODEL_TYPE}.csv"
results_df.to_csv(results_out, index=False)

print("\nSaved test predictions:")
print(results_out)