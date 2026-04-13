# TRAIN FINAL STACKING (FUSION MODEL)
# use texture_prob, lighting_prob, depth_prob to predict label (0=real, 1=tampered)

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedGroupKFold, cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# choose final stacking classifier (logreg, rf, svm)
MODEL_TYPE = "logreg"

# go to project's root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
STACKING_CSV = DATA_DIR / "stacking_features_oof.csv"

# load stacking dataset
stack_df = pd.read_csv(STACKING_CSV)

print("\nColumns found:")
print(stack_df.columns.tolist())

def base_image_group(image_id):
    '''
    group related files so variants of the same base image stay in the same fold

    examples:
      15.jpg -> 15
      15-edited.jpg -> 15
      15.2-edited.jpg -> 15
    '''
    name = Path(str(image_id)).stem
    root = name.split("-")[0].split(".")[0]
    return root

# PREPARE X AND y
# X = the 3 learned probabilities from stage 1
X = stack_df[["texture_prob", "lighting_prob", "depth_prob"]].copy()
# y = true labels
y = stack_df["label"].copy()

groups = stack_df["image_id"].apply(base_image_group)

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

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(final_model, X, y, cv=cv, groups=groups, scoring="accuracy")

print(f"\nFusion model: {MODEL_TYPE}")
print("Fold accuracies:", scores)
print("Mean accuracy:", scores.mean())
print("Std:", scores.std())

# OUT-OF-FOLD PREDICTIONS
y_pred = cross_val_predict(final_model, X, y, cv=cv, groups=groups, method="predict")
# fusion model’s own probability for class 1, where true values are 0=real 1=tampered
y_prob = cross_val_predict(final_model, X, y, cv=cv, groups=groups, method="predict_proba")[:, 1]

# METRICS FROM OOF PREDICTIONS
acc = accuracy_score(y, y_pred)

print("\nOverall OOF Results")
print(f"Accuracy: {acc:.3f}")

# CONFUSION MATRIX
labels = [0, 1]
class_names = ["Real", "Tampered"]

cm = confusion_matrix(y, y_pred, labels=labels)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues")

plt.title(f"Fusion Model Confusion Matrix ({MODEL_TYPE})")
plt.show()

# CLASSIFICATION REPORT
print("\nClassification Report:")
print(classification_report(
    y,
    y_pred,
    target_names=class_names,
    digits=3
))

# separate probabilities by true class
real_df = stack_df[stack_df["label"] == 0]
tampered_df = stack_df[stack_df["label"] == 1]

# prepare data for boxplot
data = [
    real_df["texture_prob"],
    tampered_df["texture_prob"],
    real_df["lighting_prob"],
    tampered_df["lighting_prob"],
    real_df["depth_prob"],
    tampered_df["depth_prob"],
]

labels = [
    "Texture\nReal",
    "Texture\nTampered",
    "Lighting\nReal",
    "Lighting\nTampered",
    "Depth\nReal",
    "Depth\nTampered",
]

plt.figure(figsize=(10, 6))
plt.boxplot(data, tick_labels=labels)
plt.ylabel("Predicted Probability of Tampered")
plt.title("Module Probability Distributions by True Class")
plt.xticks(rotation=0)
plt.tight_layout()

plot_path = DATA_DIR / "module_probability_boxplot.png"
plt.savefig(plot_path, dpi=300)
plt.show()

print("\nSaved plot:")
print(plot_path)

# SAVE OOF PREDICTIONS
results_df = X.copy()
results_df["true_label"] = y.values
results_df["pred_label"] = y_pred
results_df["pred_prob"] = y_prob

results_out = DATA_DIR / f"stacking_results_{MODEL_TYPE}.csv"
results_df.to_csv(results_out, index=False)

print("\nSaved OOF predictions:")
print(results_out)