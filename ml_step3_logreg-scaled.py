# cross validation for logistic regression (scaled)
# normalizes features before fitting
# NOTE: more stable and better for scientific evaluation (however, still constrained by small dataset of 54 training images)

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd
import joblib

df = pd.read_csv("data/features.csv")

non_feature_cols = ["filename", "label"]
feature_cols = [c for c in df.columns if c not in non_feature_cols]

print("Detected feature columns:", feature_cols)

X = df[feature_cols]
y = df["label"]

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=5000)
)

# cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
print("CV Accuracy:", scores)
print("Mean accuracy:", scores.mean())
print("Std deviation:", scores.std())

# fit on all data and save
model.fit(X, y)
joblib.dump({"model": model, "feature_cols": feature_cols}, "data/logreg-scaled.joblib")
print("Saved model to data/logreg-scaled.joblib")
