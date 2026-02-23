import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# load data
# -----------------------------
CSV_PATH = "data/features.csv"

df = pd.read_csv(CSV_PATH)
df = df.select_dtypes(include=[np.number])

assert "label" in df.columns, "Label column not found"

X = df.drop(columns=["label"])
y = df["label"].values

feature_names = X.columns.tolist()

# -----------------------------
# standardize features
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# train logistic regression
# -----------------------------
model = LogisticRegression(
    penalty="l2",
    solver="liblinear",
    max_iter=1000
)

model.fit(X_scaled, y)

# -----------------------------
# extract weights and bias
# -----------------------------
weights = model.coef_[0]
bias = model.intercept_[0]

# combine into table
coef_df = pd.DataFrame({
    "feature": feature_names,
    "weight": weights,
    "abs_weight": np.abs(weights)
}).sort_values("abs_weight", ascending=False)

print("\n==============================")
print("LOGISTIC REGRESSION RESULTS")
print("==============================")
print(f"\nBias (intercept): {bias:.4f}")
print("\nTop 25 Features by |Weight|:\n")
print(coef_df.head(25))

# save full table
coef_df.to_csv("data/logistic_weights.csv", index=False)

# -----------------------------
# cross-validation accuracy
# -----------------------------
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accs = []

for train_idx, test_idx in kf.split(X_scaled, y):
    X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    model_cv = LogisticRegression(
        penalty="l2",
        solver="liblinear",
        max_iter=1000
    )
    model_cv.fit(X_tr, y_tr)
    preds = model_cv.predict(X_te)
    accs.append(accuracy_score(y_te, preds))

print("\n5-Fold Cross-Validation Accuracy:")
print(f"Mean: {np.mean(accs):.4f}")
print(f"Std : {np.std(accs):.4f}")