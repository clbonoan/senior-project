import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# comparing the performance/quality since accuracy can be unstable is small datasets

CSV_PATH = "data/features.csv"
df = pd.read_csv(CSV_PATH)

def test_auc(prefix):
    features = [c for c in df.columns if c.startswith(prefix)]
    X = df[features]
    y = df["label"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for train_idx, test_idx in kf.split(X_scaled, y):
        X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        model = LogisticRegression(
            penalty="l2",
            solver="liblinear",
            max_iter=1000
        )
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_te)[:, 1]
        aucs.append(roc_auc_score(y_te, probs))

    print(f"\n{prefix.upper()} ONLY")
    print(f"Mean AUC: {np.mean(aucs):.4f}")
    print(f"Std Dev : {np.std(aucs):.4f}")

test_auc("texture_")
test_auc("lighting_")
test_auc("depth_")