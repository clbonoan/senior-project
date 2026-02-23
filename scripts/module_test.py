import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

CSV_PATH = "data/features.csv"
df = pd.read_csv(CSV_PATH)

def test_feature_set(feature_prefix):
    features = [c for c in df.columns if c.startswith(feature_prefix)]
    X = df[features]
    y = df["label"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs = []

    for train_idx, test_idx in kf.split(X_scaled, y):
        X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        model = LogisticRegression(
            penalty="l2",
            solver="liblinear",
            max_iter=1000
        )
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        accs.append(accuracy_score(y_te, preds))

    print(f"\n{feature_prefix.upper()} ONLY")
    print(f"Features: {len(features)}")
    print(f"Mean Acc: {np.mean(accs):.4f}")
    print(f"Std Dev : {np.std(accs):.4f}")

test_feature_set("texture_")
test_feature_set("lighting_")
test_feature_set("depth_")