import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

CSV_PATH = "data/features.csv"
df = pd.read_csv(CSV_PATH)

# -----------------------------
# manually select core features
# -----------------------------
selected_features = [

    # depth (hardness + direction)
    "depth_edge_width_median",
    "depth_edge_width_cv",
    "depth_dir_std_deg",
    "depth_num_dir_samples",

    # lighting (strength consistency)
    "lighting_sr_log_ratio_iqr",
    "lighting_sr_dist_median",
    "lighting_Yn_median",
    "lighting_usable_frac",

    # texture (boundary texture + entropy)
    "texture_chi2_mean",
    "texture_chi2_std",
    "texture_lbp_entropy_in",
    "texture_normal_mrl",
    "texture_contrast_shadow_vs_non"
]

df = df[selected_features + ["label"]]

X = df.drop(columns=["label"])
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

print("\nReduced Feature Set CV Accuracy:")
print(f"Mean: {np.mean(accs):.4f}")
print(f"Std : {np.std(accs):.4f}")