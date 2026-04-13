import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

def select_texture_features(X, feature_names, y, k=20):
    '''
    simple feature selection pipeline:
    1. impute missing values
    2. remove near-constant features
    3. remove highly correlated features
    4. keep top-k important features using random forest
    '''

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)

    # ---------------------------------
    # STEP 0: fill missing values first
    # ---------------------------------
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)

    # ---------------------------------
    # STEP 1: variance filter
    # ---------------------------------
    var_selector = VarianceThreshold(threshold=1e-4)
    X_var = var_selector.fit_transform(X)

    names_var = [
        name for name, keep in zip(feature_names, var_selector.get_support())
        if keep
    ]

    if X_var.shape[1] == 0:
        return X, list(feature_names)

    # ---------------------------------
    # STEP 2: correlation filter
    # ---------------------------------
    corr = np.corrcoef(X_var, rowvar=False)

    keep_indices = []
    removed = set()

    for i in range(len(names_var)):
        if i in removed:
            continue
        keep_indices.append(i)
        for j in range(i + 1, len(names_var)):
            if abs(corr[i, j]) > 0.9:
                removed.add(j)

    X_corr = X_var[:, keep_indices]
    names_corr = [names_var[i] for i in keep_indices]

    if X_corr.shape[1] == 0:
        return X_var, names_var

    # ---------------------------------
    # STEP 3: RF importance
    # ---------------------------------
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_corr, y)

    importances = rf.feature_importances_

    k = min(k, len(importances))
    top_idx = np.argsort(importances)[-k:]

    X_final = X_corr[:, top_idx]
    names_final = [names_corr[i] for i in top_idx]

    return X_final, names_final