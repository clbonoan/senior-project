# unscaled logistic regression on single train/test split
# tries to learn directly from the feature magnitudes
# ISSUE: can be very unstable since it may depend on large-range features (may give higher accuracy by chance)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/features.csv")

non_feature_cols = ["filename", "label"]
feature_cols = [c for c in df.columns if c not in non_feature_cols]

X = df[feature_cols]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Logistic Regression Test Accuracy:", accuracy_score(y_test, pred))
