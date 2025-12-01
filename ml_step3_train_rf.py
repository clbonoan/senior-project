# training a random forest model
# fit it on the training data and test it on 14 samples
# ISSUE: prone to overfitting on our dataset since the dataset is small; could also be an issue of too many features (there are 74 features rn)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score
import joblib

# load and read the feature csv file
CSV_PATH = "data/features.csv"
df = pd.read_csv(CSV_PATH)

non_feature_cols = ["filename", "label"]
feature_cols = [c for c in df.columns if c not in non_feature_cols]

X = df[feature_cols]
y = df["label"]

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,      # 20% of 68 is about 13 or 14 images for testing
    random_state=42,    # produce the same split of data every time this script is ran
    stratify=y      # balance class
)

# train random forest 
model = RandomForestClassifier(
    n_estimators=100,   # more trees means more stable
    max_depth=3,
    min_samples_leaf=4,
    class_weight="balanced",
    random_state=42,
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
print("CV Accuracy:", scores)
print("Mean accuracy:", scores.mean())
print("Std deviation:", scores.std())

model.fit(X_train, y_train)

print("\nModel has been trained")

# evaluate the model on the test set
y_pred = model.predict(X_test)

print("\nEvaluating on Test Set")
print("Accuracy:", accuracy_score(y_test, y_pred))

train_pred = model.predict(X_train)
train_acc = accuracy_score(y_train, train_pred)
print("Train accuracy:", train_acc)

# precision score for class 1 (tampered images)
precision = precision_score(y_test, y_pred, pos_label=1)
print("Precision (tampered class = 1):", precision)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# important features
importances = pd.Series(model.feature_importances_, index=feature_cols)
importances = importances.sort_values(ascending=False)

print("\nTop 15 most important features:")
print(importances.head(15))

# save important features for inspection
importances.to_csv("data/feature_importances.csv", index=True)

# save the model for later use
joblib.dump(model, "data/rf_model.joblib")
print("\nSaved model to data/rf_model.joblib")

print("\nSTEP 3 COMPLETE")