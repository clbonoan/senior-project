# script to prepare X and y and train/test split images
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# load and read the csv file
CSV_PATH = "data/features.csv"
df = pd.read_csv(CSV_PATH)
print("Loaded dataframe with shape:", df.shape)

# identify feature columns
non_feature_cols = ["filename", "label"]

feature_cols = [c for c in df.columns if c not in non_feature_cols]

print("\nNumber of features:", len(feature_cols))

# X = feature matrix
X = df[feature_cols]

# y = label vector
y = df["label"]

print("X shape:", X.shape)
print("y shape:", y.shape)

# make sure features are numeric values
non_numeric_cols = X.select_dtypes(exclude=["float64", "int64"]).columns
print("\nNon-numeric feature columns:", list(non_numeric_cols))

if len(non_numeric_cols) > 0:
    raise ValueError("Non-numeric columns found. Clean before training data")

# print("label dtype:", df["label"].dtype)
# print("df label isna sum:", df["label"].isna().sum())
# 
#If you already created y earlier, print its details too:
# print("y type:", type(y))
# print("y dtype:", getattr(y, "dtype", None))
# print("y length:", len(y))
# 
#Check NaNs in y robustly (works for pandas Series or numpy array)
# y_arr = np.asarray(y, dtype=float)  # force float so we can test isnan
# print("NaNs in y (np.isnan):", np.isnan(y_arr).sum())
# 
#Show indices where y is NaN
# nan_idx = np.where(np.isnan(y_arr))[0]
# print("NaN indices:", nan_idx[:20])
# if len(nan_idx) > 0:
    #If y came from df without shuffling, this should line up:
    # print("Rows in df at NaN indices:\n", df.iloc[nan_idx][["filename", "label"]])

# create train/test split (68 images: 80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,      # 20% of 68 is about 13 or 14 images for testing
    random_state=42,    # produce the same split of data every time this script is ran
    stratify=y  # balance class
)

print("\nTrain/Test Split")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

print("\nTraining label distribution:")
print(y_train.value_counts())

print("\nTesting label distribution:")
print(y_test.value_counts())

print("\nSTEP 2 COMPLETE")