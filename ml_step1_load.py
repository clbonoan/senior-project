# script to load the features.csv file and inspect the data to clean
import pandas as pd

# load and read the csv file
CSV_PATH = "data/features.csv"
df = pd.read_csv(CSV_PATH)
print("Loaded dataframe with shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nDataFrame info:")
print(df.info())

# display the first few rows of the DataFrame to verify
print("\nSample rows:")
print(df.head())

# output for if there are missing values
print("\nMissing values per column (top 20):")
missing = df.isna().sum().sort_values(ascending=False)
print(missing.head(20))

# check labels
if "label" not in df.columns:
    raise ValueError("ERROR: 'label' column is missing in the csv file")

print("\nLabel distribution (counts):")
print(df["label"].value_counts())

print("\nLabel distribution (percentage):")
print(df["label"].value_counts(normalize=True)*100)

print("\nSTEP 1 COMPLETE")