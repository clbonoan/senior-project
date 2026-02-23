import pandas as pd

df = pd.read_csv("data/features.csv")
print(df["label"].value_counts())