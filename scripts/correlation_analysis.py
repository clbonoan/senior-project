import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# load data
path = "data/features.csv"

df = pd.read_csv(path)

# keep numeric columns
df_num = df.select_dtypes(include=[np.number])

assert "label" in df_num.columns, "Label column not found"

# standardize features
X = df_num.drop(columns=["label"])
y = df_num["label"]

X_std = (X - X.mean()) / (X.std() + 1e-8)

df_std = X_std.copy()
df_std["label"] = y.values

# correlation matrix
corr = df_std.corr(method="pearson")

# heatmap
plt.figure(figsize=(14,12))
sns.heatmap(
    corr,
    cmap="coolwarm",
    center=0,
    square=True,
    cbar_kws={"shrink": 0.8}
)
plt.title("Feature Correlation Matric (standardized)")
plt.tight_layout()
plt.show()

# sorted feature-label correlation
label_corr = corr["label"].drop("label").sort_values(key=np.abs, ascending=False)

print("\nTop correlations with LABEL:\n")
print(label_corr.head(20))
