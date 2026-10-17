import pandas as pd

input_file = "dataset/processed.cleveland.data"

columns = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target"
]

df = pd.read_csv(
    input_file,
    header=None,
    names=columns,
    na_values="?"
)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

# Remove rows containing missing values
df = df.dropna()

# Convert target into 0 or 1
# 0 = No heart disease
# 1 = Heart disease
df["target"] = (df["target"] > 0).astype(int)

# Save cleaned dataset
output_file = "dataset/heart.csv"
df.to_csv(output_file, index=False)

print("\nClean dataset created successfully!")
print("Saved as:", output_file)

print("Final dataset shape:", df.shape) 