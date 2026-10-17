import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# 1. LOAD DATASET
# ==========================================

data_path = "dataset/heart.csv"

df = pd.read_csv(data_path)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# ==========================================
# 2. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop("target", axis=1)
y = df["target"]

# Convert heart disease target into binary classification
# 0 = No Heart Disease
# 1,2,3,4 = Heart Disease

y = (y > 0).astype(int)

print("\nTarget distribution:")
print(y.value_counts())

# ==========================================
# 3. SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ==========================================
# 4. FEATURE SCALING
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 5. CREATE MODELS
# ==========================================

models = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "decision_tree": DecisionTreeClassifier(
        random_state=42
    ),
    "knn": KNeighborsClassifier(
        n_neighbors=5
    ),
    "svm": SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

# ==========================================
# 6. CREATE MODELS FOLDER
# ==========================================

os.makedirs("models", exist_ok=True)

# ==========================================
# 7. TRAIN MODELS
# ==========================================

results = {}

for name, model in models.items():

    print("\n" + "=" * 50)
    print("Training:", name)
    print("=" * 50)

    # Tree models can work without scaling.
    # Scaled data is used for Logistic Regression,
    # KNN and SVM.

    if name in ["decision_tree", "random_forest"]:
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

    else:
        model.fit(X_train_scaled, y_train)

        predictions = model.predict(
            X_test_scaled
        )

        accuracy = accuracy_score(
            y_test,
            predictions
        )

    results[name] = accuracy

    print("Accuracy:", round(accuracy * 100, 2), "%")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions
        )
    )

    # Save model
    model_path = f"models/{name}.pkl"

    joblib.dump(
        model,
        model_path
    )

    print("Model saved:", model_path)


# ==========================================
# 8. SAVE SCALER
# ==========================================

scaler_path = "models/scaler.pkl"

joblib.dump(
    scaler,
    scaler_path
)

print("\nScaler saved:", scaler_path)

# ==========================================
# 9. DISPLAY MODEL COMPARISON
# ==========================================

print("\n")
print("=" * 60)
print("MODEL ACCURACY COMPARISON")
print("=" * 60)

for name, accuracy in results.items():

    print(
        f"{name:25s} : {accuracy * 100:.2f}%"
    )

# ==========================================
# 10. FIND BEST MODEL
# ==========================================

best_model = max(
    results,
    key=results.get
)

best_accuracy = results[best_model]

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print("Model:", best_model)
print(
    "Accuracy:",
    round(best_accuracy * 100, 2),
    "%"
)

# Save best model name
with open(
    "models/best_model.txt",
    "w"
) as file:

    file.write(best_model)

print("\nBest model saved successfully!")

print("\nHeartGuardAI model training completed!")