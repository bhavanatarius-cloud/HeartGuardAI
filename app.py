from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import numpy as np
import joblib
import os
import sqlite3
from datetime import datetime

app = Flask(__name__)
app.secret_key = "heartguardai_secret_key"


# =========================================================
# CONFIGURATION & CONSTANTS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER = os.path.join(BASE_DIR, "models")
BEST_MODEL_FILE = os.path.join(MODEL_FOLDER, "best_model.txt")
DB_PATH = os.path.join(BASE_DIR, "database", "heartguard.db")

# Model Accuracies from training
MODEL_ACCURACIES = {
    "Logistic Regression": 83.33,
    "Decision Tree": 70.00,
    "KNN": 88.33,
    "SVM": 85.00,
    "Random Forest": 85.00
}

# Global ML models container
trained_models = {}
scaler = None


# =========================================================
# DATABASE HELPERS (SQLITE)
# =========================================================

def get_db_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        tables = [r[0] for r in cursor.fetchall()]

        if not tables or "predictions" not in tables:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT,
                    age INTEGER,
                    sex INTEGER,
                    risk REAL,
                    prediction INTEGER,
                    model TEXT
                )
            """)
            conn.commit()
        conn.close()
    except Exception as e:
        print(f"Database initialization error: {e}")


def get_all_predictions():
    records = []
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        tables = [r[0] for r in cursor.fetchall()]

        target_table = "predictions" if "predictions" in tables else (tables[0] if tables else None)
        if target_table:
            cursor.execute(f"SELECT * FROM {target_table} ORDER BY rowid DESC")
            rows = cursor.fetchall()
            for idx, row in enumerate(rows, start=1):
                d = dict(row)
                pred_raw = d.get("prediction", 0)
                if isinstance(pred_raw, str):
                    pred_int = 1 if any(k in pred_raw.lower() for k in ["high", "detect", "yes", "1"]) else 0
                else:
                    pred_int = int(pred_raw) if pred_raw is not None else 0

                try:
                    risk_val = float(d.get("risk", d.get("probability", 0)))
                except (ValueError, TypeError):
                    risk_val = 0.0

                records.append({
                    "id": d.get("id", idx),
                    "age": d.get("age", "-"),
                    "sex": d.get("sex", 1),
                    "risk": round(risk_val, 2),
                    "prediction": pred_int,
                    "model": d.get("model", "KNN"),
                    "created_at": d.get("created_at", d.get("date", datetime.now().strftime("%d-%m-%Y %H:%M")))
                })
        conn.close()
    except Exception as e:
        print(f"Error reading predictions from database: {e}")

    # Fallback to session history if DB has no records yet
    if not records:
        session_hist = session.get("history", []) if session else []
        for idx, item in enumerate(reversed(session_hist), start=1):
            is_high = 1 if "Heart Disease Detected" in item.get("prediction", "") or item.get("risk") == "High Risk" else 0
            records.append({
                "id": idx,
                "age": item.get("age", "-"),
                "sex": item.get("sex", 1),
                "risk": item.get("probability", 0),
                "prediction": is_high,
                "model": "KNN",
                "created_at": item.get("date", datetime.now().strftime("%d-%m-%Y %H:%M"))
            })

    return records


# =========================================================
# BEST MODEL INFO HELPER
# =========================================================

def get_best_model_info():
    best_name = "KNN"
    try:
        if os.path.exists(BEST_MODEL_FILE):
            with open(BEST_MODEL_FILE, "r") as f:
                content = f.read().strip()
                name_map = {
                    "knn": "KNN",
                    "logistic_regression": "Logistic Regression",
                    "decision_tree": "Decision Tree",
                    "svm": "SVM",
                    "random_forest": "Random Forest"
                }
                if content.lower() in name_map:
                    best_name = name_map[content.lower()]
                elif content:
                    best_name = content.upper()
    except Exception as e:
        print(f"Error reading best_model.txt: {e}")

    best_acc = MODEL_ACCURACIES.get(best_name, 88.33)
    return best_name, best_acc


# =========================================================
# LOAD ALL TRAINED MODELS
# =========================================================

def load_models():
    global trained_models
    global scaler

    model_files = {
        "logistic_regression": "logistic_regression.pkl",
        "decision_tree": "decision_tree.pkl",
        "knn": "knn.pkl",
        "svm": "svm.pkl",
        "random_forest": "random_forest.pkl"
    }

    for name, filename in model_files.items():
        path = os.path.join(MODEL_FOLDER, filename)
        try:
            trained_models[name] = joblib.load(path)
            print(f"{name} loaded successfully!")
        except Exception as e:
            print(f"Could not load {name}: {e}")

    # Load scaler
    scaler_path = os.path.join(MODEL_FOLDER, "scaler.pkl")
    try:
        scaler = joblib.load(scaler_path)
        print("Scaler loaded successfully!")
    except Exception as e:
        print(f"Could not load scaler: {e}")


# Initialize ML models and database on startup
load_models()
init_db()


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():
    return render_template("index.html")


# =========================================================
# PREDICTION
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get patient data from form
        age = float(request.form.get("age"))
        sex = float(request.form.get("sex"))
        cp = float(request.form.get("cp"))
        trestbps = float(request.form.get("trestbps"))
        chol = float(request.form.get("chol"))
        fbs = float(request.form.get("fbs"))
        restecg = float(request.form.get("restecg"))
        thalach = float(request.form.get("thalach"))
        exang = float(request.form.get("exang"))
        oldpeak = float(request.form.get("oldpeak"))
        slope = float(request.form.get("slope"))
        ca = float(request.form.get("ca"))
        thal = float(request.form.get("thal"))

        # Create DataFrame
        input_data = pd.DataFrame(
            [[
                age, sex, cp, trestbps, chol, fbs,
                restecg, thalach, exang, oldpeak, slope, ca, thal
            ]],
            columns=[
                "age", "sex", "cp", "trestbps", "chol", "fbs",
                "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
            ]
        )

        if not trained_models:
            return """
            <h2>No trained models found.</h2>
            <p>Please run train_models.py first.</p>
            """

        if scaler is None:
            return """
            <h2>Scaler not found.</h2>
            <p>Please run train_models.py first.</p>
            """

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Prediction from all 5 models
        predictions = {}
        for name, model in trained_models.items():
            if name in ["decision_tree", "random_forest"]:
                pred = model.predict(input_data)[0]
            else:
                pred = model.predict(input_scaled)[0]
            predictions[name] = int(pred)

        # Best Model = KNN
        best_model = trained_models.get("knn")
        if best_model is None:
            best_model = list(trained_models.values())[0]

        best_prediction = best_model.predict(input_scaled)[0]

        # Risk Percentage
        if hasattr(best_model, "predict_proba"):
            probability = best_model.predict_proba(input_scaled)[0][1]
            risk_percentage = round(probability * 100, 2)
        else:
            risk_percentage = 88.33 if best_prediction == 1 else 15.00

        # Risk Result
        if best_prediction == 1:
            risk_category = "High Risk"
            prediction_text = "Heart Disease Detected"
            recommendation = (
                "Please consult a qualified healthcare professional "
                "for further clinical evaluation and specialized cardiovascular tests."
            )
            factors = [
                f"Elevated risk probability: {risk_percentage}%",
                f"Patient Age: {int(age)} years",
                f"Resting Blood Pressure: {int(trestbps)} mm Hg",
                f"Serum Cholesterol: {int(chol)} mg/dl"
            ]
        else:
            risk_category = "Low Risk"
            prediction_text = "No Heart Disease Detected"
            recommendation = (
                "The model predicts a lower risk of heart disease. "
                "Continue maintaining a healthy lifestyle, balanced nutrition, and attend routine checkups."
            )
            factors = [
                f"Low risk probability: {risk_percentage}%",
                "Cardiovascular indicators within normal predictive ranges",
                "Optimal resting parameters observed"
            ]

        now_str = datetime.now().strftime("%d-%m-%Y %H:%M")

        # Save to SQLite database
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO predictions (created_at, age, sex, risk, prediction, model)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                now_str,
                int(age),
                int(sex),
                risk_percentage,
                int(best_prediction),
                "KNN"
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error saving prediction to database: {e}")

        # Save to session history
        history = session.get("history", [])
        history.append({
            "date": now_str,
            "age": int(age),
            "sex": int(sex),
            "prediction": prediction_text,
            "risk": risk_category,
            "probability": risk_percentage,
            "message": recommendation,
            "recommendation": recommendation,
            "factors": factors,
            "predictions": predictions
        })
        session["history"] = history

        return render_template(
            "result.html",
            prediction=prediction_text,
            risk=risk_percentage,
            message=recommendation,
            recommendation=recommendation,
            factors=factors,
            probability=risk_percentage,
            predictions=predictions
        )

    except Exception as e:
        return f"""
        <h2>Prediction Error</h2>
        <p>{str(e)}</p>
        <br>
        <a href="/">Go Back</a>
        """


# =========================================================
# RESULT (DIRECT ACCESS GET)
# =========================================================

@app.route("/result")
def result():
    history = session.get("history", [])
    if history:
        last = history[-1]
        return render_template(
            "result.html",
            prediction=last.get("prediction", "No Heart Disease Detected"),
            risk=last.get("probability", 15.0),
            message=last.get("message", "Regular medical checkups are recommended."),
            recommendation=last.get("recommendation", "Regular medical checkups are recommended."),
            factors=last.get("factors", ["Optimal resting parameters observed"]),
            probability=last.get("probability", 15.0),
            predictions=last.get("predictions", {})
        )
    return redirect(url_for("home"))


# =========================================================
# DASHBOARD
# =========================================================

@app.route("/dashboard")
def dashboard():
    best_model, best_accuracy = get_best_model_info()
    predictions = get_all_predictions()

    total_predictions = len(predictions)
    high_risk = sum(1 for p in predictions if p.get("prediction") == 1)
    low_risk = sum(1 for p in predictions if p.get("prediction") == 0)

    return render_template(
        "dashboard.html",
        models=MODEL_ACCURACIES,
        best_model=best_model,
        best_accuracy=best_accuracy,
        total_predictions=total_predictions,
        high_risk=high_risk,
        low_risk=low_risk
    )


# =========================================================
# HISTORY
# =========================================================

@app.route("/history")
def history():
    all_preds = get_all_predictions()
    records = [
        [
            p["id"],
            p["created_at"],
            p["age"],
            p["sex"],
            "Heart Disease Detected" if p["prediction"] == 1 else "No Heart Disease Detected",
            p["risk"]
        ]
        for p in all_preds
    ]

    return render_template(
        "history.html",
        records=records,
        history=session.get("history", [])
    )


# =========================================================
# LOGIN
# =========================================================

@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("admin"):
        return redirect(url_for("admin"))

    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if username == "admin" and password == "admin123":
            session["admin"] = True
            return redirect(url_for("admin"))

        return render_template(
            "login.html",
            error="Invalid username or password"
        )

    return render_template("login.html")


# =========================================================
# ADMIN
# =========================================================

@app.route("/admin")
def admin():
    if not session.get("admin"):
        return redirect(url_for("login"))

    best_model, best_accuracy = get_best_model_info()
    predictions = get_all_predictions()

    total_predictions = len(predictions)
    high_risk = sum(1 for p in predictions if p.get("prediction") == 1)
    low_risk = sum(1 for p in predictions if p.get("prediction") == 0)

    model_name_map = {
        "logistic_regression": "Logistic Regression",
        "decision_tree": "Decision Tree",
        "knn": "KNN",
        "svm": "SVM",
        "random_forest": "Random Forest"
    }
    loaded_status = {
        disp: (key in trained_models and trained_models[key] is not None)
        for key, disp in model_name_map.items()
    }

    return render_template(
        "admin.html",
        models=MODEL_ACCURACIES,
        loaded_status=loaded_status,
        best_model=best_model,
        best_accuracy=best_accuracy,
        total_predictions=total_predictions,
        high_risk=high_risk,
        low_risk=low_risk,
        predictions=predictions
    )


# =========================================================
# LOGOUT
# =========================================================

@app.route("/logout")
def logout():
    session.pop("admin", None)
    return redirect(url_for("home"))


# =========================================================
# START FLASK SERVER
# =========================================================

if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )