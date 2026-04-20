"""
NutriMind — Flask Backend
Runs on: http://localhost:5000

Routes:
  POST /signup          → Register new user
  POST /login           → Authenticate user
  POST /predict         → ML prediction + save form + save result
  GET  /result/<email>  → Fetch latest result for a user
  GET  /history/<email> → Fetch all past results for a user
  GET  /health          → API health check

Install requirements:
  pip install flask flask-cors mysql-connector-python bcrypt scikit-learn pandas numpy
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import mysql.connector
import bcrypt
import pickle
import json
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────────────────────
# CONFIG — Change these to match your MySQL setup
# ─────────────────────────────────────────────────────────────
DB_CONFIG = {
    "host":     os.environ.get("DB_HOST", "localhost"),
    "user":     os.environ.get("DB_USER", "root"),
    "password": os.environ.get("DB_PASSWORD", ""),
    "database": os.environ.get("DB_NAME", "nutrimind_db"),
    "charset":  "utf8mb4"
}

# Path to ML model files (same folder as app.py)
BASE    = os.path.dirname(os.path.abspath(__file__))
MODEL   = os.path.join(BASE, "model.pkl")
SCALER  = os.path.join(BASE, "scaler.pkl")
ENCODER = os.path.join(BASE, "label_encoder.pkl")
META    = os.path.join(BASE, "model_meta.json")
MEALS   = os.path.join(BASE, "meal_plans.json")

# ─────────────────────────────────────────────────────────────
# LOAD ML MODEL
# ─────────────────────────────────────────────────────────────
with open(MODEL,   "rb") as f: model      = pickle.load(f)
with open(ENCODER, "rb") as f: le         = pickle.load(f)
with open(SCALER,  "rb") as f: scaler     = pickle.load(f)
with open(META,    "r")  as f: meta       = json.load(f)
with open(MEALS,   "r")  as f: meal_plans = json.load(f)

activity_map = meta["activity_map"]   # sedentary=0, light=1, moderate=2, active=2
gender_map   = meta["gender_map"]
use_scaled   = meta.get("use_scaled", False)

# ─────────────────────────────────────────────────────────────
# DB HELPER
# ─────────────────────────────────────────────────────────────
def get_db():
    return mysql.connector.connect(**DB_CONFIG)

# ─────────────────────────────────────────────────────────────
# ROUTE 1 — SIGNUP
# ─────────────────────────────────────────────────────────────
@app.route("/signup", methods=["POST"])
def signup():
    data = request.json
    name     = data.get("name", "").strip()
    phone    = data.get("phone", "").strip()
    email    = data.get("email", "").strip().lower()
    password = data.get("password", "")

    # Basic validation
    if not all([name, phone, email, password]):
        return jsonify({"status": "error", "message": "All fields are required."}), 400

    if len(password) < 8:
        return jsonify({"status": "error", "message": "Password too short."}), 400

    # Hash password
    pwd_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    try:
        db     = get_db()
        cursor = db.cursor()

        # Check duplicate email
        cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
        if cursor.fetchone():
            return jsonify({"status": "error", "message": "Email already registered. Please login."}), 409

        # Insert user
        cursor.execute(
            "INSERT INTO users (name, phone, email, password_hash) VALUES (%s, %s, %s, %s)",
            (name, phone, email, pwd_hash)
        )
        db.commit()
        user_id = cursor.lastrowid

        return jsonify({
            "status":  "success",
            "message": "Account created successfully!",
            "user_id": user_id,
            "name":    name
        })

    except mysql.connector.Error as e:
        return jsonify({"status": "error", "message": str(e)}), 500

    finally:
        cursor.close(); db.close()


# ─────────────────────────────────────────────────────────────
# ROUTE 2 — LOGIN
# ─────────────────────────────────────────────────────────────
@app.route("/login", methods=["POST"])
def login():
    data     = request.json
    email    = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not email or not password:
        return jsonify({"status": "error", "message": "Email and password required."}), 400

    try:
        db     = get_db()
        cursor = db.cursor(dictionary=True)

        cursor.execute(
            "SELECT id, name, password_hash FROM users WHERE email = %s",
            (email,)
        )
        user = cursor.fetchone()

        if not user:
            return jsonify({"status": "error", "message": "Email not registered. Please sign up."}), 404

        # Check password
        if not bcrypt.checkpw(password.encode(), user["password_hash"].encode()):
            return jsonify({"status": "error", "message": "Incorrect password. Please try again."}), 401

        return jsonify({
            "status":  "success",
            "message": "Login successful!",
            "user_id": user["id"],
            "name":    user["name"]
        })

    except mysql.connector.Error as e:
        return jsonify({"status": "error", "message": str(e)}), 500

    finally:
        cursor.close(); db.close()


# ─────────────────────────────────────────────────────────────
# ROUTE 3 — PREDICT (ML + Save Form + Save Result)
# ─────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    try:
        # ── Parse form inputs ──
        age        = float(data["age"])
        gender     = data["gender"].lower()
        height     = float(data["height"])
        weight     = float(data["weight"])
        bmi        = float(data.get("bmi") or round(weight / ((height / 100) ** 2), 1))
        bp         = float(data["bp"])
        sugar      = float(data["sugar"])
        water      = float(data["water_intake"])
        activity   = data["activity_level"].lower()
        smoking    = int(data["smoking"])
        alcohol    = int(data["alcohol"])
        email      = data.get("email", "").strip().lower()
        health_issues = [h.lower() for h in data.get("health_issues", [])]

        # ── Cholesterol proxy ──
        cholesterol = 180 + (bp - 120) * 0.3 + (bmi - 22) * 1.5
        cholesterol = round(max(150, min(300, cholesterol)), 1)

        activity_enc = activity_map.get(activity, 1)
        gender_enc   = gender_map.get(gender, 0)

        # ── Use direct health issue if user selected ──
        direct_map = {
            "diabetes":      "Diabetes",
            "hypertension":  "Hypertension",
            "obesity":       "Obesity",
            "heart disease": "Heart Disease",
            "thyroid":       "Thyroid",
            "normal":        "Normal",
        }
        predicted_condition = None
        for issue in health_issues:
            if issue in direct_map:
                predicted_condition = direct_map[issue]
                break

        # ── ML Prediction if no direct match ──
        if not predicted_condition:
            features = np.array([[age, bmi, bp, sugar, cholesterol,
                                   smoking, alcohol, activity_enc, gender_enc]])
            if use_scaled:
                features = scaler.transform(features)
            pred = model.predict(features)
            predicted_condition = le.inverse_transform(pred)[0]

        # ── Get 30-day meal plan ──
        plan_key_map = {"Heart Disease": "Heart disease"}
        plan_key     = plan_key_map.get(predicted_condition, predicted_condition)
        diet_plan    = meal_plans.get(plan_key, meal_plans["Normal"])

        # ── Save to DB if user is logged in ──
        form_id = None
        if email:
            try:
                db     = get_db()
                cursor = db.cursor(dictionary=True)

                # Get user_id from email
                cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
                user = cursor.fetchone()

                if user:
                    user_id = user["id"]

                    # Save health form → DB2
                    cursor.execute("""
                        INSERT INTO health_forms
                        (user_id, age, gender, height_cm, weight_kg, bmi,
                         water_intake, activity_level, bp, sugar,
                         alcohol, smoking, health_issues)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """, (
                        user_id, int(age), gender, height, weight, round(bmi, 2),
                        water, activity, int(bp), sugar,
                        alcohol, smoking,
                        ", ".join(data.get("health_issues", []))
                    ))
                    db.commit()
                    form_id = cursor.lastrowid

                    # Save diet result → DB3
                    cursor.execute("""
                        INSERT INTO diet_results
                        (user_id, form_id, predicted_condition, bmi, diet_plan_json)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (
                        user_id, form_id, predicted_condition,
                        round(bmi, 2), json.dumps(diet_plan)
                    ))
                    db.commit()

            except mysql.connector.Error as db_err:
                print(f"[DB Warning] {db_err}")  # Non-fatal — still return prediction
            finally:
                cursor.close(); db.close()

        return jsonify({
            "status":              "success",
            "predicted_condition": predicted_condition,
            "bmi":                 round(bmi, 2),
            "diet_plan":           diet_plan,
            "form_id":             form_id,
            "message":             f"30-day personalized diet plan generated for {predicted_condition}"
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


# ─────────────────────────────────────────────────────────────
# ROUTE 4 — GET LATEST RESULT for a user
# ─────────────────────────────────────────────────────────────
@app.route("/result/<email>", methods=["GET"])
def get_result(email):
    try:
        db     = get_db()
        cursor = db.cursor(dictionary=True)

        cursor.execute("SELECT id FROM users WHERE email = %s", (email.lower(),))
        user = cursor.fetchone()
        if not user:
            return jsonify({"status": "error", "message": "User not found."}), 404

        cursor.execute("""
            SELECT dr.predicted_condition, dr.bmi, dr.diet_plan_json, dr.generated_at,
                   hf.age, hf.gender, hf.bp, hf.sugar, hf.activity_level
            FROM diet_results dr
            JOIN health_forms hf ON dr.form_id = hf.id
            WHERE dr.user_id = %s
            ORDER BY dr.generated_at DESC
            LIMIT 1
        """, (user["id"],))

        row = cursor.fetchone()
        if not row:
            return jsonify({"status": "error", "message": "No results found for this user."}), 404

        return jsonify({
            "status":              "success",
            "predicted_condition": row["predicted_condition"],
            "bmi":                 row["bmi"],
            "diet_plan":           json.loads(row["diet_plan_json"]),
            "generated_at":        str(row["generated_at"]),
            "form_data": {
                "age":      row["age"],
                "gender":   row["gender"],
                "bp":       row["bp"],
                "sugar":    row["sugar"],
                "activity": row["activity_level"]
            }
        })

    except mysql.connector.Error as e:
        return jsonify({"status": "error", "message": str(e)}), 500

    finally:
        cursor.close(); db.close()


# ─────────────────────────────────────────────────────────────
# ROUTE 5 — GET HISTORY (all past results)
# ─────────────────────────────────────────────────────────────
@app.route("/history/<email>", methods=["GET"])
def get_history(email):
    try:
        db     = get_db()
        cursor = db.cursor(dictionary=True)

        cursor.execute("SELECT id FROM users WHERE email = %s", (email.lower(),))
        user = cursor.fetchone()
        if not user:
            return jsonify({"status": "error", "message": "User not found."}), 404

        cursor.execute("""
            SELECT dr.id, dr.predicted_condition, dr.bmi, dr.generated_at,
                   hf.age, hf.bp, hf.sugar
            FROM diet_results dr
            JOIN health_forms hf ON dr.form_id = hf.id
            WHERE dr.user_id = %s
            ORDER BY dr.generated_at DESC
        """, (user["id"],))

        rows = cursor.fetchall()
        history = [{
            "result_id":   r["id"],
            "condition":   r["predicted_condition"],
            "bmi":         r["bmi"],
            "age":         r["age"],
            "bp":          r["bp"],
            "sugar":       r["sugar"],
            "generated_at":str(r["generated_at"])
        } for r in rows]

        return jsonify({"status": "success", "history": history, "count": len(history)})

    except mysql.connector.Error as e:
        return jsonify({"status": "error", "message": str(e)}), 500

    finally:
        cursor.close(); db.close()


# ─────────────────────────────────────────────────────────────
# ROUTE 6 — HEALTH CHECK
# ─────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status":     "ok",
        "model":      meta["model_name"],
        "accuracy":   meta["accuracy"],
        "conditions": meta["conditions"],
        "timestamp":  str(datetime.now())
    })


# ─────────────────────────────────────────────────────────────
# ROUTE 7 — SERVE FRONTEND PAGES DIRECTLY FROM FLASK
# ─────────────────────────────────────────────────────────────
from flask import send_from_directory

# Frontend folder is one level up from backend/
FRONTEND = os.path.join(os.path.dirname(BASE), "frontend")

@app.route("/")
def index():
    return send_from_directory(FRONTEND, "index.html")

@app.route("/<path:filename>")
def serve_frontend(filename):
    # Only serve HTML/CSS/JS files, not API routes
    if "." in filename:
        return send_from_directory(FRONTEND, filename)
    return jsonify({"status": "error", "message": "Not found"}), 404


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import threading
    import webbrowser
    import time

    def open_browser():
        time.sleep(1.5)   # wait for Flask to fully start
        webbrowser.open("http://localhost:5000")

    print("=" * 52)
    print("  🌿 NutriMind is starting...")
    print("  🌐 Opening http://localhost:5000 in browser...")
    print("=" * 52)

    # Open browser in background thread
    threading.Thread(target=open_browser, daemon=True).start()

    # Run Flask (debug=False so browser opens only once)
    if __name__ == "__main__":
        port = int(os.environ.get("PORT", 5000))
        app.run(host="0.0.0.0", port=port)