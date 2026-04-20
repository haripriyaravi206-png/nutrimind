# NutriMind — Complete Setup Guide

## Project Folder Structure

```
nutrimind/
│
├── backend/                   ← Flask + ML
│   ├── app.py                 ← Main Flask app (all routes)
│   ├── database_setup.sql     ← MySQL tables (run once)
│   ├── requirements.txt       ← Python packages
│   ├── model.pkl              ← Trained ML model
│   ├── label_encoder.pkl      ← Label encoder
│   ├── model_meta.json        ← Model metadata
│   ├── meal_plans.json        ← 30-day meal plans
│   └── train_final_model.py   ← (optional) retrain model
│
└── frontend/                  ← HTML pages
    ├── index.html             ← Home page
    ├── login.html             ← Login page
    ├── signup.html            ← Signup page
    ├── form.html              ← Health form page
    └── result.html            ← Diet plan result page
```

---

## Step 1 — Install MySQL

Download and install MySQL Community Server:
https://dev.mysql.com/downloads/mysql/

Remember your root password!

---

## Step 2 — Create Database & Tables

Open MySQL Workbench or terminal and run:

```bash
mysql -u root -p < backend/database_setup.sql
```

OR open MySQL Workbench → paste contents of database_setup.sql → Run

This creates:
- Database: nutrimind_db
- Table 1: users         (id, name, phone, email, password_hash)
- Table 2: health_forms  (id, user_id, age, gender, height, weight, bmi, bp, sugar, ...)
- Table 3: diet_results  (id, user_id, form_id, predicted_condition, bmi, diet_plan_json)

---

## Step 3 — Install Python Packages

```bash
cd backend
pip install -r requirements.txt
```

---

## Step 4 — Copy ML Model Files

Make sure these files are in the backend/ folder:
- model.pkl
- label_encoder.pkl
- model_meta.json
- meal_plans.json

(Download from the diet_ml_final folder we created)

---

## Step 5 — Update DB Password in app.py

Open backend/app.py and find this section:

```python
DB_CONFIG = {
    "host":     "localhost",
    "user":     "root",
    "password": "your_mysql_password",   # ← CHANGE THIS
    "database": "nutrimind_db",
}
```

Replace "your_mysql_password" with your actual MySQL root password.

---

## Step 6 — Run Flask Server

```bash
cd backend
python app.py
```

You should see:
```
==================================================
  NutriMind Flask API running on http://localhost:5000
==================================================
```

---

## Step 7 — Open Frontend

Open the frontend/ folder and double-click index.html
OR use VS Code Live Server extension for best results.

---

## API Routes Summary

| Method | Route              | Purpose                        |
|--------|--------------------|--------------------------------|
| POST   | /signup            | Create new user account        |
| POST   | /login             | Authenticate user              |
| POST   | /predict           | ML prediction + save to DB     |
| GET    | /result/<email>    | Get latest diet result         |
| GET    | /history/<email>   | Get all past results           |
| GET    | /health            | API health check               |

---

## Test the API (using browser or Postman)

**Health check:**
GET http://localhost:5000/health

**Signup:**
POST http://localhost:5000/signup
Body: {"name":"Arjun Kumar","phone":"9876543210","email":"arjun@test.com","password":"Test@1234"}

**Login:**
POST http://localhost:5000/login
Body: {"email":"arjun@test.com","password":"Test@1234"}

**Predict:**
POST http://localhost:5000/predict
Body: {
  "age":35, "gender":"male", "height":170, "weight":80,
  "bp":135, "sugar":160, "water_intake":2.0,
  "activity_level":"sedentary", "smoking":0, "alcohol":0,
  "health_issues":[], "email":"arjun@test.com"
}

---

## Database Table Diagram

```
users (DB1)
├── id (PK)
├── name
├── phone
├── email (UNIQUE)
├── password_hash
└── created_at

health_forms (DB2)
├── id (PK)
├── user_id (FK → users.id)
├── age, gender, height_cm, weight_kg, bmi
├── water_intake, activity_level
├── bp, sugar
├── alcohol, smoking
├── health_issues
└── submitted_at

diet_results (DB3)
├── id (PK)
├── user_id (FK → users.id)
├── form_id (FK → health_forms.id)
├── predicted_condition
├── bmi
├── diet_plan_json (full 30-day plan)
└── generated_at
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "Access denied" MySQL | Check DB_CONFIG password in app.py |
| "Module not found" | Run pip install -r requirements.txt |
| "Cannot connect to server" | Make sure python app.py is running |
| CORS error in browser | flask-cors is installed and CORS(app) is in app.py |
| model.pkl not found | Copy ML files to backend/ folder |

---

Built with ❤️ — NutriMind Final Year Project
Flask + MySQL + Random Forest ML + HTML/CSS/JS
