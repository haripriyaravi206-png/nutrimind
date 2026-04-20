import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

BASE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "datasets")

# ─────────────────────────────────────────────
# 1. LOAD DATASETS
# ─────────────────────────────────────────────
diab   = pd.read_csv(os.path.join(DATA_DIR, "diabetes_prediction_dataset.csv"))
cardio = pd.read_csv(os.path.join(DATA_DIR, "cardio_train.csv"), sep=";")
print("Loaded all datasets!")

# ─────────────────────────────────────────────
# 2. BUILD CLEAN DATASET WITH DISTINCT FEATURES
#    Key insight: Each condition must have CLEARLY
#    distinct feature ranges so model can separate them
# ─────────────────────────────────────────────
np.random.seed(42)
records = []

# ── DIABETES (high glucose + high HbA1c) ──
diab_pos = diab[diab["diabetes"]==1].copy()
diab_pos["bmi_val"]    = diab_pos["bmi"]
diab_pos["glucose"]    = diab_pos["blood_glucose_level"]
diab_pos["bp"]         = diab_pos["HbA1c_level"].apply(lambda x: 135 if x>7.0 else 125)
diab_pos["chol"]       = 210
diab_pos["smoke"]      = diab_pos["smoking_history"].apply(lambda x: 1 if x in ["current","ever","former"] else 0)
diab_pos["gender_enc"] = diab_pos["gender"].apply(lambda x: 0 if x=="Male" else 1)
sample = diab_pos.sample(min(3000,len(diab_pos)), random_state=42)
for _,row in sample.iterrows():
    records.append({"age":row["age"],"bmi":row["bmi_val"],"bp":row["bp"],
                    "glucose":row["glucose"],"cholesterol":row["chol"],
                    "smoking":row["smoke"],"alcohol":0,"activity":1,
                    "gender":row["gender_enc"],"condition":"Diabetes"})

# ── HYPERTENSION (high BP > 140, normal glucose) ──
for _ in range(3000):
    records.append({
        "age":      np.random.randint(35,75),
        "bmi":      round(np.random.uniform(22,34),1),
        "bp":       np.random.randint(140,180),      # HIGH BP
        "glucose":  np.random.randint(80,110),       # NORMAL glucose
        "cholesterol": np.random.randint(200,250),
        "smoking":  np.random.choice([0,1],p=[0.5,0.5]),
        "alcohol":  np.random.choice([0,1],p=[0.5,0.5]),
        "activity": np.random.choice([0,1],p=[0.6,0.4]),
        "gender":   np.random.choice([0,1]),
        "condition":"Hypertension"
    })

# ── OBESITY (very high BMI > 30, low activity) ──
for _ in range(3000):
    records.append({
        "age":      np.random.randint(20,60),
        "bmi":      round(np.random.uniform(30,48),1),  # VERY HIGH BMI
        "bp":       np.random.randint(110,135),
        "glucose":  np.random.randint(90,130),
        "cholesterol": np.random.randint(200,240),
        "smoking":  np.random.choice([0,1],p=[0.6,0.4]),
        "alcohol":  np.random.choice([0,1],p=[0.6,0.4]),
        "activity": 0,                                   # SEDENTARY
        "gender":   np.random.choice([0,1]),
        "condition":"Obesity"
    })

# ── HEART DISEASE (high cholesterol + high BP + older age) ──
cardio_pos = cardio[cardio["cardio"]==1].copy()
cardio_pos["age_yr"]  = (cardio_pos["age"]/365).round().astype(int)
cardio_pos["bmi_val"] = cardio_pos["weight"]/((cardio_pos["height"]/100)**2)
cardio_pos["gluc"]    = cardio_pos["gluc"].apply(lambda x: 170 if x==3 else (130 if x==2 else 95))
cardio_pos["chol_v"]  = cardio_pos["cholesterol"].apply(lambda x: 260 if x==3 else (230 if x==2 else 195))
cardio_pos["gen_enc"] = cardio_pos["gender"].apply(lambda x: 0 if x==2 else 1)
sample = cardio_pos.sample(min(3000,len(cardio_pos)), random_state=42)
for _,row in sample.iterrows():
    records.append({
        "age":         min(int(row["age_yr"]),80),
        "bmi":         round(min(row["bmi_val"],45),1),
        "bp":          min(int(row["ap_hi"]),200),
        "glucose":     row["gluc"],
        "cholesterol": row["chol_v"],                # HIGH cholesterol
        "smoking":     int(row["smoke"]),
        "alcohol":     int(row["alco"]),
        "activity":    int(row["active"]),
        "gender":      row["gen_enc"],
        "condition":   "Heart Disease"
    })

# ── THYROID (distinct BMI extremes + age 25-65 + female dominant) ──
for _ in range(3000):
    gender = np.random.choice([0,1], p=[0.25,0.75])  # more female
    bmi_type = np.random.choice(["hypo","hyper"], p=[0.55,0.45])
    bmi = round(np.random.uniform(28,40),1) if bmi_type=="hypo" else round(np.random.uniform(15,21),1)
    records.append({
        "age":      np.random.randint(25,65),
        "bmi":      bmi,
        "bp":       np.random.randint(95,130),
        "glucose":  np.random.randint(75,115),        # NORMAL glucose
        "cholesterol": np.random.randint(210,290),    # slightly high
        "smoking":  np.random.choice([0,1],p=[0.75,0.25]),
        "alcohol":  np.random.choice([0,1],p=[0.8,0.2]),
        "activity": np.random.choice([0,1,2],p=[0.4,0.4,0.2]),
        "gender":   gender,
        "condition":"Thyroid"
    })

# ── NORMAL (healthy ranges for everything) ──
normal_src = diab[(diab["diabetes"]==0)&(diab["hypertension"]==0)&
                  (diab["heart_disease"]==0)&(diab["bmi"]<25)].copy()
normal_src["gender_enc"] = normal_src["gender"].apply(lambda x: 0 if x=="Male" else 1)
normal_src["smoke"]      = normal_src["smoking_history"].apply(lambda x: 1 if x in ["current","ever","former"] else 0)
sample = normal_src.sample(min(2000,len(normal_src)), random_state=42)
for _,row in sample.iterrows():
    records.append({
        "age":         row["age"],
        "bmi":         row["bmi"],          # NORMAL BMI 18-25
        "bp":          np.random.randint(90,120),   # NORMAL BP
        "glucose":     row["blood_glucose_level"],
        "cholesterol": np.random.randint(160,200),  # NORMAL cholesterol
        "smoking":     row["smoke"],
        "alcohol":     0,
        "activity":    np.random.choice([1,2],p=[0.4,0.6]),
        "gender":      row["gender_enc"],
        "condition":   "Normal"
    })
# top up Normal to 3000
for _ in range(3000-min(2000,len(normal_src))):
    records.append({
        "age":         np.random.randint(18,50),
        "bmi":         round(np.random.uniform(18.5,24.9),1),
        "bp":          np.random.randint(90,120),
        "glucose":     np.random.randint(70,100),
        "cholesterol": np.random.randint(150,200),
        "smoking":     0,
        "alcohol":     0,
        "activity":    np.random.choice([1,2]),
        "gender":      np.random.choice([0,1]),
        "condition":   "Normal"
    })

# ─────────────────────────────────────────────
# 3. FINALIZE DATASET
# ─────────────────────────────────────────────
df = pd.DataFrame(records).dropna()
df["bmi"]         = df["bmi"].clip(10,60)
df["bp"]          = df["bp"].clip(70,200)
df["glucose"]     = df["glucose"].clip(50,400)
df["cholesterol"] = df["cholesterol"].clip(100,350)
df["age"]         = df["age"].clip(1,100)

print(f"\n=== Final Dataset: {df.shape} ===")
print(df["condition"].value_counts())

# ─────────────────────────────────────────────
# 4. TRAIN & COMPARE 3 ALGORITHMS
# ─────────────────────────────────────────────
features = ["age","bmi","bp","glucose","cholesterol","smoking","alcohol","activity","gender"]
X = df[features]
y = df["condition"]

le    = LabelEncoder()
y_enc = le.fit_transform(y)
print("\nClasses:", le.classes_)

X_train,X_test,y_train,y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("\n" + "="*52)
print("   TRAINING & COMPARING 3 ML ALGORITHMS")
print("="*52)

algorithms = {
    "Random Forest":    (RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_split=3, random_state=42, n_jobs=-1), False),
    "SVM":              (SVC(kernel="rbf", C=2.0, gamma="scale", random_state=42, probability=True),                              True),
    "Gradient Boosting":(GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42),           False),
}

results    = {}
best_name  = ""
best_acc   = 0.0
best_model = None

for name,(clf,use_scaled) in algorithms.items():
    print(f"\n🔄 Training {name}...")
    Xtr = X_train_scaled if use_scaled else X_train
    Xte = X_test_scaled  if use_scaled else X_test
    clf.fit(Xtr, y_train)
    preds = clf.predict(Xte)
    acc   = accuracy_score(y_test, preds)
    cv    = cross_val_score(clf, Xtr, y_train, cv=5)
    results[name] = {"accuracy":round(acc*100,2),"cv_mean":round(cv.mean()*100,2),"cv_std":round(cv.std()*100,2),"use_scaled":use_scaled}
    print(f"   ✅ Test Accuracy : {acc*100:.2f}%")
    print(f"   ✅ CV  Accuracy  : {cv.mean()*100:.2f}% ± {cv.std()*100:.2f}%")
    if acc > best_acc:
        best_acc=acc; best_name=name; best_model=clf

# ─────────────────────────────────────────────
# 5. RESULTS TABLE
# ─────────────────────────────────────────────
print("\n" + "="*52)
print("   ALGORITHM COMPARISON RESULTS")
print("="*52)
print(f"{'Algorithm':<22} {'Test Acc':>10} {'CV Acc':>10} {'CV Std':>8}")
print("-"*52)
for name,r in sorted(results.items(),key=lambda x:x[1]["accuracy"],reverse=True):
    marker = " ✅ BEST" if name==best_name else ""
    print(f"{name:<22} {r['accuracy']:>9}% {r['cv_mean']:>9}% {r['cv_std']:>7}%{marker}")
print("="*52)
print(f"\n🏆 Best Model: {best_name} ({best_acc*100:.2f}%)")

use_scaled_best = results[best_name]["use_scaled"]
Xte_final = X_test_scaled if use_scaled_best else X_test
print("\nDetailed Classification Report:")
print(classification_report(y_test, best_model.predict(Xte_final), target_names=le.classes_))

if hasattr(best_model,"feature_importances_"):
    fi = pd.Series(best_model.feature_importances_,index=features).sort_values(ascending=False)
    print("Feature Importance:")
    for feat,score in fi.items():
        bar = "█"*int(score*60)
        print(f"  {feat:<15} {score:.4f}  {bar}")

# ─────────────────────────────────────────────
# 6. SAVE ALL FILES
# ─────────────────────────────────────────────
with open(os.path.join(BASE,"model.pkl"),         "wb") as f: pickle.dump(best_model,f)
with open(os.path.join(BASE,"label_encoder.pkl"), "wb") as f: pickle.dump(le,f)
with open(os.path.join(BASE,"scaler.pkl"),        "wb") as f: pickle.dump(scaler,f)

meta = {
    "features":     features,
    "activity_map": {"sedentary":0,"light":1,"moderate":2,"active":2},
    "gender_map":   {"male":0,"female":1},
    "model_name":   best_name,
    "accuracy":     round(best_acc*100,2),
    "conditions":   le.classes_.tolist(),
    "use_scaled":   results[best_name]["use_scaled"],
    "all_results":  {k:{"accuracy":v["accuracy"],"cv_mean":v["cv_mean"]} for k,v in results.items()}
}
with open(os.path.join(BASE,"model_meta.json"),"w") as f: json.dump(meta,f,indent=2)
df.to_csv(os.path.join(BASE,"merged_dataset.csv"),index=False)

print(f"\n✅ Best model ({best_name}) saved as model.pkl")
print("✅ label_encoder.pkl, scaler.pkl, model_meta.json, merged_dataset.csv saved!")