import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# === 1. Load data ===
csv_path = "logs/decisions.csv"
if not os.path.exists(csv_path):
    print("❌ CSV file not found.")
    exit()

df = pd.read_csv(csv_path)

# === 2. Drop rows with missing or invalid values ===
df = df.dropna()
df = df[df["left_distance"] >= 0]
df = df[df["right_distance"] >= 0]

# === 3. Encode categorical labels ===
le_vehicle = LabelEncoder()
le_road = LabelEncoder()
le_decision = LabelEncoder()

df["left_vehicle_enc"] = le_vehicle.fit_transform(df["left_vehicle"])
df["right_vehicle_enc"] = le_vehicle.transform(df["right_vehicle"])
df["road_enc"] = le_road.fit_transform(df["road"])
df["decision_enc"] = le_decision.fit_transform(df["decision"])

# === 4. Prepare feature matrix X and target y ===
X = df[[
    "left_vehicle_enc", "right_vehicle_enc",
    "left_distance", "right_distance",
    "slope_diff", "left_slope", "right_slope",
    "road_enc"
]].astype(float)

y = df["decision_enc"]

# === 5. Train the model ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# === 6. Save model and encoders ===
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/traffic_priority_model.pkl")
joblib.dump(le_vehicle, "model/vehicle_encoder.pkl")
joblib.dump(le_road, "model/road_encoder.pkl")
joblib.dump(le_decision, "model/decision_encoder.pkl")

print("✅ Model training complete and saved.")
