import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# === 1. Load the updated CSV ===
log_file = "logs/decisions.csv"
if not os.path.exists(log_file):
    print("❌ decisions.csv not found.")
    exit()

df = pd.read_csv(log_file)

# === 2. Drop incomplete rows ===
df.dropna(inplace=True)

# === 3. Encode categorical columns ===
encoders = {}
for col in ["left_vehicle", "right_vehicle", "road", "decision"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# === 4. Feature columns ===
features = [
    "left_vehicle", "right_vehicle",
    "left_distance", "right_distance",
    "slope_diff", "left_slope", "right_slope",
    "road"
]
X = df[features]
y = df["decision"]

# === 5. Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === 6. Train RandomForest ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === 7. Evaluate ===
acc = model.score(X_test, y_test)
print(f"✅ Model trained — Accuracy: {acc * 100:.2f}%")

# === 8. Save model and encoders ===
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.pkl")
joblib.dump(encoders, "model/encoders.pkl")
print("📦 Saved: model/model.pkl and model/encoders.pkl")
