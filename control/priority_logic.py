import joblib
import os

# === Load model and encoders once ===
model_path = "model/model.pkl"
enc_path = "model/encoders.pkl"

if not os.path.exists(model_path) or not os.path.exists(enc_path):
    print("❌ Trained model or encoders not found.")
    model = None
    encoders = None
else:
    model = joblib.load(model_path)
    encoders = joblib.load(enc_path)

# === Predict decision from data ===
def decide_priority(data):
    if model is None or encoders is None:
        return "RIGHT GREEN | LEFT RED"  # fallback default
    
    try:
        # encode categorical inputs
        left_v = encoders["left_vehicle"].transform([data["left_vehicle"]])[0]
        right_v = encoders["right_vehicle"].transform([data["right_vehicle"]])[0]
        road = encoders["road"].transform([data["road"]])[0]

        # build input feature list
        features = [[
            left_v,
            right_v,
            data["left_distance"],
            data["right_distance"],
            data["slope_diff"],
            data["left_slope"],
            data["right_slope"],
            road
        ]]

        # predict and decode label
        pred = model.predict(features)[0]
        decision = encoders["decision"].inverse_transform([pred])[0]
        return decision

    except Exception as e:
        print(f"⚠️ ML prediction error: {e}")
        return "BOTH RED"  # safe fallback
