import cv2
import numpy as np
import os
import csv
from datetime import datetime
from sensors.slope_estimator import get_slope
from sensors.weather_sensor import estimate_road_condition

# === YOLOv3 Config ===
cfg_path = "yolov3/yolov3.cfg"
weights_path = "yolov3/yolov3.weights"
names_path = "yolov3/coco.names"

if not all(os.path.exists(p) for p in [cfg_path, weights_path, names_path]):
    print("❌ YOLO files missing.")
    exit()

net = cv2.dnn.readNet(weights_path, cfg_path)
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getUnconnectedOutLayersNames()

def estimate_distance(pixel_width, real_width=2.5, focal_length=300):
    if pixel_width <= 0:
        return -1
    return round((real_width * focal_length) / pixel_width, 2)

def detect_vehicle_and_distance(frame, draw=True):
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(layer_names)
    h, w = frame.shape[:2]
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            label = classes[class_id]
            if conf > 0.5 and label in ["car", "bus", "truck"]:
                center_x, center_y, bw, bh = (detection[0:4] * np.array([w, h, w, h])).astype(int)
                x = int(center_x - bw / 2)
                y = int(center_y - bh / 2)
                dist = estimate_distance(bw)
                if draw:
                    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {dist}m", (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                return label, dist
    return "none", -1

def set_light(decision):
    if decision == "left":
        print("🔴 LEFT RED | 🟢 RIGHT GREEN")
    elif decision == "right":
        print("🟢 LEFT GREEN | 🔴 RIGHT RED")
    else:
        print("🟢 BOTH GREEN")
    return decision

left_alt = 320
right_alt = 300
temp = 18
humidity = 85

cap_left = cv2.VideoCapture("videos/demo2.mp4")
cap_right = cv2.VideoCapture("videos/demo2.mp4")
if not cap_left.isOpened() or not cap_right.isOpened():
    print("❌ One or both camera feeds failed to open.")
    exit()

log_file = "logs/decisions.csv"
os.makedirs("logs", exist_ok=True)
if not os.path.exists(log_file):
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame", "timestamp", "left_vehicle", "right_vehicle",
            "left_distance", "right_distance",
            "slope_diff", "left_slope", "right_slope",
            "road", "decision"
        ])

frame_id = 0
while True:
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()
    if not ret_left or not ret_right:
        print("✅ End of stream or capture failed.")
        break

    frame_id += 1
    if frame_id % 30 != 0:
        continue

    left_vehicle, left_dist = detect_vehicle_and_distance(frame_left)
    right_vehicle, right_dist = detect_vehicle_and_distance(frame_right)
    slope_diff = get_slope(left_alt, right_alt)
    road_condition = estimate_road_condition(temp, humidity)

    if left_dist == -1 and right_dist == -1:
        decision = "both"
    elif left_dist == -1:
        decision = "right"
    elif right_dist == -1:
        decision = "left"
    elif abs(left_dist - right_dist) < 2:
        decision = "both"
    elif left_dist < right_dist:
        decision = "left"
    else:
        decision = "right"

    decision_str = set_light(decision)

    print(f"🍉 Frame {frame_id}")
    print(f"🚗 Left: {left_vehicle} @ {left_dist}m | 🚙 Right: {right_vehicle} @ {right_dist}m")
    print(f"Decision: {decision_str}")

    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            frame_id,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            left_vehicle,
            right_vehicle,
            left_dist,
            right_dist,
            slope_diff,
            left_alt,
            right_alt,
            road_condition,
            decision_str
        ])

    screen_res = 1280, 720
    margin = 20
    win_w = screen_res[0] - 2 * margin
    win_h = screen_res[1] - 2 * margin
    frame_w = win_w // 2
    frame_h = win_h

    frame_left = cv2.resize(frame_left, (frame_w, frame_h))
    frame_right = cv2.resize(frame_right, (frame_w, frame_h))
    combined = np.hstack((frame_left, frame_right))
    padded = cv2.copyMakeBorder(combined, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    cv2.imshow("📺 Dual View", padded)

    if cv2.waitKey(1) == 27:
        break

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
