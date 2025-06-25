import cv2
import numpy as np
import os
import time
import platform
import threading

def continuous_beep(stop_event):
    if platform.system() == "Windows":
        import winsound
        while not stop_event.is_set():
            winsound.Beep(1000, 300)
            time.sleep(0.2)
    elif platform.system() in ("Linux", "Darwin"):
        while not stop_event.is_set():
            os.system("play -nq -t alsa synth 0.3 sine 1000")

cfg = "yolov3/yolov3.cfg"
weights = "yolov3/yolov3.weights"
names = "yolov3/coco.names"
net = cv2.dnn.readNet(weights, cfg)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

with open(names, "r") as f:
    classes = [line.strip() for line in f.readlines()]
layers = net.getUnconnectedOutLayersNames()

def estimate_distance(pixel_width, real_width=1.8, focal_length=700):
    if pixel_width <= 0: return -1
    return round((real_width * focal_length) / pixel_width, 2)

def detect_vehicle(frame, prev_dist, last_time):
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (416, 416)), 1/255, (416, 416), swapRB=True)
    net.setInput(blob)
    outs = net.forward(layers)
    h, w = frame.shape[:2]

    for out in outs:
        for det in out:
            scores = det[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            label = classes[class_id]
            if conf > 0.5 and label in ["car", "bus", "truck"]:
                cx, cy, bw, bh = det[0:4]
                x = int((cx - bw / 2) * w)
                y = int((cy - bh / 2) * h)
                bw_px = int(bw * w)
                bh_px = int(bh * h)
                dist = estimate_distance(bw_px)

                now = time.time()
                dt = now - last_time if last_time else 1e-6
                speed = 0
                if dist != -1 and prev_dist != -1 and dt > 0:
                    speed = max(0, (prev_dist - dist) / dt * 3.6)

                cv2.rectangle(frame, (x, y), (x + bw_px, y + bh_px), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {dist:.1f}m", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"{speed:.1f} km/h", (x, y + bh_px + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 2)

                return label, dist, speed, now

    return "none", -1, 0, time.time()

def is_approaching(current, previous):
    return current != -1 and previous != -1 and current < previous

cap_left = cv2.VideoCapture(1)
cap_right = cv2.VideoCapture('videos/demo4.mp4')
if not cap_left.isOpened() or not cap_right.isOpened():
    print("❌ Feed failed.")
    exit()

signal = "green"
lock = False
timer = time.time()
hold_time = {"green": 5, "yellow": 2, "red": 4}
last_printed = ""
beep_thread = None
beep_stop = threading.Event()

prev_left_dist = prev_right_dist = -1
last_left_time = last_right_time = 0

while True:
    ret1, fL = cap_left.read()
    ret2, fR = cap_right.read()
    cap_right.read()
    cap_right.read()
    if not ret1:
        break
    if not ret2:
        cap_right.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret2, fR = cap_right.read()
        cap_right.read()
        cap_right.read()
        if not ret2:
            break

    labelL, distL, speedL, last_left_time = detect_vehicle(fL, prev_left_dist, last_left_time)
    labelR, distR, speedR, last_right_time = detect_vehicle(fR, prev_right_dist, last_right_time)

    nearL = 0 < distL < 12
    nearR = 0 < distR < 12
    approachL = is_approaching(distL, prev_left_dist)
    approachR = is_approaching(distR, prev_right_dist)
    prev_left_dist = distL
    prev_right_dist = distR

    if not lock:
        if nearL and nearR:
            new_signal = "red"
        elif (nearL and approachR) or (nearR and approachL):
            new_signal = "yellow"
        elif nearL or nearR:
            new_signal = "green"
        else:
            new_signal = "green"

        if new_signal != signal:
            signal = new_signal
            timer = time.time()
            lock = True
    else:
        if time.time() - timer > hold_time[signal]:
            lock = False

    if signal == "red":
        if not beep_thread or not beep_thread.is_alive():
            beep_stop.clear()
            beep_thread = threading.Thread(target=continuous_beep, args=(beep_stop,))
            beep_thread.start()
    else:
        if beep_thread and beep_thread.is_alive():
            beep_stop.set()

    icons = {"green": "🟢", "yellow": "🟡", "red": "🔴"}
    if signal != last_printed:
        print(icons[signal])
        last_printed = signal

    fL = cv2.resize(fL, (640, 360))
    fR = cv2.resize(fR, (640, 360))
    combined = np.hstack((fL, fR))
    col = {"green": (0,255,0), "yellow": (0,255,255), "red": (0,0,255)}[signal]
    cv2.rectangle(combined, (10, 10), (240, 70), (0, 0, 0), -1)
    cv2.putText(combined, f"{signal.upper()} SIGNAL", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)

    cv2.imshow("🚦 Real-Time CCTV", combined)
    if cv2.waitKey(1) & 0xFF == 27:
        break

beep_stop.set()
cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
