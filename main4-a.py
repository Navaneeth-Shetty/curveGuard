import cv2
import numpy as np
import os
import time
import platform
import threading
import datetime
from priority_logic_b import get_signal_with_slope_priority, get_signal_no_slope

# YOLO config and weights paths
cfg = "yolov3/yolov3.cfg"
weights = "yolov3/yolov3.weights"
names = "yolov3/coco.names"
SPEED_LIMIT = 30

# Load YOLO model
net = cv2.dnn.readNet(weights, cfg)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
with open(names, "r") as f:
    classes = [line.strip() for line in f.readlines()]
layers = net.getUnconnectedOutLayersNames()

def continuous_beep(stop_event):
    if platform.system() == "Windows":
        import winsound
        while not stop_event.is_set():
            winsound.Beep(1000, 300)
            time.sleep(0.2)
    else:
        while not stop_event.is_set():
            os.system("play -nq -t alsa synth 0.3 sine 1000 > /dev/null 2>&1")

def estimate_distance(pw, rw=1.8, fl=700):
    return round((rw * fl) / pw, 2) if pw > 0 else -1

def save_overspeed_image(frame, side, speed):
    base_dir = "CapturedImages/overspeed"
    dir_path = os.path.join(base_dir, side)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    now = datetime.datetime.now()
    ts_display = now.strftime('%Y-%m-%d %H:%M:%S')
    ts_filename = now.strftime('%Y-%m-%d_%H-%M-%S')
    filename = os.path.join(dir_path, f"overspeed_{ts_filename}_{int(speed)}kmh.jpg")
    frame_to_save = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    color = (0, 255, 255)
    thickness = 2
    position = (10, 30)
    text_size, _ = cv2.getTextSize(ts_display, font, font_scale, thickness)
    cv2.rectangle(frame_to_save, (position[0]-5, position[1]-text_size[1]-5), (position[0]+text_size[0]+5, position[1]+5), (0,0,0), -1)
    cv2.putText(frame_to_save, ts_display, position, font, font_scale, color, thickness)
    cv2.imwrite(filename, frame_to_save)
    print(f"📸 Overspeed event captured and saved to: {filename}")

def detect_vehicle(frame, prev_dist, last_time):
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (416, 416)), 1/255, (416, 416), swapRB=True)
    net.setInput(blob)
    h, w = frame.shape[:2]
    best_detection = None
    max_area = 0
    for out in net.forward(layers):
        for det in out:
            scores = det[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            label = classes[class_id]
            if conf > 0.5 and label in ["car", "bus", "truck"]:
                box_w = int(det[2] * w)
                box_h = int(det[3] * h)
                area = box_w * box_h
                if area > max_area:
                    max_area = area
                    best_detection = (label, det[0:4])
    if best_detection:
        label, (cx, cy, bw, bh) = best_detection
        x = int((cx - bw/2) * w)
        y = int((cy - bh/2) * h)
        bw_px = int(bw * w)
        bh_px = int(bh * h)
        dist = estimate_distance(bw_px)
        now = time.time()
        dt = now - last_time if last_time else 1e-6
        speed = max(0, (prev_dist - dist) / dt * 3.6) if dist != -1 and prev_dist != -1 and dt > 0 else 0
        cv2.rectangle(frame, (x, y), (x + bw_px, y + bh_px), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {dist:.1f}m", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"{int(speed)} km/h", (x, y + bh_px + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 2)
        return label, dist, speed, now
    return "none", -1, 0, time.time()

def is_approaching(current, previous):
    return current != -1 and previous != -1 and current < previous

def get_user_configuration():
    slope_present = False
    left_orientation = 'flat'
    right_orientation = 'flat'
    if input("Is there a slope present? (y/n): ").lower() == 'y':
        slope_present = True
        left_choice = input("Is the LEFT side uphill or downhill? (up/down): ").lower()
        left_orientation = 'uphill' if left_choice == 'up' else 'downhill'
        right_choice = input("Is the RIGHT side uphill or downhill? (up/down): ").lower()
        right_orientation = 'uphill' if right_choice == 'up' else 'downhill'
    print("\n✅ Configuration set. Starting...\n")
    return slope_present, left_orientation, right_orientation

if __name__ == "__main__":
    slope_present, left_orientation, right_orientation = get_user_configuration()
    cap = cv2.VideoCapture(1)  # Change index or path as needed

    if not cap.isOpened():
        exit("❌ Error: Video feed failed to open.")

    signal = "green"
    is_locked = False
    lock_timer = time.time()
    hold_time = {"green": 5, "yellow": 2, "red": 5}
    last_printed_msg = ""
    beep_thread, beep_stop_event = None, threading.Event()
    prev_left_dist, prev_right_dist = -1, -1
    last_left_time, last_right_time = 0, 0
    left_speeding_logged, right_speeding_logged = False, False

    signal_display_config = {
        "green": {"terminal": "🟢 Safe to drive", "video": "Safe to Drive", "color": (0, 255, 0)},
        "yellow": {"terminal": "🟡 Be cautious", "video": "Be Cautious", "color": (0, 255, 255)},
        "red": {"terminal": "🔴 Danger", "video": "Danger", "color": (0, 0, 255)}
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Split frame into left and right halves
        height, width = frame.shape[:2]
        left_frame = frame[:, :width//2].copy()
        right_frame = frame[:, width//2:].copy()

        # Detect vehicles in each half
        labelL, distL, speedL, last_left_time = detect_vehicle(left_frame, prev_left_dist, last_left_time)
        labelR, distR, speedR, last_right_time = detect_vehicle(right_frame, prev_right_dist, last_right_time)

        nearL, nearR = (0 < distL < 15), (0 < distR < 15)
        approachL, approachR = is_approaching(distL, prev_left_dist), is_approaching(distR, prev_right_dist)
        prev_left_dist, prev_right_dist = distL, distR

        reason = ""
        if not is_locked:
            if slope_present:
                new_signal, reason = get_signal_with_slope_priority(labelL, nearL, approachL, labelR, nearR, approachR, left_orientation, right_orientation)
            else:
                new_signal, reason = get_signal_no_slope(labelL, nearL, approachL, labelR, nearR, approachR)
            if new_signal != signal:
                signal = new_signal
                lock_timer = time.time()
                is_locked = True
        elif time.time() - lock_timer > hold_time[signal]:
            is_locked = False

        # Audio alert for red signal
        if signal == "red" and (not beep_thread or not beep_thread.is_alive()):
            beep_stop_event.clear()
            beep_thread = threading.Thread(target=continuous_beep, args=(beep_stop_event,))
            beep_thread.start()
        elif signal != "red" and beep_thread and beep_thread.is_alive():
            beep_stop_event.set()

        display_config = signal_display_config[signal]
        if display_config["terminal"] != last_printed_msg:
            print(f"Signal: {display_config['terminal']}")
            last_printed_msg = display_config["terminal"]

        # Combine left and right frames side by side for display
        combined_frame = np.hstack((left_frame, right_frame))

        # Overlay signal info
        cv2.rectangle(combined_frame, (10, 10), (400, 80), (0, 0, 0), -1)
        cv2.putText(combined_frame, f"SIGNAL: {display_config['video']}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, display_config['color'], 2)

        if "PRIORITY" in reason or "DANGER" in reason:
            cv2.rectangle(combined_frame, (10, combined_frame.shape[0] - 50), (600, combined_frame.shape[0] - 10), (0, 0, 0), -1)
            cv2.putText(combined_frame, reason, (20, combined_frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 255), 2)

        # Speeding alerts
        speeding_alert_text = ""
        if speedL > SPEED_LIMIT:
            speeding_alert_text = f"LEFT: OVER SPEEDING ({int(speedL)} km/h)"
            if not left_speeding_logged:
                save_overspeed_image(left_frame, "left", speedL)
                left_speeding_logged = True
        elif labelL == "none":
            left_speeding_logged = False

        if speedR > SPEED_LIMIT:
            speeding_alert_text = f"RIGHT: OVER SPEEDING ({int(speedR)} km/h)"
            if not right_speeding_logged:
                save_overspeed_image(right_frame, "right", speedR)
                right_speeding_logged = True
        elif labelR == "none":
            right_speeding_logged = False

        if speeding_alert_text:
            cv2.rectangle(combined_frame, (combined_frame.shape[1] - 420, 10), (combined_frame.shape[1] - 10, 80), (0, 0, 0), -1)
            cv2.putText(combined_frame, "SPEED ALERT", (combined_frame.shape[1] - 400, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            cv2.putText(combined_frame, speeding_alert_text, (combined_frame.shape[1] - 400, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        cv2.imshow("🚦 Real-Time CCTV", combined_frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
            break

    beep_stop_event.set()
    cap.release()
    cv2.destroyAllWindows()
