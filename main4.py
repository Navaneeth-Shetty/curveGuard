# main4.py — single-camera 90° curve system
# shows each vehicle’s arc-distance to the junction and, when both are detected,
# shows / prints the curved distance between them (dl + dr).

import cv2, numpy as np, os, time, platform, threading, datetime, math
from priority_logic_b import get_signal_with_slope_priority, get_signal_no_slope

# yolo files
CFG, WGT, NMS = "yolov3/yolov3.cfg", "yolov3/yolov3.weights", "yolov3/coco.names"
net = cv2.dnn.readNet(WGT, CFG)
net.setPreferableBackend(0)
net.setPreferableTarget(0)
with open(NMS) as f:
    classes = [l.strip() for l in f]
layers = net.getUnconnectedOutLayersNames()

# constants
SPEED_LIMIT = 30          # km/h
CURVE_R = 30             # metres, radius of each quarter-circle
HOLD = {"green": 5, "yellow": 2, "red": 5}

# helpers
def beep(stop):
    pc = platform.system() == "Windows"
    while not stop.is_set():
        if pc:
            import winsound; winsound.Beep(1000, 300)
        else:
            os.system("play -q -n synth 0.3 sin 1000")
        time.sleep(0.2)

def curved_dist(px, w, is_left):
    # map both halves to the same 0‒1 scale toward the junction
    norm = px / w if is_left else (w - px) / w
    theta = max(0, min(norm, 1)) * (math.pi / 2)          # radians from start
    return round(CURVE_R * (math.pi / 2 - theta), 2)       # metres remaining

def save_overspeed(img, side, sp):
    path = f"CapturedImages/overspeed/{side}"
    os.makedirs(path, exist_ok=True)
    t = datetime.datetime.now()
    name = t.strftime("%F_%H-%M-%S")
    cv2.putText(img, t.strftime("%F %T"), (10, 30), 0, .7, (0, 255, 255), 2)
    cv2.imwrite(f"{path}/over_{name}_{int(sp)}kmh.jpg", img)

def detect(view, prev_d, prev_t, is_left):
    blob = cv2.dnn.blobFromImage(cv2.resize(view, (416, 416)),
                                 1/255, (416, 416), swapRB=True)
    net.setInput(blob)
    h, w = view.shape[:2]
    best, best_area = None, 0
    for out in net.forward(layers):
        for det in out:
            scores = det[5:]
            cid = np.argmax(scores)
            conf = scores[cid]
            label = classes[cid]
            if conf > .5 and label in ("car", "bus", "truck"):
                area = int(det[2]*w) * int(det[3]*h)
                if area > best_area:
                    best_area = area
                    best = (label, det[0:4])
    if best:
        label, (cx, cy, bw, bh) = best
        x = int((cx - bw/2) * w)
        y = int((cy - bh/2) * h)
        bw = int(bw * w)
        bh = int(bh * h)
        px_center = x + bw // 2
        dist_j = curved_dist(px_center, w, is_left)
        now = time.time()
        dt = now - prev_t if prev_t else 1e-6
        speed = max(0, (prev_d - dist_j) / dt * 3.6) if prev_d != -1 else 0
        cv2.rectangle(view, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.putText(view, f"{label} {dist_j:.1f} m",
                    (x, y - 10), 0, .6, (255, 255, 255), 2)
        cv2.putText(view, f"{int(speed)} km/h",
                    (x, y + bh + 20), 0, .5, (200, 200, 255), 2)
        return label, dist_j, speed, now
    return "none", -1, 0, time.time()

# slope configuration (optional priority logic)
slope = False
left_ori = right_ori = "flat"
if input("slope present? (y/n): ") == "y":
    slope = True
    left_ori  = "uphill" if input("left uphill? (y/n): ")  == "y" else "downhill"
    right_ori = "uphill" if input("right uphill? (y/n): ") == "y" else "downhill"

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise SystemExit("❌ camera error")

# state
sig, lock, t_lock = "green", False, time.time()
prev_L, prev_R, tL, tR = -1, -1, 0, 0
beeper = None
stop_event = threading.Event()
logL = logR = False
console_last = ""

# ui colors
disp = {"green": ("🟢 safe",   "Safe",    (0, 255,   0)),
        "yellow":("🟡 caution","Caution", (0, 255, 255)),
        "red":   ("🔴 danger", "Danger",  (0,   0, 255))}

while True:
    ok, frame = cap.read()
    if not ok:
        break

    h, w = frame.shape[:2]
    half = w // 2
    left  = frame[:, :half].copy()
    right = frame[:, half:].copy()

    lblL, dL, spL, tL = detect(left,  prev_L, tL, True)
    lblR, dR, spR, tR = detect(right, prev_R, tR, False)
    prev_L, prev_R = dL, dR

    nearL = 0 < dL < 15
    nearR = 0 < dR < 15
    appL  = dL != -1 and prev_L != -1 and dL < prev_L
    appR  = dR != -1 and prev_R != -1 and dR < prev_R

    if not lock:
        if slope:
            sig_new, _ = get_signal_with_slope_priority(
                lblL, nearL, appL, lblR, nearR, appR, left_ori, right_ori)
        else:
            sig_new, _ = get_signal_no_slope(lblL, nearL, appL,
                                             lblR, nearR, appR)
        if sig_new != sig:
            sig, lock, t_lock = sig_new, True, time.time()
    elif time.time() - t_lock > HOLD[sig]:
        lock = False

    # continuous beep on red
    if sig == "red" and (not beeper or not beeper.is_alive()):
        stop_event.clear()
        beeper = threading.Thread(target=beep, args=(stop_event,))
        beeper.start()
    elif sig != "red" and beeper and beeper.is_alive():
        stop_event.set()

    # compose output frame
    dash = np.hstack((cv2.resize(left, (640, 360)),
                      cv2.resize(right, (640, 360))))
    tag, tag_v, col = disp[sig]
    cv2.putText(dash, f"signal: {tag_v}", (20, 40), 0, 1, col, 2)

    # distance between vehicles if both present
    if dL != -1 and dR != -1:
        dist_between = round(dL + dR, 2)          # metres along curve
        cv2.putText(dash, f"distance between vehicles: {dist_between} m",
                    (20, 80), 0, .8, (255, 255, 0), 2)
        if console_last != dist_between:
            print(f"distance between vehicles: {dist_between} m")
            console_last = dist_between

    # overspeed alerts
    if spL > SPEED_LIMIT:
        if not logL:
            save_overspeed(left, "left", spL)
            logL = True
    elif lblL == "none":
        logL = False
    if spR > SPEED_LIMIT:
        if not logR:
            save_overspeed(right, "right", spR)
            logR = True
    elif lblR == "none":
        logR = False

    cv2.imshow("🚦 real-time cctv", dash)
    if cv2.waitKey(1) == 27:
        break

stop_event.set()
cap.release()
cv2.destroyAllWindows()
