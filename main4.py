# main4.py — single camera, split-frame curved road system

import cv2, numpy as np, os, time, platform, threading, datetime, math
from priority_logic_b import get_signal_with_slope_priority, get_signal_no_slope

CFG, WGT, NMS = "yolov3/yolov3.cfg", "yolov3/yolov3.weights", "yolov3/coco.names"
net = cv2.dnn.readNet(WGT, CFG); net.setPreferableBackend(0); net.setPreferableTarget(0)
with open(NMS) as f: classes = [l.strip() for l in f]; layers = net.getUnconnectedOutLayersNames()

SPEED_LIMIT = 30
CURVE_R = 30
HOLD = {"green": 5, "yellow": 2, "red": 5}

def beep(stop): 
    pc = platform.system() == "Windows"
    while not stop.is_set():
        if pc:
            import winsound
            winsound.Beep(1000, 300)
        else:
            os.system("play -q -n synth 0.3 sin 1000")
        time.sleep(0.2)

def curved_dist(px, w): 
    θ = (px - 0.1 * w) / (0.8 * w) * (math.pi / 2)
    θ = max(0, min(θ, math.pi / 2))
    return round(CURVE_R * (math.pi / 2 - θ), 2)

def overspeed(frame, side, sp):
    d = f"CapturedImages/overspeed/{side}"
    os.makedirs(d, exist_ok=True)
    ts = datetime.datetime.now()
    name = ts.strftime("%F_%H-%M-%S")
    cv2.putText(frame, ts.strftime("%F %T"), (10, 30), 0, 0.7, (0, 255, 255), 2)
    cv2.imwrite(f"{d}/over_{name}_{int(sp)}kmh.jpg", frame)

def detect(sub, prev, t0):
    blob = cv2.dnn.blobFromImage(cv2.resize(sub, (416, 416)), 1/255, (416, 416), swapRB=True)
    net.setInput(blob)
    h, w = sub.shape[:2]
    best, area = None, 0
    for o in net.forward(layers):
        for d in o:
            sco = d[5:]
            cid = np.argmax(sco)
            conf, lbl = sco[cid], classes[cid]
            if conf > 0.5 and lbl in ["car", "bus", "truck"]:
                ar = int(d[2]*w) * int(d[3]*h)
                if ar > area:
                    area = ar
                    best = (lbl, d[:4])
    if best:
        lbl, (cx, cy, bw, bh) = best
        x, y = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        bw, bh = int(bw * w), int(bh * h)
        px_center = x + bw // 2
        dist = curved_dist(px_center, w)
        now = time.time()
        dt = now - t0 if t0 else 1e-6
        speed = max(0, (prev - dist) / dt * 3.6) if prev != -1 else 0
        cv2.rectangle(sub, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.putText(sub, f"{lbl} {dist:.1f}m", (x, y - 10), 0, 0.6, (255, 255, 255), 2)
        cv2.putText(sub, f"{int(speed)} km/h", (x, y + bh + 20), 0, 0.5, (200, 200, 255), 2)
        return lbl, dist, speed, now
    return "none", -1, 0, time.time()

if __name__ == "__main__":
    slope = False
    slopeL, slopeR = "flat", "flat"
    if input("slope present? (y/n): ") == "y":
        slope = True
        slopeL = "uphill" if input("left uphill? (y/n): ") == "y" else "downhill"
        slopeR = "uphill" if input("right uphill? (y/n): ") == "y" else "downhill"
    print("✅ ready")

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        quit("❌ camera error")

    sig = "green"
    lock = False
    timer = time.time()
    last = ""
    beeper = None
    stop = threading.Event()
    pl, pr, tl, tr = -1, -1, 0, 0
    logL, logR = False, False

    disp = {
        "green": ("🟢 safe", "Safe to Drive", (0, 255, 0)),
        "yellow": ("🟡 caution", "Be Cautious", (0, 255, 255)),
        "red": ("🔴 danger", "Danger", (0, 0, 255))
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        mid = w // 2
        left = frame[:, :mid].copy()
        right = frame[:, mid:].copy()

        l, dl, slv, tl = detect(left, pl, tl)
        r, dr, srv, tr = detect(right, pr, tr)

        nearL = 0 < dl < 15
        nearR = 0 < dr < 15
        appL = dl != -1 and pl != -1 and dl < pl
        appR = dr != -1 and pr != -1 and dr < pr
        pl, pr = dl, dr

        if not lock:
            if slope:
                sig_new, reason = get_signal_with_slope_priority(l, nearL, appL, r, nearR, appR, slopeL, slopeR)
            else:
                sig_new, reason = get_signal_no_slope(l, nearL, appL, r, nearR, appR)
            if sig_new != sig:
                sig = sig_new
                lock = True
                timer = time.time()
        elif time.time() - timer > HOLD[sig]:
            lock = False

        if sig == "red" and (not beeper or not beeper.is_alive()):
            stop.clear()
            beeper = threading.Thread(target=beep, args=(stop,))
            beeper.start()
        elif sig != "red" and beeper and beeper.is_alive():
            stop.set()

        txt, dispVid, col = disp[sig]
        if txt != last:
            print(f"signal: {txt}")
            last = txt

        out = np.hstack((cv2.resize(left, (640, 360)), cv2.resize(right, (640, 360))))
        cv2.rectangle(out, (out.shape[1] - 410, 10), (out.shape[1] - 10, 80), (0, 0, 0), -1)
        cv2.putText(out, f"signal: {dispVid}", (out.shape[1] - 400, 55), 0, 1, col, 2)

        alert = ""
        if slv > SPEED_LIMIT:
            alert = f"left overspeed {int(slv)} km/h"
            if not logL:
                overspeed(left, "left", slv)
                logL = True
        elif l == "none":
            logL = False

        if srv > SPEED_LIMIT:
            alert = f"right overspeed {int(srv)} km/h"
            if not logR:
                overspeed(right, "right", srv)
                logR = True
        elif r == "none":
            logR = False

        if alert:
            cv2.rectangle(out, (10, 10), (410, 80), (0, 0, 0), -1)
            cv2.putText(out, "speed alert", (20, 40), 0, 0.8, (0, 165, 255), 2)
            cv2.putText(out, alert, (20, 70), 0, 0.7, (0, 165, 255), 2)

        cv2.imshow("🚦 real-time cctv", out)
        if cv2.waitKey(1) == 27:
            break

    stop.set()
    cap.release()
    cv2.destroyAllWindows()
