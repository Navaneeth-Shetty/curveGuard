#!/usr/bin/env python3
# curve‑road safety ai — moving‑object metre‑based gap logic with speed + overspeed image crop

from __future__ import annotations
from dataclasses import dataclass, field
import time, sys, math, cv2, numpy as np, os, datetime

try:
    import winsound
    _CAN_BEEP = True
except ImportError:
    _CAN_BEEP = False

@dataclass
class Config:
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    curve_radius_px: int = 400
    caution_zone_px: int = 250
    conf_thresh: float = 0.5
    iou_thresh_nms: float = 0.45
    centroid_match_px: int = 100
    iou_match_thresh: float = 0.30
    history_len: int = 5
    focal_length_px: int = 800
    curve_length_m: float = 3.0
    min_safe_gap_m: float = 30.0
    speed_limit_kmh: float = 200.0
    beep_freq: int = 1000
    beep_dur: int = 200
    beep_cooldown: float = 2.0
    min_motion_area: int = 1500
    vehicle_widths: dict[str, float] = field(default_factory=lambda: {"obj": 1.8})

    def __post_init__(self): self.cx = self.frame_width // 2
    @property
    def vehicle_classes(self): return {"obj"}

def bbox_iou(a, b):
    x1a, y1a, wa, ha = a; x2a, y2a = x1a + wa, y1a + ha
    x1b, y1b, wb, hb = b; x2b, y2b = x1b + wb, y1b + hb
    xa, ya = max(x1a, x1b), max(y1a, y1b); xb, yb = min(x2a, x2b), min(y2a, y2b)
    iw, ih = max(0, xb - xa), max(0, yb - ya); inter = iw * ih
    return 0.0 if inter == 0 else inter / (wa * ha + wb * hb - inter)

def estimate_distance(width_px: int, cls: str, cfg: Config) -> float:
    real = cfg.vehicle_widths.get(cls, 1.8)
    return float('inf') if width_px == 0 else (real * cfg.focal_length_px) / width_px

def beep(cfg: Config):
    if _CAN_BEEP: winsound.Beep(cfg.beep_freq, cfg.beep_dur)
    else: print("\a", end="", flush=True)

def save_cropped_vehicle(frame, box, side):
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = f"CapturedImages/overspeed/{side}"
    os.makedirs(path, exist_ok=True)
    x, y, w, h = box
    pad = 20
    x1 = max(x - pad, 0)
    y1 = max(y - pad, 0)
    x2 = min(x + w + pad, frame.shape[1])
    y2 = min(y + h + pad, frame.shape[0])
    crop = frame[y1:y2, x1:x2]
    cv2.imwrite(f"{path}/{ts}.jpg", crop)

class Tracker:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cents, self.boxes, self.names, self.lost = {}, {}, {}, {}
        self.last_d, self.last_t, self.speed = {}, {}, {}
        self.next_id = 0

    def _pair(self, a, b): return np.linalg.norm(a[:, None] - b[None, :], axis=2)

    def _add(self, c, b, n, t):
        oid = self.next_id
        self.cents[oid] = c
        self.boxes[oid] = b
        self.names[oid] = n
        self.lost[oid] = 0
        self.last_d[oid] = None
        self.last_t[oid] = t
        self.speed[oid] = 0.0
        self.next_id += 1

    def _update_speed(self, oid, width_px, t):
        d = estimate_distance(width_px, self.names[oid], self.cfg)
        prev_d, prev_t = self.last_d[oid], self.last_t[oid]
        self.last_d[oid], self.last_t[oid] = d, t
        if prev_d is None or t == prev_t: return
        self.speed[oid] = abs((prev_d - d) / (t - prev_t)) * 3.6

    def update(self, cents, boxes, names, t):
        cfg = self.cfg
        if not cents:
            for oid in list(self.lost):
                self.lost[oid] += 1
                if self.lost[oid] > cfg.history_len:
                    for d in (self.cents, self.boxes, self.names,
                              self.lost, self.last_d, self.last_t, self.speed):
                        d.pop(oid, None)
            return
        if not self.cents:
            for c, b, n in zip(cents, boxes, names):
                self._add(c, b, n, t)
            return
        oids = list(self.cents)
        D = self._pair(np.array([self.cents[i] for i in oids]), np.array(cents))
        used_r, used_c = set(), set()
        for r in D.min(1).argsort():
            c = int(D[r].argmin()); oid = oids[r]
            if r in used_r or c in used_c: continue
            if D[r, c] > cfg.centroid_match_px or bbox_iou(self.boxes[oid], boxes[c]) < cfg.iou_match_thresh:
                continue
            self.cents[oid], self.boxes[oid], self.names[oid] = cents[c], boxes[c], names[c]
            self.lost[oid] = 0
            self._update_speed(oid, boxes[c][2], t)
            used_r.add(r); used_c.add(c)
        for r, oid in enumerate(oids):
            if r not in used_r:
                self.lost[oid] += 1
                if self.lost[oid] > cfg.history_len:
                    for d in (self.cents, self.boxes, self.names,
                              self.lost, self.last_d, self.last_t, self.speed):
                        d.pop(oid, None)
        for c in range(len(cents)):
            if c not in used_c: self._add(cents[c], boxes[c], names[c], t)

class MotionDetector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.bg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
    def __call__(self, frame):
        fg = self.bg.apply(frame)
        fg = cv2.medianBlur(fg, 5)
        _, fg = cv2.threshold(fg, 127, 255, cv2.THRESH_BINARY)
        fg = cv2.dilate(fg, None, iterations=2)
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes, cents, names = [], [], []
        for cnt in contours:
            if cv2.contourArea(cnt) < self.cfg.min_motion_area: continue
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append([x, y, w, h])
            cents.append((x + w // 2, y + h // 2))
            names.append("obj")
        return cents, boxes, names

def main(cfg: Config):
    os.makedirs("CapturedImages/overspeed/left",  exist_ok=True)
    os.makedirs("CapturedImages/overspeed/right", exist_ok=True)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): sys.exit(f"could not open camera {cfg.camera_index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)
    detect = MotionDetector(cfg)
    tracker = Tracker(cfg)
    last_beep, state = 0.0, "green"
    colours = {"green": (0,255,0), "yellow": (0,255,255), "red": (0,0,255)}
    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.resize(frame, (cfg.frame_width, cfg.frame_height))
        tnow = time.time()
        cents, boxes, names = detect(frame)
        tracker.update(cents, boxes, names, tnow)
        dist = {oid: estimate_distance(tracker.boxes[oid][2], tracker.names[oid], cfg)
                for oid in tracker.cents}
        for oid in list(tracker.cents):
            if tracker.speed[oid] > cfg.speed_limit_kmh:
                side = "left" if tracker.cents[oid][0] < cfg.cx else "right"
                save_cropped_vehicle(frame, tracker.boxes[oid], side)
        left, right = set(), set()
        for oid, (x, _) in tracker.cents.items():
            (left if x < cfg.cx else right).add(oid)
        near_left = min([dist[i] for i in left if not math.isinf(dist[i])], default=None)
        near_right = min([dist[i] for i in right if not math.isinf(dist[i])], default=None)
        gap = (near_left + near_right + cfg.curve_length_m) if (near_left and near_right) else None
        if not left and not right: state = "green"
        elif gap and gap < cfg.min_safe_gap_m: state = "red"
        else: state = "yellow"
        if state == "red" and tnow - last_beep > cfg.beep_cooldown:
            beep(cfg); last_beep = tnow
        cv2.line(frame, (cfg.cx,0), (cfg.cx,cfg.frame_height), (200,200,200), 1)
        for oid, box in tracker.boxes.items():
            x,y,w,h = box
            col = (255,0,0)
            spd = tracker.speed[oid]
            cv2.rectangle(frame, (x,y), (x+w,y+h), col, 2)
            cv2.putText(frame, f"{oid}:{spd:.0f}k {dist[oid]:.1f}m", (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        if gap:
            cv2.putText(frame, f"gap {gap:.1f} m", (20, cfg.frame_height-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0,255,255) if state!="red" else (0,0,255), 2)
        cv2.rectangle(frame, (10,10), (150,60), colours[state], -1)
        cv2.putText(frame, state.upper(), (20,45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 3)
        cv2.imshow("curve‑road traffic monitor", frame)
        if cv2.waitKey(1) & 0xFF == 27: break
    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main(Config())
