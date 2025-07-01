#!/usr/bin/env python3
"""
curve‑road safety ai — metre‑based gap logic
===========================================

• partitions view into left / right by centre line
• estimates vehicle‑to‑camera distance (metres) from bbox width
• adds fixed curve length to obtain inter‑vehicle gap
• traffic signal: green = clear, yellow = one‑side traffic or long gap, red = short gap
"""

from __future__ import annotations
from dataclasses import dataclass, field
import time, sys, math
import cv2, numpy as np

try:
    import winsound
    _CAN_BEEP = True
except ImportError:
    _CAN_BEEP = False

from ultralytics import YOLO

# ───────────────────────────── configuration ────────────────────────────────
@dataclass
class Config:
    model_path: str = "yolov8n.pt"
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    curve_radius_px: int = 400               # half curve width in pixels
    caution_zone_px: int = 250
    conf_thresh: float = 0.5
    iou_thresh_nms: float = 0.45
    centroid_match_px: int = 100
    iou_match_thresh: float = 0.30
    history_len: int = 5
    focal_length_px: int = 800               # pre‑calibrated focal length
    curve_length_m: float = 60.0             # physical curve length
    min_safe_gap_m: float = 85.0             # red if gap below this
    beep_freq: int = 1000
    beep_dur: int = 200
    beep_cooldown: float = 2.0

    heavy_classes: set[str] = field(default_factory=lambda: {"bus", "truck"})
    light_classes: set[str] = field(default_factory=lambda: {"car", "motorbike", "bicycle"})
    vehicle_widths: dict[str, float] = field(default_factory=lambda: {
        "bus": 2.5, "truck": 2.5,
        "car": 1.8, "motorbike": 0.8, "bicycle": 0.6
    })

    def __post_init__(self):
        self.cx = self.frame_width // 2

    @property
    def vehicle_classes(self): return self.heavy_classes | self.light_classes

# ──────────────────────────── utility helpers ───────────────────────────────
def bbox_iou(a, b):
    x1a, y1a, wa, ha = a; x2a, y2a = x1a + wa, y1a + ha
    x1b, y1b, wb, hb = b; x2b, y2b = x1b + wb, y1b + hb
    xa, ya = max(x1a, x1b), max(y1a, y1b)
    xb, yb = min(x2a, x2b), min(y2a, y2b)
    iw, ih = max(0, xb - xa), max(0, yb - ya)
    inter = iw * ih
    return 0.0 if inter == 0 else inter / (wa * ha + wb * hb - inter)

def estimate_distance(width_px: int, cls: str, cfg: Config) -> float:
    real_w = cfg.vehicle_widths.get(cls, 1.8)
    return float('inf') if width_px == 0 else (real_w * cfg.focal_length_px) / width_px

def beep(cfg: Config):
    if _CAN_BEEP: winsound.Beep(cfg.beep_freq, cfg.beep_dur)
    else:         print("\a", end="", flush=True)

# ───────────────────────────── tracker class ────────────────────────────────
class Tracker:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cents: dict[int, tuple[int, int]] = {}
        self.boxes: dict[int, list[int]] = {}
        self.names: dict[int, str] = {}
        self.lost: dict[int, int] = {}
        self.next_id = 0

    def _pair(self, a, b): return np.linalg.norm(a[:, None] - b[None, :], axis=2)

    def update(self, cents, boxes, names):
        cfg = self.cfg
        if not cents:
            for oid in list(self.lost):
                self.lost[oid] += 1
                if self.lost[oid] > cfg.history_len:
                    for d in (self.cents, self.boxes, self.names, self.lost):
                        d.pop(oid, None)
            return
        if not self.cents:
            for c, b, n in zip(cents, boxes, names): self._add(c, b, n); return
        oids = list(self.cents); D = self._pair(np.array([self.cents[i] for i in oids]), np.array(cents))
        used_r, used_c = set(), set()
        for r in D.min(1).argsort():
            c = int(D[r].argmin()); oid = oids[r]
            if r in used_r or c in used_c: continue
            if D[r, c] > cfg.centroid_match_px or bbox_iou(self.boxes[oid], boxes[c]) < cfg.iou_match_thresh: continue
            self.cents[oid], self.boxes[oid], self.names[oid] = cents[c], boxes[c], names[c]
            self.lost[oid] = 0; used_r.add(r); used_c.add(c)
        for r, oid in enumerate(oids):
            if r not in used_r:
                self.lost[oid] += 1
                if self.lost[oid] > cfg.history_len:
                    for d in (self.cents, self.boxes, self.names, self.lost):
                        d.pop(oid, None)
        for c in range(len(cents)):
            if c not in used_c: self._add(cents[c], boxes[c], names[c])

    def _add(self, c, b, n):
        self.cents[self.next_id] = c; self.boxes[self.next_id] = b
        self.names[self.next_id] = n; self.lost[self.next_id] = 0; self.next_id += 1

# ──────────────────────────── detector wrapper ───────────────────────────────
class YoloV8Detector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = YOLO(cfg.model_path)
        self.names = self.model.names

    def __call__(self, frame):
        res = self.model.predict(frame, imgsz=640, conf=self.cfg.conf_thresh,
                                 iou=self.cfg.iou_thresh_nms, verbose=False)[0]
        boxes, cents, names = [], [], []
        for xywh, cls in zip(res.boxes.xywh.cpu().numpy(), res.boxes.cls.cpu().numpy()):
            n = self.names[int(cls)]
            if n not in self.cfg.vehicle_classes: continue
            cx, cy, bw, bh = map(int, xywh)
            boxes.append([int(cx - bw / 2), int(cy - bh / 2), int(bw), int(bh)])
            cents.append((cx, cy)); names.append(n)
        return cents, boxes, names

# ─────────────────────────────── main loop ──────────────────────────────────
def main(cfg: Config):
    cap = cv2.VideoCapture(cfg.camera_index)
    if not cap.isOpened(): sys.exit(f"could not open camera {cfg.camera_index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)

    detect, tracker = YoloV8Detector(cfg), Tracker(cfg)
    last_beep, state = 0.0, "green"
    colours = {"green": (0,255,0), "yellow": (0,255,255), "red": (0,0,255)}

    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.resize(frame, (cfg.frame_width, cfg.frame_height))

        cents, boxes, names = detect(frame); tracker.update(cents, boxes, names)

        # distance per object
        dist = {oid: estimate_distance(tracker.boxes[oid][2], tracker.names[oid], cfg)
                for oid in tracker.cents}

        # side partitioning
        left, right = set(), set()
        for oid, (x, _) in tracker.cents.items():
            region = left if x < cfg.cx else right
            if abs(x - cfg.cx) < cfg.curve_radius_px + cfg.caution_zone_px: region.add(oid)

        nleft, nright = len(left), len(right)
        near_left  = min([dist[i] for i in left],  default=None)
        near_right = min([dist[i] for i in right], default=None)
        gap = (near_left + near_right + cfg.curve_length_m) if (near_left and near_right) else None

        # traffic signal logic
        if nleft == 0 and nright == 0: state = "green"
        elif gap and gap < cfg.min_safe_gap_m: state = "red"
        else: state = "yellow"

        if state == "red" and time.time() - last_beep > cfg.beep_cooldown:
            beep(cfg); last_beep = time.time()

        # drawing
        cv2.line(frame, (cfg.cx,0), (cfg.cx,cfg.frame_height), (200,200,200), 1)
        for oid, box in tracker.boxes.items():
            x,y,w,h = box
            col = (0,0,255) if tracker.names[oid] in cfg.heavy_classes else (255,0,0)
            cv2.rectangle(frame, (x,y), (x+w,y+h), col, 2)
            cv2.putText(frame, f"{oid}:{dist[oid]:.1f}m", (x, y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

        if gap: cv2.putText(frame, f"gap {gap:.1f} m",
                            (20, cfg.frame_height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0,255,255) if state!="red" else (0,0,255), 2)

        cv2.rectangle(frame, (10,10), (150,60), colours[state], -1)
        cv2.putText(frame, state.upper(), (20,45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 3)

        cv2.imshow("curve‑road traffic monitor", frame)
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release(); cv2.destroyAllWindows()

# ──────────────────────────── entry point ───────────────────────────────────
if __name__ == "__main__":
    main(Config())
