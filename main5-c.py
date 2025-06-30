#!/usr/bin/env python3
"""
smarter curve‑road safety ai
============================
this version implements a more intuitive and robust logic for accident prevention,
based on defined zones and vehicle presence on opposite sides of the curve.

• intelligent zone‑based alerts:
  – curve zone (green): the critical area of the curve itself.
  – caution zone (yellow): a new buffer area on either side of the curve. a
    vehicle entering this zone is considered "near" the curve.

• revised state logic:
  – RED   : triggered when vehicles are detected in the monitored zones (curve
            or caution) on *both the left and right sides* of the road. this is
            a clear conflict scenario. a beep sounds.
  – YELLOW: triggered when any vehicle is in the curve or caution zone, but there
            is no opposing vehicle. this is a general warning.
  – GREEN : the safe state. explicitly used if only one or zero vehicles are
            detected in total, preventing unnecessary alerts.

• enhanced visualization:
  – both the green curve zone and the new yellow caution zones are drawn on the
    frame, making the ai's logic clear.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time
import math
import sys
from pathlib import Path

import cv2
import numpy as np
try:
    import winsound
    _CAN_BEEP = True
except ImportError:
    _CAN_BEEP = False

from ultralytics import YOLO

# ─────────────────────────── fallback yolov3 paths ────────────────────────────
Y3_CFG = "yolov3/yolov3.cfg"
Y3_WTS = "yolov3/yolov3.weights"
Y3_NAMES = "yolov3/coco.names"

# ───────────────────────────── configuration ──────────────────────────────────
@dataclass
class Config:
    model_path: str = "yolov8n.pt"
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    curve_radius: int = 400
    caution_zone_width: int = 250 # NEW: Width of the yellow warning zone next to the curve
    conf_thresh: float = 0.5
    iou_thresh_nms: float = 0.45
    centroid_dist_thresh: int = 100
    iou_match_thresh: float = 0.30
    history_len: int = 5
    beep_freq: int = 1000
    beep_dur: int = 200
    beep_cooldown: float = 2.0
    fallback_frames: int = 60
    recovery_frames: int = 5
    heavy_classes: set[str] = field(default_factory=lambda: {"bus", "truck"})
    light_classes: set[str] = field(default_factory=lambda: {"car", "motorbike", "bicycle"})
    curve_cx: int = field(init=False)

    def __post_init__(self):
        self.curve_cx = self.frame_width // 2

    @property
    def vehicle_classes(self):
        return self.heavy_classes | self.light_classes

# ───────────────────────────── util helpers ───────────────────────────────────

def bbox_iou(a: list[int], b: list[int]) -> float:
    x1A, y1A, wA, hA = a; x2A, y2A = x1A + wA, y1A + hA
    x1B, y1B, wB, hB = b; x2B, y2B = x1B + wB, y1B + hB
    xA, yA = max(x1A, x1B), max(y1A, y1B)
    xB, yB = min(x2A, x2B), min(y2A, y2B)
    iw, ih = max(0, xB - xA), max(0, yB - yA)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = wA * hA + wB * hB - inter
    return inter / union

def beep(cfg: Config):
    if _CAN_BEEP:
        winsound.Beep(cfg.beep_freq, cfg.beep_dur)
    else:
        print("\a", end="", flush=True)

# ─────────────────────────── trackers & detectors ─────────────────────────────
class Tracker:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.centroids, self.boxes, self.names = {}, {}, {}
        self.lost, self.hist = {}, {}
        self.next_id = 0

    def _pair(self, a: np.ndarray, b: np.ndarray):
        return np.linalg.norm(a[:, None] - b[None, :], axis=2)

    def update(self, cents, boxes, names):
        cfg = self.cfg
        if not cents:
            for oid in list(self.lost):
                self.lost[oid] += 1
                if self.lost[oid] > cfg.history_len:
                    for d in (self.centroids, self.boxes, self.names, self.lost, self.hist):
                        d.pop(oid, None)
            return
        if not self.centroids:
            for c, b, n in zip(cents, boxes, names):
                self._add(c, b, n)
            return
        oids = list(self.centroids)
        D = self._pair(np.array([self.centroids[i] for i in oids]), np.array(cents))
        used_r, used_c = set(), set()
        for r in D.min(1).argsort():
            c = int(D[r].argmin()); oid = oids[r]
            if r in used_r or c in used_c: continue
            if D[r, c] > cfg.centroid_dist_thresh or bbox_iou(self.boxes[oid], boxes[c]) < cfg.iou_match_thresh:
                continue
            self._match(oid, cents[c], boxes[c], names[c])
            used_r.add(r); used_c.add(c)
        for r, oid in enumerate(oids):
            if r not in used_r:
                self.lost[oid] += 1
                if self.lost[oid] > cfg.history_len:
                    for d in (self.centroids, self.boxes, self.names, self.lost, self.hist):
                        d.pop(oid, None)
        for c in range(len(cents)):
            if c not in used_c:
                self._add(cents[c], boxes[c], names[c])

    def _add(self, c, b, n):
        self.centroids[self.next_id] = c
        self.boxes[self.next_id] = b
        self.names[self.next_id] = n
        self.lost[self.next_id] = 0
        self.hist[self.next_id] = [c]
        self.next_id += 1

    def _match(self, oid, c, b, n):
        self.centroids[oid] = c
        self.boxes[oid] = b
        self.names[oid] = n
        self.lost[oid] = 0
        self.hist[oid].append(c)
        self.hist[oid] = self.hist[oid][-self.cfg.history_len:]

class YoloV8Detector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = YOLO(cfg.model_path)
        self.names = self.model.names

    def __call__(self, frame):
        cfg = self.cfg
        res = self.model.predict(frame, imgsz=640, conf=cfg.conf_thresh, iou=cfg.iou_thresh_nms, verbose=False)[0]
        boxes, cents, names = [], [], []
        for xywh, cls in zip(res.boxes.xywh.cpu().numpy(), res.boxes.cls.cpu().numpy()):
            n = self.names[int(cls)]
            if n not in cfg.vehicle_classes: continue
            cx, cy, bw, bh = map(int, xywh)
            boxes.append([int(cx - bw/2), int(cy - bh/2), int(bw), int(bh)])
            cents.append((cx, cy)); names.append(n)
        return cents, boxes, names

class YoloV3Detector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.net = cv2.dnn.readNet(Y3_WTS, Y3_CFG)
        self.ln = self.net.getUnconnectedOutLayersNames()
        with open(Y3_NAMES) as f:
            self.names = [x.strip() for x in f]

    def __call__(self, frame):
        cfg = self.cfg
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True)
        self.net.setInput(blob)
        outs = self.net.forward(self.ln)
        boxes, cents, names, confs = [], [], [], []
        for out in outs:
            for det in out:
                scores = det[5:]; cid = int(np.argmax(scores)); conf = scores[cid]
                if conf < cfg.conf_thresh: continue
                name = self.names[cid]
                if name not in cfg.vehicle_classes: continue
                cx, cy, bw, bh = (det[0:4] * np.array([w, h, w, h])).astype(int)
                boxes.append([int(cx - bw/2), int(cy - bh/2), int(bw), int(bh)])
                cents.append((cx, cy)); names.append(name); confs.append(float(conf))
        idxs = cv2.dnn.NMSBoxes(boxes, confs, cfg.conf_thresh, cfg.iou_thresh_nms)
        if len(idxs):
            idxs = idxs.flatten()
            boxes = [boxes[i] for i in idxs]
            cents = [cents[i] for i in idxs]
            names = [names[i] for i in idxs]
        else:
            boxes, cents, names = [], [], []
        return cents, boxes, names

class CombinedDetector:
    def __init__(self, cfg: Config):
        self.primary = YoloV8Detector(cfg)
        self.secondary = YoloV3Detector(cfg)
        self.cfg = cfg
        self.use_primary = True
        self.primary_failure_streak = 0
        self.primary_recovery_streak = 0

    def __call__(self, frame):
        if self.use_primary:
            cents, boxes, names = self.primary(frame)
            if cents: self.primary_failure_streak = 0
            else: self.primary_failure_streak += 1
            if self.primary_failure_streak >= self.cfg.fallback_frames:
                print(f"INFO: YOLOv8 primary failed for {self.cfg.fallback_frames} frames. Switching to YOLOv3 fallback.", file=sys.stderr)
                self.use_primary, self.primary_failure_streak = False, 0
            return cents, boxes, names
        else:
            cents, boxes, names = self.secondary(frame)
            primary_cents, _, _ = self.primary(frame)
            if primary_cents: self.primary_recovery_streak += 1
            else: self.primary_recovery_streak = 0
            if self.primary_recovery_streak >= self.cfg.recovery_frames:
                print("INFO: YOLOv8 primary has recovered. Switching back.", file=sys.stderr)
                self.use_primary, self.primary_recovery_streak = True, 0
            return cents, boxes, names

# ───────────────────────────── main loop ──────────────────────────────────────

def main(cfg: Config):
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {cfg.camera_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)

    detect = CombinedDetector(cfg)
    tracker = Tracker(cfg)

    last_beep, state = 0.0, "GREEN"
    colours = {"GREEN": (0, 255, 0), "YELLOW": (0, 255, 255), "ORANGE": (0, 128, 255), "RED": (0, 0, 255)}

    while True:
        ok, frame = cap.read()
        if not ok: break

        frame = cv2.resize(frame, (cfg.frame_width, cfg.frame_height))
        cents, boxes, names = detect(frame)
        tracker.update(cents, boxes, names)

        # --- NEW Smarter Logic ---
        left_side_vehicles = set()
        right_side_vehicles = set()
        
        # Determine which vehicles are in the critical zones (caution + curve)
        for oid, (x, y) in tracker.centroids.items():
            dist_from_center = abs(x - cfg.curve_cx)
            # A vehicle is a concern if it's in the curve or the caution zone next to it
            if dist_from_center <= cfg.curve_radius + cfg.caution_zone_width:
                if x < cfg.curve_cx:
                    left_side_vehicles.add(oid)
                else:
                    right_side_vehicles.add(oid)

        # --- NEW State Machine ---
        # Rule 1 (Highest Priority): If only one car is on screen, it's always green.
        if len(tracker.centroids) <= 1:
            state = "GREEN"
        # Rule 2: If there are vehicles on BOTH sides, it's a conflict. RED ALERT.
        elif left_side_vehicles and right_side_vehicles:
            state = "RED"
        # Rule 3: If there are vehicles on EITHER side (but not both), it's a warning.
        elif left_side_vehicles or right_side_vehicles:
            state = "YELLOW"
        # Rule 4: Otherwise, no vehicles are near the curve. Safe.
        else:
            state = "GREEN"
        
        if state == "RED" and time.time() - last_beep > cfg.beep_cooldown:
            beep(cfg)
            last_beep = time.time()

        # --- Drawing Overlays ---
        # Draw vehicle boxes and labels
        for oid, box in tracker.boxes.items():
            x, y, w, h = box
            col = (255, 255, 255)
            if tracker.names[oid] in cfg.heavy_classes: col = (0, 0, 255)
            elif tracker.names[oid] in cfg.light_classes: col = (255, 0, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), col, 2)
            cv2.putText(frame, f"{oid} {tracker.names[oid]}", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

        # Draw the new Caution and Curve zones for clarity
        cx = cfg.curve_cx
        # Curve Zone (Green)
        cv2.rectangle(frame, (cx - cfg.curve_radius, 0), (cx + cfg.curve_radius, cfg.frame_height), (0, 255, 0), 2)
        # Caution Zone Left (Yellow)
        cv2.rectangle(frame, (cx - cfg.curve_radius - cfg.caution_zone_width, 0), (cx - cfg.curve_radius, cfg.frame_height), (0, 255, 255), 2)
        # Caution Zone Right (Yellow)
        cv2.rectangle(frame, (cx + cfg.curve_radius, 0), (cx + cfg.curve_radius + cfg.caution_zone_width, cfg.frame_height), (0, 255, 255), 2)

        # Draw status display
        cv2.rectangle(frame, (10, 10), (170, 60), colours[state], -1)
        cv2.putText(frame, state, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

        cv2.imshow("Smarter Curve Traffic Monitor", frame)
        if cv2.waitKey(1) & 0xFF == 27: # ESC key to exit
            break

    cap.release()
    cv2.destroyAllWindows()

# ───────────────────────────── entry point ────────────────────────────────────

if __name__ == "__main__":
    config = Config()
    main(config)
