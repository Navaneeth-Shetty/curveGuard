#!/usr/bin/env python3
# birdeye curve‑road safety ai — pixel‑gap logic (no camera distance)

from __future__ import annotations
from dataclasses import dataclass, field
import time, sys, cv2, numpy as np

try:
    import winsound
    _CAN_BEEP = True
except ImportError:
    _CAN_BEEP = False

# ───────── configuration ─────────
@dataclass
class Config:
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    curve_radius_px: int = 400
    caution_zone_px: int = 250
    centroid_match_px: int = 100
    iou_match_thresh: float = 0.30
    history_len: int = 5
    min_motion_area: int = 1500
    min_safe_gap_px: int = 150        # pixel gap threshold
    beep_freq: int = 1000
    beep_dur: int = 200
    beep_cooldown: float = 2.0

    def __post_init__(self): self.cx = self.frame_width // 2

# ───────── helpers ─────────
def beep(cfg: Config):
    if _CAN_BEEP: winsound.Beep(cfg.beep_freq, cfg.beep_dur)
    else: print("\a", end="", flush=True)

# ───────── tracker ─────────
class Tracker:
    def __init__(self, cfg: Config):
        self.cfg, self.cents, self.boxes, self.lost, self.hist, self.next_id = cfg, {}, {}, {}, {}, 0
    def _pair(self, a, b): return np.linalg.norm(a[:, None] - b[None, :], axis=2)
    def update(self, cents, boxes):
        cfg = self.cfg
        if not cents:
            for oid in list(self.lost):
                self.lost[oid] += 1
                if self.lost[oid] > cfg.history_len:
                    for d in (self.cents, self.boxes, self.lost, self.hist): d.pop(oid, None)
            return
        if not self.cents:
            for c, b in zip(cents, boxes): self._add(c, b); return
        oids = list(self.cents)
        D = self._pair(np.array([self.cents[i] for i in oids]), np.array(cents))
        used_r, used_c = set(), set()
        for r in D.min(1).argsort():
            c = int(D[r].argmin()); oid = oids[r]
            if r in used_r or c in used_c or D[r, c] > cfg.centroid_match_px: continue
            self.hist.setdefault(oid, []).append(self.cents[oid])
            self.cents[oid], self.boxes[oid] = cents[c], boxes[c]
            self.lost[oid] = 0; used_r.add(r); used_c.add(c)
        for r, oid in enumerate(oids):
            if r not in used_r:
                self.lost[oid] += 1
                if self.lost[oid] > cfg.history_len:
                    for d in (self.cents, self.boxes, self.lost, self.hist): d.pop(oid, None)
        for c in range(len(cents)):
            if c not in used_c: self._add(cents[c], boxes[c])
    def _add(self, c, b):
        self.cents[self.next_id] = c; self.boxes[self.next_id] = b; self.lost[self.next_id] = 0; self.hist[self.next_id] = [c]
        self.next_id += 1

# ───────── simple motion detector ─────────
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
        boxes, cents = [], []
        for cnt in contours:
            if cv2.contourArea(cnt) < self.cfg.min_motion_area: continue
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append([x, y, w, h])
            cents.append((x + w // 2, y + h // 2))
        return cents, boxes

# ───────── main loop ─────────
def main(cfg: Config):
    cap = cv2.VideoCapture(1)
    if not cap.isOpened(): sys.exit(f"could not open camera {cfg.camera_index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)
    detect, tracker = MotionDetector(cfg), Tracker(cfg)
    last_beep, state = 0.0, "green"
    colours = {"green": (0, 255, 0), "yellow": (0, 255, 255), "red": (0, 0, 255)}

    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.resize(frame, (cfg.frame_width, cfg.frame_height))
        cents, boxes = detect(frame)
        tracker.update(cents, boxes)

        left_dist = right_dist = None
        for oid, (cx, cy) in tracker.cents.items():
            if cx < cfg.cx:
                d = cfg.cx - cx
                left_dist = d if left_dist is None or d < left_dist else left_dist
            else:
                d = cx - cfg.cx
                right_dist = d if right_dist is None or d < right_dist else right_dist

        gap = left_dist + right_dist if left_dist and right_dist else None
        if not tracker.cents: state = "green"
        elif gap and gap < cfg.min_safe_gap_px: state = "red"
        else: state = "yellow"
        if state == "red" and time.time() - last_beep > cfg.beep_cooldown:
            beep(cfg); last_beep = time.time()

        cv2.line(frame, (cfg.cx, 0), (cfg.cx, cfg.frame_height), (200, 200, 200), 1)
        for box in tracker.boxes.values():
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        if gap:
            col = (0, 0, 255) if state == "red" else (0, 255, 255)
            cv2.putText(frame, f"gap {gap:.0f} px", (20, cfg.frame_height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
        cv2.rectangle(frame, (10, 10), (150, 60), colours[state], -1)
        cv2.putText(frame, state.upper(), (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
        cv2.imshow("birdeye pixel‑gap monitor", frame)
        if cv2.waitKey(1) & 0xFF == 27: break
    cap.release(); cv2.destroyAllWindows()

# ───────── entry ─────────
if __name__ == "__main__":
    main(Config())
