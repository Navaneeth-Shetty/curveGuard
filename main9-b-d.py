#!/usr/bin/env python3
# dual-camera curve-road safety with blind spot crossover detection

from __future__ import annotations
from dataclasses import dataclass, field
import time, sys, math, cv2, numpy as np
from collections import deque

@dataclass
class Config:
    frame_width: int = 640
    frame_height: int = 480
    caution_zone_px: int = 250
    centroid_match_px: int = 100
    iou_match_thresh: float = 0.3
    history_len: int = 5
    focal_length_px: int = 800
    blind_spot_m: float = 50.0
    crossover_window: float = 2.5
    min_motion_area: int = 1200
    vehicle_widths: dict[str, float] = field(default_factory=lambda: {"obj": 1.8})

def bbox_iou(a, b):
    x1a, y1a, wa, ha = a
    x2a, y2a = x1a + wa, y1a + ha
    x1b, y1b, wb, hb = b
    x2b, y2b = x1b + wb, y1b + hb
    xa, ya = max(x1a, x1b), max(y1a, y1b)
    xb, yb = min(x2a, x2b), min(y2a, y2b)
    iw, ih = max(0, xb - xa), max(0, yb - ya)
    inter = iw * ih
    return 0.0 if inter == 0 else inter / (wa * ha + wb * hb - inter)

def estimate_distance(width_px: int, cls: str, cfg: Config) -> float:
    real = cfg.vehicle_widths.get(cls, 1.8)
    return float('inf') if width_px == 0 else (real * cfg.focal_length_px) / width_px

class Tracker:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cents, self.boxes, self.names, self.lost = {}, {}, {}, {}
        self.next_id = 0

    def _pair(self, a, b):
        return np.linalg.norm(a[:, None] - b[None, :], axis=2)

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
            for c, b, n in zip(cents, boxes, names):
                self._add(c, b, n)
            return
        oids = list(self.cents)
        D = self._pair(np.array([self.cents[i] for i in oids]), np.array(cents))
        used_r, used_c = set(), set()
        for r in D.min(1).argsort():
            c = int(D[r].argmin())
            oid = oids[r]
            if r in used_r or c in used_c:
                continue
            if D[r, c] > cfg.centroid_match_px or bbox_iou(self.boxes[oid], boxes[c]) < cfg.iou_match_thresh:
                continue
            self.cents[oid], self.boxes[oid], self.names[oid] = cents[c], boxes[c], names[c]
            self.lost[oid] = 0
            used_r.add(r)
            used_c.add(c)
        for r, oid in enumerate(oids):
            if r not in used_r:
                self.lost[oid] += 1
                if self.lost[oid] > cfg.history_len:
                    for d in (self.cents, self.boxes, self.names, self.lost):
                        d.pop(oid, None)
        for c in range(len(cents)):
            if c not in used_c:
                self._add(cents[c], boxes[c], names[c])

    def _add(self, c, b, n):
        self.cents[self.next_id] = c
        self.boxes[self.next_id] = b
        self.names[self.next_id] = n
        self.lost[self.next_id] = 0
        self.next_id += 1

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
            area = cv2.contourArea(cnt)
            if area < self.cfg.min_motion_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append([x, y, w, h])
            cents.append((x + w // 2, y + h // 2))
            names.append("obj")
        return cents, boxes, names

def main(cfg: Config):
    capL = cv2.VideoCapture('videos/demo2.mp4')
    capR = cv2.VideoCapture('videos/demo2.mp4')
    for cap in (capL, capR):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)

    detectL, detectR = MotionDetector(cfg), MotionDetector(cfg)
    trackerL, trackerR = Tracker(cfg), Tracker(cfg)
    recent_left_to_right, recent_right_to_left = deque(), deque()
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.namedWindow("dual camera", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("dual camera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        retL, frameL = capL.read()
        retR, frameR = capR.read()
        if not retL or not retR:
            break

        centsL, boxesL, namesL = detectL(frameL)
        centsR, boxesR, namesR = detectR(frameR)
        trackerL.update(centsL, boxesL, namesL)
        trackerR.update(centsR, boxesR, namesR)

        now = time.time()

        def log_departures(tracker, queue):
            for oid in list(tracker.lost):
                if tracker.lost[oid] == 1:
                    box = tracker.boxes[oid]
                    dist = estimate_distance(box[2], "obj", cfg)
                    queue.append((now, dist))

        log_departures(trackerL, recent_left_to_right)
        log_departures(trackerR, recent_right_to_left)

        # cleanup expired
        recent_left_to_right = deque([(t, d) for t, d in recent_left_to_right if now - t < cfg.crossover_window])
        recent_right_to_left = deque([(t, d) for t, d in recent_right_to_left if now - t < cfg.crossover_window])

        def detect_crossover(queue, d):
            for t, prev_d in queue:
                if abs(prev_d - d) <= cfg.blind_spot_m:
                    return True
            return False

        for oid, box in trackerR.boxes.items():
            x, y, w, h = box
            dist = estimate_distance(w, "obj", cfg)
            if detect_crossover(recent_left_to_right, dist):
                cv2.putText(frameR, "← FROM LEFT", (x, y - 10), font, 0.7, (0, 0, 255), 2)
            cv2.rectangle(frameR, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frameR, f"{oid}:{dist:.1f}m", (x, y - 30), font, 0.6, (255, 255, 255), 2)

        for oid, box in trackerL.boxes.items():
            x, y, w, h = box
            dist = estimate_distance(w, "obj", cfg)
            if detect_crossover(recent_right_to_left, dist):
                cv2.putText(frameL, "FROM RIGHT →", (x, y - 10), font, 0.7, (0, 0, 255), 2)
            cv2.rectangle(frameL, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frameL, f"{oid}:{dist:.1f}m", (x, y - 30), font, 0.6, (255, 255, 255), 2)

        combined = np.hstack((frameL, frameR))
        cv2.imshow("dual camera", combined)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    capL.release()
    capR.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(Config())
