#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass, field
import time, sys, cv2, numpy as np

try:
    import winsound
    _CAN_BEEP = True
except ImportError:
    _CAN_BEEP = False

# ───────────── config ─────────────
@dataclass
class Config:
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    conf_thresh: float = 0.3
    iou_thresh_nms: float = 0.45
    centroid_match_px: int = 100
    iou_match_thresh: float = 0.30
    history_len: int = 5
    focal_length_px: int = 800
    curve_length_m: float = 3.0
    min_safe_gap_m: float = 30.0
    beep_freq: int = 1000
    beep_dur: int = 200
    beep_cooldown: float = 2.0
    yolo_cfg: str = "yolov3/yolov3-tiny.cfg"
    yolo_weights: str = "yolov3/yolov3-tiny.weights"
    yolo_names: str = "yolov3/coco.names"
    vehicle_widths: dict[str, float] = field(default_factory=lambda: {
        "car": 1.8, "motorbike": 0.8, "bus": 2.5, "truck": 2.5, "bicycle": 0.6
    })
    class_map: dict[str, str] = field(default_factory=lambda: {
        "car": "light", "motorbike": "light", "bicycle": "light",
        "bus": "heavy", "truck": "heavy"
    })
    def __post_init__(self): self.cx = self.frame_width // 2

# ───────────── helpers ─────────────
def bbox_iou(a, b):
    x1a, y1a, wa, ha = a; x2a, y2a = x1a + wa, y1a + ha
    x1b, y1b, wb, hb = b; x2b, y2b = x1b + wb, y1b + hb
    xa, ya = max(x1a, x1b), max(y1a, y1b); xb, yb = min(x2a, x2b), min(y2a, y2b)
    iw, ih = max(0, xb - xa), max(0, yb - ya); inter = iw * ih
    return 0.0 if inter == 0 else inter / (wa * ha + wb * hb - inter)

def estimate_distance(width_px: int, label: str, cfg: Config) -> float:
    real = cfg.vehicle_widths.get(label, 1.8)
    return float('inf') if width_px == 0 else (real * cfg.focal_length_px) / width_px

def beep(cfg: Config):
    if _CAN_BEEP: winsound.Beep(cfg.beep_freq, cfg.beep_dur)
    else: print("\a", end="", flush=True)

# ───────────── tracker ─────────────
class Tracker:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cents, self.boxes, self.names, self.lost = {}, {}, {}, {}
        self.next_id = 0
    def _pair(self, a, b): return np.linalg.norm(a[:, None] - b[None, :], axis=2)
    def update(self, cents, boxes, names):
        cfg = self.cfg
        if not cents:
            for oid in list(self.lost):
                self.lost[oid] += 1
                if self.lost[oid] > cfg.history_len:
                    for d in (self.cents, self.boxes, self.names, self.lost): d.pop(oid, None)
            return
        if not self.cents:
            for c, b, n in zip(cents, boxes, names): self._add(c, b, n)
            return
        oids = list(self.cents)
        D = self._pair(np.array([self.cents[i] for i in oids]), np.array(cents))
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
                    for d in (self.cents, self.boxes, self.names, self.lost): d.pop(oid, None)
        for c in range(len(cents)):
            if c not in used_c: self._add(cents[c], boxes[c], names[c])
    def _add(self, c, b, n):
        self.cents[self.next_id] = c; self.boxes[self.next_id] = b
        self.names[self.next_id] = n; self.lost[self.next_id] = 0
        self.next_id += 1

# ───────────── yolo detector ─────────────
class YoloDetector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.net = cv2.dnn.readNetFromDarknet(cfg.yolo_cfg, cfg.yolo_weights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        try: self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        except: self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        ln = self.net.getLayerNames()
        self.out_layers = [ln[i - 1] for i in self.net.getUnconnectedOutLayers().flatten()]
        with open(cfg.yolo_names) as f: self.labels = [l.strip() for l in f]
    def __call__(self, frame):
        H, W = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.out_layers)
        boxes, cents, names, confid = [], [], [], []
        for out in outs:
            for det in out:
                scores = det[5:]
                cls_id = int(np.argmax(scores)); conf = scores[cls_id]
                if conf < self.cfg.conf_thresh: continue
                cx, cy, w, h = det[0:4] * np.array([W, H, W, H])
                x, y = int(cx - w/2), int(cy - h/2)
                boxes.append([x, y, int(w), int(h)])
                cents.append((int(cx), int(cy)))
                names.append(self.labels[cls_id])
                confid.append(float(conf))
        if boxes:
            idxs = cv2.dnn.NMSBoxes(boxes, confid, self.cfg.conf_thresh, self.cfg.iou_thresh_nms)
            boxes = [boxes[i] for i in idxs.flatten()]
            cents = [cents[i] for i in idxs.flatten()]
            names = [names[i] for i in idxs.flatten()]
        return cents, boxes, names

# ───────────── main loop ─────────────
def main(cfg: Config):
    cap = cv2.VideoCapture(1)
    if not cap.isOpened(): sys.exit(f"camera {cfg.camera_index} not found")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)
    detect, tracker = YoloDetector(cfg), Tracker(cfg)
    last_beep, state = 0.0, "green"
    colours = {"green": (0,255,0), "yellow": (0,255,255), "red": (0,0,255)}
    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.resize(frame, (cfg.frame_width, cfg.frame_height))
        cents, boxes, names = detect(frame); tracker.update(cents, boxes, names)
        dist = {oid: estimate_distance(tracker.boxes[oid][2], tracker.names[oid], cfg)
                for oid in tracker.cents}

        left, right, heavy_left, heavy_right = set(), set(), False, False
        for oid, (x, _) in tracker.cents.items():
            cls = tracker.names[oid]
            side = left if x < cfg.cx else right
            side.add(oid)
            if cfg.class_map.get(cls) == "heavy":
                if x < cfg.cx: heavy_left = True
                else: heavy_right = True

        near_left = min([dist[i] for i in left], default=None)
        near_right = min([dist[i] for i in right], default=None)
        gap = (near_left + near_right + cfg.curve_length_m) if (near_left and near_right) else None

        if not left and not right: state = "green"
        elif gap and gap < cfg.min_safe_gap_m: state = "red"
        else: state = "yellow"
        if state == "red" and time.time() - last_beep > cfg.beep_cooldown:
            beep(cfg); last_beep = time.time()

        cv2.line(frame, (cfg.cx,0), (cfg.cx,cfg.frame_height), (200,200,200), 1)
        for oid, box in tracker.boxes.items():
            x,y,w,h = box
            cls = tracker.names[oid]
            is_heavy = cfg.class_map.get(cls) == "heavy"
            col = (0,0,255) if is_heavy else (255,0,0)
            cv2.rectangle(frame, (x,y), (x+w,y+h), col, 2)
            label = f"{oid}:{cls[:3]} {dist[oid]:.1f}m"
            cv2.putText(frame, label, (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

        if gap:
            gap_text = f"gap {gap:.1f} m"
        else:
            gap_text = "gap –"
        cv2.putText(frame, gap_text, (20, cfg.frame_height-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0,255,255) if state!="red" else (0,0,255), 2)

        cv2.rectangle(frame, (10,10), (150,60), colours[state], -1)
        cv2.putText(frame, state.upper(), (20,45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 3)

        if heavy_left or heavy_right:
            hl_text = "heavy load approaching from "
            if heavy_left and heavy_right: hl_text += "both"
            elif heavy_left: hl_text += "left"
            elif heavy_right: hl_text += "right"
            cv2.putText(frame, hl_text, (cfg.cx - 200, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)

        cv2.imshow("curve‑road traffic monitor", frame)
        if cv2.waitKey(1) & 0xFF == 27: break
    cap.release(); cv2.destroyAllWindows()

# ───────────── entry ─────────────
if __name__ == "__main__":
    main(Config())
