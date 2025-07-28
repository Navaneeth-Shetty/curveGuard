#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass, field
import cv2, numpy as np, time, os, sys, datetime

try:
    import winsound
    _CAN_BEEP = True
except ImportError:
    _CAN_BEEP = False

# ────────── config ──────────
@dataclass
class Config:
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    conf_thresh: float = 0.3
    iou_thresh_nms: float = 0.45
    centroid_match_px: int = 110
    iou_match_thresh: float = 0.3
    history_len: int = 8
    focal_length_px: int = 800
    curve_length_m: float = 3.0
    min_safe_gap_m: float = 30.0
    speed_limit_kmh: float = 100.0
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
    blob_size: int = 320
    use_cuda: bool = False
    def __post_init__(self): self.cx = self.frame_width // 2

# ────────── helpers ──────────
def bbox_iou(a, b):
    x1a, y1a, wa, ha = a; x2a, y2a = x1a + wa, y1a + ha
    x1b, y1b, wb, hb = b; x2b, y2b = x1b + wb, y1b + hb
    xa, ya = max(x1a, x1b), max(y1a, y1b); xb, yb = min(x2a, x2b), min(y2a, y2b)
    iw, ih = max(0, xb - xa), max(0, yb - ya); inter = iw * ih
    return 0.0 if inter == 0 else inter / (wa * ha + wb * hb - inter)

def est_dist(px_width: int, cls: str, cfg: Config) -> float:
    real_w = cfg.vehicle_widths.get(cls, 1.8)
    return (real_w * cfg.focal_length_px) / px_width if px_width else float('inf')

def beep(cfg: Config):
    if _CAN_BEEP: winsound.Beep(1000, 200)
    else: print("\a", end="", flush=True)

def save_overspeed(frame, side: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = f"CapturedImages/overspeed/{side}"
    os.makedirs(path, exist_ok=True)
    cv2.imwrite(f"{path}/{ts}.jpg", frame)

# ────────── tracker ──────────
class Tracker:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cents, self.boxes, self.labels = {}, {}, {}
        self.lost, self.last_d, self.last_t, self.speed = {}, {}, {}, {}
        self.next_id = 0
    def _pair(self, a, b): return np.linalg.norm(a[:, None] - b[None, :], axis=2)
    def update(self, cents, boxes, labels, tnow):
        cfg = self.cfg
        if not cents:
            for oid in list(self.lost):
                self.lost[oid] += 1
                if self.lost[oid] > cfg.history_len:
                    for d in (self.cents, self.boxes, self.labels,
                              self.lost, self.last_d, self.last_t, self.speed):
                        d.pop(oid, None)
            return
        if not self.cents:
            for c, b, n in zip(cents, boxes, labels):
                self._add(c, b, n, tnow)
            return
        oids = list(self.cents)
        D = self._pair(np.array([self.cents[i] for i in oids]), np.array(cents))
        used_r, used_c = set(), set()
        for r in D.min(1).argsort():
            c = int(D[r].argmin()); oid = oids[r]
            if r in used_r or c in used_c: continue
            if D[r, c] > cfg.centroid_match_px or bbox_iou(self.boxes[oid], boxes[c]) < cfg.iou_match_thresh: continue
            self.cents[oid], self.boxes[oid], self.labels[oid] = cents[c], boxes[c], labels[c]
            self.lost[oid] = 0
            self._update_speed(oid, boxes[c][2], tnow)
            used_r.add(r); used_c.add(c)
        for r, oid in enumerate(oids):
            if r not in used_r:
                self.lost[oid] += 1
                if self.lost[oid] > cfg.history_len:
                    for d in (self.cents, self.boxes, self.labels,
                              self.lost, self.last_d, self.last_t, self.speed):
                        d.pop(oid, None)
        for c in range(len(cents)):
            if c not in used_c: self._add(cents[c], boxes[c], labels[c], tnow)
    def _add(self, c, b, n, tnow):
        self.cents[self.next_id] = c; self.boxes[self.next_id] = b
        self.labels[self.next_id] = n; self.lost[self.next_id] = 0
        self.last_d[self.next_id] = None; self.last_t[self.next_id] = tnow
        self.speed[self.next_id] = 0.0
        self.next_id += 1
    def _update_speed(self, oid, w_px, tnow):
        d = est_dist(w_px, self.labels[oid], self.cfg)
        prev_d, prev_t = self.last_d.get(oid), self.last_t.get(oid)
        self.last_d[oid], self.last_t[oid] = d, tnow
        if prev_d is None or tnow == prev_t: return
        self.speed[oid] = abs((prev_d - d) / (tnow - prev_t)) * 3.6

# ────────── detector ──────────
class YoloDetector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.net = cv2.dnn.readNetFromDarknet(cfg.yolo_cfg, cfg.yolo_weights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        if cfg.use_cuda:
            try: self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            except: self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        else:
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        ln = self.net.getLayerNames()
        self.out_layers = [ln[i - 1] for i in self.net.getUnconnectedOutLayers().flatten()]
        with open(cfg.yolo_names) as f: self.labels = [l.strip() for l in f]
    def __call__(self, frame):
        H, W = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0,
                                     (self.cfg.blob_size, self.cfg.blob_size),
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.out_layers)
        boxes, cents, names, confs = [], [], [], []
        for out in outs:
            for det in out:
                scores = det[5:]; cid = int(np.argmax(scores)); conf = scores[cid]
                if conf < self.cfg.conf_thresh: continue
                cx, cy, w, h = det[0:4] * np.array([W, H, W, H])
                x, y = int(cx - w/2), int(cy - h/2)
                boxes.append([x, y, int(w), int(h)])
                cents.append((int(cx), int(cy)))
                names.append(self.labels[cid]); confs.append(float(conf))
        if boxes:
            idxs = cv2.dnn.NMSBoxes(boxes, confs, self.cfg.conf_thresh, self.cfg.iou_thresh_nms)
            boxes = [boxes[i] for i in idxs.flatten()]
            cents = [cents[i] for i in idxs.flatten()]
            names = [names[i] for i in idxs.flatten()]
        return cents, boxes, names

# ────────── main ──────────
def main(cfg: Config):
    os.makedirs("CapturedImages/overspeed/left", exist_ok=True)
    os.makedirs("CapturedImages/overspeed/right", exist_ok=True)
    cap = cv2.VideoCapture('videos/demo2.mp4')     # use 0 for live camera
    if not cap.isOpened(): sys.exit("camera not found")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)
    detect, tracker = YoloDetector(cfg), Tracker(cfg)
    colours = {"green": (0,255,0), "yellow": (0,255,255), "red": (0,0,255)}
    state, last_beep, fps_t = "green", 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.resize(frame, (cfg.frame_width, cfg.frame_height))
        tnow = time.time()
        cents, boxes, names = detect(frame)
        tracker.update(cents, boxes, names, tnow)

        # overspeed capture
        for oid in list(tracker.cents):
            if tracker.speed[oid] > cfg.speed_limit_kmh:
                side = "left" if tracker.cents[oid][0] < cfg.cx else "right"
                save_overspeed(frame, side)

        # gap & traffic state
        left = {oid for oid in tracker.cents if tracker.cents[oid][0] < cfg.cx}
        right = set(tracker.cents) - left
        near_left = min([tracker.last_d[i] for i in left if tracker.last_d[i] is not None], default=None)
        near_right = min([tracker.last_d[i] for i in right if tracker.last_d[i] is not None], default=None)
        gap = near_left + near_right + cfg.curve_length_m if (near_left and near_right) else None
        state = "green" if not tracker.cents else ("red" if gap and gap < cfg.min_safe_gap_m else "yellow")
        if state == "red" and tnow - last_beep > 2: beep(cfg); last_beep = tnow

        # draw hud
        cv2.line(frame, (cfg.cx,0), (cfg.cx,cfg.frame_height), (200,200,200), 1)
        for oid in tracker.cents:
            x,y,w,h = tracker.boxes[oid]
            lbl = tracker.labels[oid]
            spd = tracker.speed[oid]
            dist = tracker.last_d[oid] if tracker.last_d[oid] is not None else 0
            heavy = cfg.class_map.get(lbl) == "heavy"
            col = (0,0,255) if heavy else (255,0,0)
            cv2.rectangle(frame, (x,y), (x+w,y+h), col, 2)
            cv2.putText(frame,
                        f"{oid}:{lbl[:3]} {spd:.0f}k {dist:.0f}m",
                        (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

        gap_txt = f"gap {gap:.1f} m" if gap else "gap –"
        cv2.putText(frame, gap_txt, (20, cfg.frame_height-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0,255,255) if state!="red" else (0,0,255), 2)
        cv2.rectangle(frame, (10,10), (130,55), colours[state], -1)
        cv2.putText(frame, state.upper(), (18,45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 3)

        # fps
        new = time.time(); fps = 1.0 / (new - fps_t); fps_t = new
        cv2.putText(frame, f"{fps:.1f} fps",
                    (cfg.frame_width-120, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        cv2.imshow("curve‑road safety", frame)
        if cv2.waitKey(1) & 0xFF == 27: break
    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main(Config())
