import cv2, numpy as np, time, math
from dataclasses import dataclass, field
from ultralytics import YOLO

try:
    import winsound
    CAN_BEEP = True
except ImportError:
    CAN_BEEP = False

@dataclass
class Config:
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    focal_length_px: int = 800
    curve_length_m: float = 3.0
    caution_zone_px: int = 250
    min_safe_gap_m: float = 40.0
    history_len: int = 5
    centroid_match_px: int = 75
    iou_match_thresh: float = 0.5
    beep_freq: int = 1000
    beep_dur: int = 200
    beep_cooldown: float = 2.0
    vehicle_widths: dict[str, float] = field(default_factory=lambda: {
        "car": 1.8, "truck": 2.5, "bus": 2.5, "motorbike": 0.7, "obj": 1.8
    })

    def __post_init__(self):
        self.cx = self.frame_width // 2

    @property
    def vehicle_classes(self):
        return {"car", "truck", "bus", "motorbike"}

def estimate_distance(width_px, cls, cfg):
    real_width = cfg.vehicle_widths.get(cls, 1.8)
    return float("inf") if width_px == 0 else (real_width * cfg.focal_length_px) / width_px

def estimate_focal_length(frame, known_distance_m=5.0, known_width_m=1.8, class_name="car"):
    model = YOLO("yolov8n.pt")
    result = model(frame, verbose=False)[0]
    for box in result.boxes:
        if result.names[int(box.cls[0])] == class_name:
            x1, _, x2, _ = map(int, box.xyxy[0])
            px_width = x2 - x1
            if px_width > 0:
                return int((px_width * known_distance_m) / known_width_m)
    return None

def bbox_iou(a, b):
    xa, ya, wa, ha = a
    xb, yb, wb, hb = b
    xa2, ya2, xb2, yb2 = xa + wa, ya + ha, xb + wb, yb + hb
    inter_x1, inter_y1 = max(xa, xb), max(ya, yb)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter_area, union = iw * ih, wa * ha + wb * hb - iw * ih
    return inter_area / union if union else 0

def beep(cfg):
    winsound.Beep(cfg.beep_freq, cfg.beep_dur) if CAN_BEEP else print("\a", end="", flush=True)

class Tracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.cents, self.boxes, self.names, self.lost, self.last_positions = {}, {}, {}, {}, {}
        self.next_id = 0

    def _pairwise(self, a, b):
        return np.linalg.norm(a[:, None] - b[None, :], axis=2)

    def update(self, cents, boxes, names):
        cfg = self.cfg
        if not cents:
            for oid in list(self.lost):
                self.lost[oid] += 1
                if self.lost[oid] > cfg.history_len:
                    for d in (self.cents, self.boxes, self.names, self.lost, self.last_positions):
                        d.pop(oid, None)
            return
        if not self.cents:
            for c, b, n in zip(cents, boxes, names):
                self._add(c, b, n)
            return
        oids = list(self.cents)
        D = self._pairwise(np.array([self.cents[i] for i in oids]), np.array(cents))
        used_r, used_c = set(), set()
        for r in D.min(1).argsort():
            c = int(D[r].argmin())
            oid = oids[r]
            if r in used_r or c in used_c:
                continue
            if D[r, c] > cfg.centroid_match_px or bbox_iou(self.boxes[oid], boxes[c]) < cfg.iou_match_thresh:
                continue
            self.last_positions[oid] = self.cents[oid]
            self.cents[oid], self.boxes[oid], self.names[oid] = cents[c], boxes[c], names[c]
            self.lost[oid] = 0
            used_r.add(r)
            used_c.add(c)
        for r, oid in enumerate(oids):
            if r not in used_r:
                self.lost[oid] += 1
                if self.lost[oid] > cfg.history_len:
                    for d in (self.cents, self.boxes, self.names, self.lost, self.last_positions):
                        d.pop(oid, None)
        for c in range(len(cents)):
            if c not in used_c:
                self._add(cents[c], boxes[c], names[c])

    def _add(self, c, b, n):
        self.cents[self.next_id] = c
        self.boxes[self.next_id] = b
        self.names[self.next_id] = n
        self.lost[self.next_id] = 0
        self.last_positions[self.next_id] = c
        self.next_id += 1

class YOLOv8Detector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = YOLO("yolov8n.pt")

    def __call__(self, frame):
        result = self.model(frame, verbose=False)[0]
        boxes, cents, names = [], [], []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            cls_name = result.names[cls_id]
            if cls_name not in self.cfg.vehicle_classes:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            boxes.append([x1, y1, w, h])
            cents.append((x1 + w // 2, y1 + h // 2))
            names.append(cls_name)
        return cents, boxes, names

def main(cfg):
    source = 0  # change to 'videos/demo2.mp4' or 1 as needed
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)

    # auto-calculate focal length if new device
    if not (isinstance(source, int) and source == 1) and not (isinstance(source, str) and source.startswith("videos/")):
        ret, frame = cap.read()
        if ret:
            focal = estimate_focal_length(frame)
            if focal:
                cfg.focal_length_px = focal
                print(f"[INFO] Focal length set to: {focal}px")

    detector = YOLOv8Detector(cfg)
    tracker = Tracker(cfg)
    last_beep_time = 0
    state = "green"
    colors = {"green": (0, 255, 0), "yellow": (0, 255, 255), "red": (0, 0, 255)}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (cfg.frame_width, cfg.frame_height))

        raw_cents, raw_boxes, raw_names = detector(frame)
        filtered = []
        for i, box_i in enumerate(raw_boxes):
            keep = True
            for j, box_j in enumerate(raw_boxes):
                if i != j and raw_names[i] == raw_names[j]:
                    if bbox_iou(box_i, box_j) > 0.6:
                        if i > j:
                            keep = False
                            break
            if keep:
                filtered.append((raw_cents[i], raw_boxes[i], raw_names[i]))

        cents, boxes, names = zip(*filtered) if filtered else ([], [], [])
        tracker.update(cents, boxes, names)

        distances = {
            oid: estimate_distance(tracker.boxes[oid][2], tracker.names[oid], cfg)
            for oid in tracker.cents
        }

        left_ids = {oid for oid, (x, _) in tracker.cents.items() if x < cfg.cx}
        right_ids = set(tracker.cents.keys()) - left_ids

        near_left = min([distances[i] for i in left_ids], default=None)
        near_right = min([distances[i] for i in right_ids], default=None)

        gap = (near_left + near_right + cfg.curve_length_m
               if near_left is not None and near_right is not None else None)

        moving_away = False
        if len(left_ids) == 1 and len(right_ids) == 1:
            l_id = list(left_ids)[0]
            r_id = list(right_ids)[0]
            lx_prev, _ = tracker.last_positions.get(l_id, tracker.cents[l_id])
            rx_prev, _ = tracker.last_positions.get(r_id, tracker.cents[r_id])
            lx_now, _ = tracker.cents[l_id]
            rx_now, _ = tracker.cents[r_id]
            moving_away = (lx_now < lx_prev) and (rx_now > rx_prev)

        if len(left_ids | right_ids) == 1:
            state = "green"
        elif gap is not None and gap < cfg.min_safe_gap_m and not moving_away:
            state = "red"
        else:
            state = "yellow"

        if state == "red" and time.time() - last_beep_time > cfg.beep_cooldown:
            beep(cfg)
            last_beep_time = time.time()

        for oid in tracker.cents:
            x, y, w, h = tracker.boxes[oid]
            dist = distances[oid]
            col = (255, 0, 0) if oid in left_ids else (0, 255, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), col, 2)
            cv2.putText(frame, f"{tracker.names[oid]}-{oid} {dist:.1f}m", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

        if gap:
            cv2.putText(frame, f"Gap: {gap:.1f} m", (20, cfg.frame_height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.line(frame, (cfg.cx, 0), (cfg.cx, cfg.frame_height), (200, 200, 200), 1)

        if left_ids or right_ids:
            cv2.rectangle(frame, (10, 10), (150, 60), colors[state], -1)
            cv2.putText(frame, state.upper(), (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

        cv2.imshow("CurveGuard - YOLOv8", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(Config())
