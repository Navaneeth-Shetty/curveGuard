import cv2, numpy as np, time

# ------------- constants & settings -------------
CFG, WTS, NAMES = "yolov3/yolov3.cfg", "yolov3/yolov3.weights", "yolov3/coco.names"
FW, FH = 1280, 720
CURVE_CX, CURVE_RAD = FW // 2, 150
METERS_PER_PIXEL = 0.05
DANGER, CAUTION = 20, 30
VEHICLE_CLASSES = {"car", "bus", "truck", "motorbike", "bicycle"}

# ------------- load YOLO -------------
net = cv2.dnn.readNet(WTS, CFG)
ln = net.getUnconnectedOutLayersNames()
with open(NAMES) as f:
    NMS_LIST = [x.strip() for x in f]

# ------------- detection -------------
def detect(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True)
    net.setInput(blob)
    outs = net.forward(ln)

    boxes, centers, class_ids = [], [], []
    for out in outs:
        for det in out:
            scores = det[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.5 and NMS_LIST[class_id] in VEHICLE_CLASSES:
                cx, cy, bw, bh = (det[0:4] * np.array([w, h, w, h])).astype(int)
                x = int(cx - bw/2)
                y = int(cy - bh/2)
                boxes.append([x, y, bw, bh])
                centers.append((cx, cy))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, [0.9]*len(boxes), 0.5, 0.4)
    if len(indices) > 0:
        indices = indices.flatten()
    else:
        return [], []

    return [centers[i] for i in indices], [boxes[i] for i in indices]

# ------------- tracker -------------
class Tracker:
    def __init__(self, max_lost=10):
        self.centroids = {}
        self.boxes = {}
        self.lost = {}
        self.next_id = 0
        self.max_lost = max_lost

    def update(self, new_centroids, new_boxes):
        if not new_centroids:
            for i in list(self.lost):
                self.lost[i] += 1
                if self.lost[i] > self.max_lost:
                    self.centroids.pop(i, None)
                    self.boxes.pop(i, None)
                    self.lost.pop(i, None)
            return self.centroids, self.boxes

        if not self.centroids:
            for i, c in enumerate(new_centroids):
                self.centroids[self.next_id] = c
                self.boxes[self.next_id] = new_boxes[i]
                self.lost[self.next_id] = 0
                self.next_id += 1
            return self.centroids, self.boxes

        obj_ids = list(self.centroids.keys())
        obj_pts = np.array([self.centroids[i] for i in obj_ids])
        new_pts = np.array(new_centroids)

        if obj_pts.ndim != 2 or new_pts.ndim != 2:
            return self.centroids, self.boxes

        D = np.linalg.norm(obj_pts[:, None] - new_pts[None, :], axis=2)
        used_rows, used_cols = set(), set()

        for row in D.min(axis=1).argsort():
            col = D[row].argmin()
            if row in used_rows or col in used_cols or D[row, col] > 50:
                continue
            oid = obj_ids[row]
            self.centroids[oid] = new_centroids[col]
            self.boxes[oid] = new_boxes[col]
            self.lost[oid] = 0
            used_rows.add(row)
            used_cols.add(col)

        unused_ids = [obj_ids[r] for r in range(len(obj_ids)) if r not in used_rows]
        for oid in unused_ids:
            self.lost[oid] += 1
            if self.lost[oid] > self.max_lost:
                self.centroids.pop(oid, None)
                self.boxes.pop(oid, None)
                self.lost.pop(oid, None)

        for c in range(len(new_centroids)):
            if c not in used_cols:
                self.centroids[self.next_id] = new_centroids[c]
                self.boxes[self.next_id] = new_boxes[c]
                self.lost[self.next_id] = 0
                self.next_id += 1

        return self.centroids, self.boxes

# ------------- main -------------
def main(cam=0):
    cap = cv2.VideoCapture(1)
    tracker = Tracker()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (FW, FH))
        centers, boxes = detect(frame)
        objects, tracked_boxes = tracker.update(centers, boxes)

        # filter vehicles near the curve center
        near_curve_ids = [oid for oid, pt in objects.items() if abs(pt[0] - CURVE_CX) <= CURVE_RAD]

        min_gap = None
        min_pair = None
        for i in range(len(near_curve_ids)):
            for j in range(i+1, len(near_curve_ids)):
                a = objects[near_curve_ids[i]]
                b = objects[near_curve_ids[j]]
                d = np.linalg.norm(np.subtract(a, b)) * METERS_PER_PIXEL
                if min_gap is None or d < min_gap:
                    min_gap = d
                    min_pair = (a, b)

        # signal logic
        if min_gap is not None:
            if min_gap < DANGER:
                signal = "RED"
            elif min_gap < CAUTION:
                signal = "YELLOW"
            else:
                signal = "GREEN"
        else:
            signal = "GREEN"

        # draw visuals
        for oid, box in tracked_boxes.items():
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        if min_pair:
            cv2.line(frame, min_pair[0], min_pair[1], (0, 255, 255), 2)

        sig_color = {"GREEN": (0, 255, 0), "YELLOW": (0, 255, 255), "RED": (0, 0, 255)}[signal]
        cv2.rectangle(frame, (0, 0), (120, 40), sig_color, -1)
        cv2.putText(frame, signal, (10, 28), 0, .9, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"Gap: {min_gap:.1f} m" if min_gap else "Gap: N/A",
            (10, 70),
            0, .8, (255, 255, 255), 2, cv2.LINE_AA
        )

        cv2.imshow("curve monitor", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(0)
