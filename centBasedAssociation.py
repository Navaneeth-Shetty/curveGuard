import cv2
import numpy as np

cap = cv2.VideoCapture(1)
tracker_history = {}

detector = cv2.createBackgroundSubtractorMOG2()
next_id = 0

def assign_id(centroid, existing, threshold=50):
    for oid, c in existing.items():
        if np.linalg.norm(np.array(centroid) - np.array(c[-1])) < threshold:
            return oid
    return None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    fgmask = detector.apply(frame)
    fgmask = cv2.medianBlur(fgmask, 5)
    _, binary = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)
    binary = cv2.dilate(binary, None, iterations=2)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    new_centroids = []

    for cnt in contours:
        if cv2.contourArea(cnt) < 1500:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w // 2, y + h // 2
        new_centroids.append(((x, y, w, h), (cx, cy)))

    seen_ids = {}
    for box, centroid in new_centroids:
        oid = assign_id(centroid, tracker_history)
        if oid is None:
            oid = next_id
            next_id += 1
            tracker_history[oid] = []
        tracker_history[oid].append(centroid)
        seen_ids[oid] = True
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for oid, path in tracker_history.items():
        for i in range(1, len(path)):
            cv2.line(frame, path[i - 1], path[i], (255, 0, 0), 2)
        if oid in seen_ids:
            cx, cy = path[-1]
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"id:{oid}", (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    cv2.imshow("Tracking Trail", frame)
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
