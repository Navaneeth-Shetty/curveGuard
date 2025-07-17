import cv2
import numpy as np

cap = cv2.VideoCapture(1)
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

next_id = 0
trackers = {}  # id: centroid
boxes = {}     # id: bbox
cx_line = 640  # center vertical line (curve reference)

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))
    fg = fgbg.apply(frame)
    fg = cv2.medianBlur(fg, 5)
    _, fg = cv2.threshold(fg, 127, 255, cv2.THRESH_BINARY)
    fg = cv2.dilate(fg, None, iterations=2)

    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    current = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 1500:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w // 2, y + h // 2
        current.append(((cx, cy), (x, y, w, h)))

    updated = {}
    new_boxes = {}
    for (cx, cy), box in current:
        found = False
        for oid, old_cent in trackers.items():
            if euclidean((cx, cy), old_cent) < 100:
                updated[oid] = (cx, cy)
                new_boxes[oid] = box
                found = True
                break
        if not found:
            updated[next_id] = (cx, cy)
            new_boxes[next_id] = box
            next_id += 1

    trackers = updated
    boxes = new_boxes

    # classify as left or right of curve
    left_ids = [oid for oid, (cx, _) in trackers.items() if cx < cx_line]
    right_ids = [oid for oid, (cx, _) in trackers.items() if cx >= cx_line]

    # find nearest left and right (based on y position)
    left_near = min(left_ids, key=lambda i: trackers[i][1], default=None)
    right_near = min(right_ids, key=lambda i: trackers[i][1], default=None)

    # draw vertical curve center line
    cv2.line(frame, (cx_line, 0), (cx_line, 720), (200, 200, 200), 2)

    # draw bounding boxes and ids
    for oid, (cx, cy) in trackers.items():
        x, y, w, h = boxes[oid]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.circle(frame, (cx, cy), 6, (0, 255, 255), -1)
        cv2.putText(frame, f'ID {oid}', (cx - 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # draw gap line if both vehicles are found
    if left_near is not None and right_near is not None:
        pt1 = trackers[left_near]
        pt2 = trackers[right_near]
        cv2.line(frame, pt1, pt2, (0, 0, 255), 2)
        gap = euclidean(pt1, pt2)
        mid = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
        cv2.putText(frame, f'gap: {gap:.1f}px', mid, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Curve Line + Tracking + Gap", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
