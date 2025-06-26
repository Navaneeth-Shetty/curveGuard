import cv2
import numpy as np
import math
import time

# --- Camera and Scene Parameters (adjust as needed) ---
FOCAL_MM = 6.0         # focal length in mm
SENSOR_W_MM = 4.8      # sensor width in mm
FRAME_W = 1280         # frame width in pixels
FRAME_H = 720          # frame height in pixels
CAMERA_HEIGHT = 5.0    # camera height from road in meters
CURVE_DEG = 45         # curve angle in degrees (for demonstration)
REAL_LANE_WIDTH = 3.7  # meters

# --- Derived Camera Parameter ---
FL_PX = FOCAL_MM * FRAME_W / SENSOR_W_MM  # focal length in pixels

# --- YOLO Model Files (adjust paths as needed) ---
CFG = "yolov3/yolov3.cfg"
WTS = "yolov3/yolov3.weights"
NMS = "yolov3/coco.names"

# --- Load YOLO ---
net = cv2.dnn.readNet(WTS, CFG)
classes = [l.strip() for l in open(NMS)]
layers = net.getUnconnectedOutLayersNames()

# --- Lane Detection Helper ---
def detect_lane_center(frame):
    """Detect the road center curve using lane lines and fit a polynomial."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    edges = cv2.Canny(blur, 50, 150)
    mask = np.zeros_like(edges)
    h, w = edges.shape
    cv2.rectangle(mask, (0, int(h*0.5)), (w, h), 255, -1)
    roi = cv2.bitwise_and(edges, mask)
    lines = cv2.HoughLinesP(roi, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=150)
    lane_pts = []
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                lane_pts.append(((x1+x2)//2, (y1+y2)//2))
    if len(lane_pts) >= 5:
        lane_pts = np.array(lane_pts)
        # Fit a 2nd degree polynomial (y = a*x^2 + b*x + c)
        poly = np.polyfit(lane_pts[:,1], lane_pts[:,0], 2)
        return poly
    return None

def get_curve_center(poly, y):
    """Given a polynomial and y, return the x coordinate of the curve center."""
    if poly is None:
        return FRAME_W // 2
    return int(poly[0]*y**2 + poly[1]*y + poly[2])

# --- Vehicle Detection and Centroid ---
def detect_vehicles(frame):
    """Detect vehicles and return bounding boxes and centroids."""
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (416,416)), 1/255, (416,416), swapRB=True)
    net.setInput(blob)
    h, w = frame.shape[:2]
    boxes, centroids = [], []
    for out in net.forward(layers):
        for d in out:
            scores = d[5:]
            cid = np.argmax(scores)
            conf = scores[cid]
            if conf > 0.5 and classes[cid] in ["car", "bus", "truck"]:
                cx, cy, bw, bh = d[:4]
                x = int((cx - bw/2)*w)
                y = int((cy - bh/2)*h)
                bwpx, bhpx = int(bw*w), int(bh*h)
                boxes.append((x, y, bwpx, bhpx))
                centroids.append((int(x + bwpx/2), int(y + bhpx)))
    return boxes, centroids

# --- Project Centroid to Road Plane ---
def centroid_to_world(cx, cy, fl_px=FL_PX, cam_h=CAMERA_HEIGHT):
    """Project centroid to ground plane (assume flat road, camera facing forward)."""
    # Assume principal point at image center
    px, py = FRAME_W/2, FRAME_H/2
    # Vertical FoV projection
    y_cam = cam_h
    # Estimate forward distance Z using similar triangles
    Z = (fl_px * cam_h) / (cy - py) if (cy - py) != 0 else 1e-6
    # Estimate X (lateral) position
    X = (cx - px) * Z / fl_px
    return X, Z

# --- Project to Curve and Compute Arc Length ---
def project_to_curve(X, Z, poly):
    """Project a world point onto the road center curve and return arc length."""
    # For demonstration, treat curve as a 2D path in (X, Z)
    # Find the closest point on the curve to (X, Z)
    min_dist, min_s = float('inf'), 0
    for s in np.linspace(0, Z, num=100):
        y = int(FRAME_H - s * (FRAME_H/27))  # 27m visible, adjust as needed
        x_curve = get_curve_center(poly, y)
        x_world = (x_curve - FRAME_W/2) * s / FL_PX
        dist = math.hypot(X - x_world, Z - s)
        if dist < min_dist:
            min_dist, min_s = dist, s
    return min_s

# --- Main ---
cap = cv2.VideoCapture(1)  # Use your camera index or video file

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Detect lane/road center curve
    poly = detect_lane_center(frame)
    # 2. Detect vehicles and centroids
    boxes, centroids = detect_vehicles(frame)

    # 3. Project centroids to world and onto curve
    arc_positions = []
    for (cx, cy) in centroids:
        X, Z = centroid_to_world(cx, cy)
        s = project_to_curve(X, Z, poly)
        arc_positions.append(s)
        # Draw centroid and projection
        cv2.circle(frame, (cx, cy), 6, (0,255,0), -1)
        cv2.putText(frame, f"{s:.1f}m", (cx, cy-10), 0, 0.7, (0,255,0), 2)

    # 4. Compute and display gaps along the curve
    arc_positions.sort()
    for i in range(len(arc_positions)-1):
        gap = arc_positions[i+1] - arc_positions[i]
        if gap > 0:
            cv2.putText(frame, f"Gap: {gap:.2f}m", (50, 50+30*i), 0, 0.8, (255,255,255), 2)

    # 5. Draw lane center curve
    if poly is not None:
        for y in range(FRAME_H//2, FRAME_H, 10):
            x = get_curve_center(poly, y)
            cv2.circle(frame, (x, y), 2, (255,0,0), -1)

    cv2.imshow("Vehicle Distance Along Curve", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
