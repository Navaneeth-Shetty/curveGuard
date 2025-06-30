import cv2, os, time
from datetime import datetime

cam_idx      = 1            # virtual cam1
interval_s   = 1.0          # seconds between saves
save_dir     = "dataset/truck"
os.makedirs(save_dir, exist_ok=True)

cap   = cv2.VideoCapture(cam_idx)
last  = time.time()

def stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

while True:
    ok, frame = cap.read()
    if not ok:
        print("camera feed unavailable"); break

    now = time.time()
    if now - last >= interval_s:
        path = os.path.join(save_dir, f"{stamp()}.png")
        while os.path.exists(path):                      # just in case
            path = os.path.join(save_dir, f"{stamp()}.png")
        cv2.imwrite(path, frame)
        last = now

    cv2.imshow("cam1 preview", frame)
    if cv2.waitKey(1) & 0xff == 27: break               # esc

cap.release()
cv2.destroyAllWindows()
