import cv2

cap = cv2.VideoCapture(1)  # or your video stream/url
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))
    fgmask = fgbg.apply(frame)

    cv2.imshow('Original Frame', frame)
    cv2.imshow('Foreground Mask', fgmask)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
