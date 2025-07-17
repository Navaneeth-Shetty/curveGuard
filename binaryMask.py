import cv2

cap = cv2.VideoCapture(1)  # or your stream URL
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))
    fg = fgbg.apply(frame)  # raw foreground mask

    fg = cv2.medianBlur(fg, 5)  # remove salt-and-pepper noise
    _, fg = cv2.threshold(fg, 127, 255, cv2.THRESH_BINARY)  # binarize
    fg = cv2.dilate(fg, None, iterations=2)  # close small gaps

    cv2.imshow('Original Frame', frame)
    cv2.imshow('Processed Foreground Mask', fg)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
