import cv2

def detect_vehicles(frame, net, classes):
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416,416), swapRB=True, crop=False)
    net.setInput(blob)
    output = net.forward(net.getUnconnectedOutLayersNames())

    boxes = []
    class_ids = []
    for out in output:
        for detection in out:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] in ['car', 'bus', 'truck']:
                center_x, center_y, w, h = (detection[0:4] * [frame.shape[1], frame.shape[0]]*2).astype('int')
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                boxes.append((x, y, int(w), int(h), classes[class_id]))
    return boxes
