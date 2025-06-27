import cv2, numpy as np, time, winsound

# ---------- constants ----------
CFG, WTS, NAMES = "yolov3/yolov3.cfg", "yolov3/yolov3.weights", "yolov3/coco.names"
FW, FH, CURVE_RAD = 1280, 720, 500
CURVE_CX, METERS_PER_PIXEL = FW // 2, 0
DANGER, CAUTION = 30, 40
HEAVY_VEHICLES = {"bus", "truck"}
LIGHT_VEHICLES = {"car", "motorbike", "bicycle"}
VEHICLE_CLASSES = HEAVY_VEHICLES | LIGHT_VEHICLES
BEEP_FREQUENCY, BEEP_DURATION, BEEP_COOLDOWN = 1000, 200, 2
CENTROID_DISTANCE_THRESHOLD, IOU_THRESHOLD, HISTORY_LEN = 100, 0.3, 5

# ---------- iou ----------
def bbox_iou(a, b):
    x1A, y1A, wA, hA = a; x2A, y2A = x1A + wA, y1A + hA
    x1B, y1B, wB, hB = b; x2B, y2B = x1B + wB, y1B + hB
    xA, yA, xB, yB = max(x1A, x1B), max(y1A, y1B), min(x2A, x2B), min(y2A, y2B)
    iw, ih = max(0, xB - xA), max(0, yB - yA)
    inter = iw * ih
    if inter == 0: return 0.0
    union = wA * hA + wB * hB - inter
    return inter / union

# ---------- yolo ----------
net = cv2.dnn.readNet(WTS, CFG)
ln = net.getUnconnectedOutLayersNames()
with open(NAMES) as f: NMS_LIST = [x.strip() for x in f]

def detect(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416,416), swapRB=True)
    net.setInput(blob)
    outs = net.forward(ln)
    boxes, centers, ids, confs = [], [], [], []
    for out in outs:
        for det in out:
            scores = det[5:]; cid = np.argmax(scores); conf = scores[cid]
            if conf > .5 and NMS_LIST[cid] in VEHICLE_CLASSES:
                cx, cy, bw, bh = (det[0:4]*np.array([w,h,w,h])).astype(int)
                x, y = int(cx-bw/2), int(cy-bh/2)
                boxes.append([x,y,bw,bh]); centers.append((cx,cy))
                ids.append(cid); confs.append(float(conf))
    idxs = cv2.dnn.NMSBoxes(boxes, confs, .5, .45)  # slightly higher nms
    fc, fb, fn = [], [], []
    if len(idxs):
        for i in idxs.flatten():
            fc.append(centers[i]); fb.append(boxes[i]); fn.append(NMS_LIST[ids[i]])
    return fc, fb, fn

# ---------- tracker ----------
class Tracker:
    def __init__(s,max_lost=10):
        s.cents,s.boxes,s.names,s.lost,s.hist,s.next = {},{},{},{},{},0
        s.max_lost=max_lost
    def update(s,new_c,new_b,new_n):
        if not new_c:
            for i in list(s.lost):
                s.lost[i]+=1
                if s.lost[i]>s.max_lost:
                    for d in (s.cents,s.boxes,s.names,s.lost,s.hist): d.pop(i,None)
            return s.cents,s.boxes,s.names
        if not s.cents:
            for i,(c,b,n) in enumerate(zip(new_c,new_b,new_n)):
                s.cents[s.next]=c; s.boxes[s.next]=b; s.names[s.next]=n
                s.lost[s.next]=0; s.hist[s.next]=[c]; s.next+=1
            return s.cents,s.boxes,s.names
        oids=list(s.cents); obj_pts=np.array([s.cents[i] for i in oids])
        new_pts=np.array(new_c)
        D=np.linalg.norm(obj_pts[:,None]-new_pts[None,:],axis=2)
        used_r,used_c=set(),set()
        for r in D.min(1).argsort():
            c=int(D[r].argmin()); oid=oids[r]
            if r in used_r or c in used_c: continue
            if D[r,c]>CENTROID_DISTANCE_THRESHOLD or bbox_iou(s.boxes[oid],new_b[c])<IOU_THRESHOLD: continue
            s.cents[oid]=new_c[c]; s.boxes[oid]=new_b[c]; s.names[oid]=new_n[c]; s.lost[oid]=0
            s.hist[oid].append(new_c[c]); s.hist[oid]=s.hist[oid][-HISTORY_LEN:]
            used_r.add(r); used_c.add(c)
        for oid in set(oids)-{oids[r] for r in used_r}:
            s.lost[oid]+=1
            if s.lost[oid]>s.max_lost:
                for d in (s.cents,s.boxes,s.names,s.lost,s.hist): d.pop(oid,None)
        for c in range(len(new_c)):
            if c not in used_c:
                s.cents[s.next]=new_c[c]; s.boxes[s.next]=new_b[c]; s.names[s.next]=new_n[c]; s.lost[s.next]=0
                s.hist[s.next]=[new_c[c]]; s.next+=1
        return s.cents,s.boxes,s.names

def x_velocity(hist):
    if len(hist)<2: return 0.0
    return hist[-1][0]-hist[0][0]

# ---------- main ----------
def main():
    cap=cv2.VideoCapture('videos/demo2.mp4'); tracker=Tracker(); last_beep=0; signal="GREEN"
    while True:
        ok,frame=cap.read()
        if not ok: break
        frame=cv2.resize(frame,(FW,FH))
        c,b,n=detect(frame)
        objs,boxes,names=tracker.update(c,b,n)

        approaching={}
        for oid,pt in objs.items():
            vx=x_velocity(tracker.hist[oid])
            if pt[0]<CURVE_CX and vx>0: approaching[oid]=pt
            if pt[0]>CURVE_CX and vx<0: approaching[oid]=pt

        ids=list(approaching)
        min_gap=None; pair=None
        for i in range(len(ids)):
            for j in range(i+1,len(ids)):
                a,b=approaching[ids[i]],approaching[ids[j]]
                if (a[0]-CURVE_CX)*(b[0]-CURVE_CX)>0: continue  # same side
                d=np.linalg.norm(np.subtract(a,b))*METERS_PER_PIXEL
                if min_gap is None or d<min_gap: min_gap,pair=d,(a,b)
        if min_gap is not None:
            signal="RED" if min_gap<DANGER else "YELLOW" if min_gap<CAUTION else "GREEN"
        else: signal="GREEN"
        if signal=="RED" and time.time()-last_beep>BEEP_COOLDOWN:
            winsound.Beep(BEEP_FREQUENCY,BEEP_DURATION); last_beep=time.time()

        for oid,box in boxes.items():
            x,y,w,h=box
            col=(255,255,255)
            if names[oid] in HEAVY_VEHICLES: col=(0,0,255)
            elif names[oid] in LIGHT_VEHICLES: col=(255,0,0)
            cv2.rectangle(frame,(x,y),(x+w,y+h),col,2)
            cv2.putText(frame,f"{oid} {names[oid]}",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,.7,col,2)
        if pair: cv2.line(frame,pair[0],pair[1],(0,255,255),2)
        cv2.rectangle(frame,(CURVE_CX-CURVE_RAD,0),(CURVE_CX+CURVE_RAD,FH),(0,255,0),2)
        cv2.putText(frame,"curve zone",(CURVE_CX-CURVE_RAD,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,0),2)
        sig_col={"GREEN":(0,255,0),"YELLOW":(0,255,255),"RED":(0,0,255)}[signal]
        cv2.rectangle(frame,(0,0),(140,40),sig_col,-1)
        cv2.putText(frame,signal,(10,28),cv2.FONT_HERSHEY_SIMPLEX,.9,(0,0,0),2)
        gaptxt=f"gap: {min_gap:.1f} m" if min_gap else "gap: n/a"
        cv2.putText(frame,gaptxt,(10,70),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
        cv2.imshow("curve traffic monitor",frame)
        if cv2.waitKey(1)==27: break
    cap.release(); cv2.destroyAllWindows()

if __name__=="__main__": main()
