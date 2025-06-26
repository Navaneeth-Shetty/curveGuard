import cv2, numpy as np, os, time, platform, threading, datetime, math
from priority_logic_b import get_signal_with_slope_priority, get_signal_no_slope

FOCAL_MM, SENSOR_MM, FRAME_W, CAM_H, CURVE_DEG = 6.0, 4.8, 1280, 5.0, 45
FL_PX = FOCAL_MM * FRAME_W / SENSOR_MM
CFG, WTS, NMS, SPEED_LIMIT = "yolov3/yolov3.cfg", "yolov3/yolov3.weights", "yolov3/coco.names", 30
net = cv2.dnn.readNet(WTS, CFG); net.setPreferableBackend(0); net.setPreferableTarget(0)
classes = [l.strip() for l in open(NMS)]; layers = net.getUnconnectedOutLayersNames()

def beep(stop):
    play = lambda: os.system("play -nq -t alsa synth 0.3 sine 1000 >/dev/null 2>&1")
    if platform.system()=="Windows":
        import winsound; play=lambda: winsound.Beep(1000,300)
    while not stop.is_set(): play(); time.sleep(0.2)

def est_dist(pw,rw=1.8,fl=FL_PX,ch=CAM_H,deg=CURVE_DEG):
    if pw<=0: return -1
    los=(rw*fl)/pw; flat=math.sqrt(max(0,los**2-ch**2))
    return round(flat/math.cos(math.radians(deg)),2)

def detect(f,prev,last):
    blob=cv2.dnn.blobFromImage(cv2.resize(f,(416,416)),1/255,(416,416),swapRB=True); net.setInput(blob)
    h,w=f.shape[:2]; best,maxa=None,0
    for out in net.forward(layers):
        for d in out:
            cid=np.argmax(d[5:]); conf=d[5+cid]
            if conf>0.5 and classes[cid] in["car","bus","truck"]:
                bw=int(d[2]*w); bh=int(d[3]*h); area=bw*bh
                if area>maxa: maxa, best=area,(d,cid)
    if not best: return"none",-1,0,time.time()
    d,cid=best; cx,cy,bw,bh=d[:4]; x=int((cx-bw/2)*w); y=int((cy-bh/2)*h)
    bwpx, bhpx=int(bw*w),int(bh*h); dist=est_dist(bwpx); now=time.time()
    speed=max(0,(prev-dist)/(now-last)*3.6) if dist!=-1 and prev!=-1 else 0
    cv2.rectangle(f,(x,y),(x+bwpx,y+bhpx),(0,255,0),2)
    cv2.putText(f,f"{classes[cid]} {dist:.1f}m",(x,y-10),0,0.6,(255,255,255),2)
    cv2.putText(f,f"{int(speed)} km/h",(x,y+bhpx+20),0,0.5,(200,200,255),2)
    return classes[cid],dist,speed,now

def cfg():
    if input("slope present? (y/n): ").lower()=="y":
        l=input("left up/down: "); r=input("right up/down: "); return True,l,r
    return False,"flat","flat"

if __name__=="__main__":
    slope,L,R=cfg(); cap=cv2.VideoCapture(1)
    if not cap.isOpened(): exit("feed error")
    sig,islock,lt=time.time(),False,time.time()
    holds={"green":5,"yellow":2,"red":5}; lastmsg=""
    prevL=prevR=-1; tL=tR=0; stop=threading.Event(); bthr=None
    disp={"green":("🟢 safe","safe",(0,255,0)),
          "yellow":("🟡 caution","caution",(0,255,255)),
          "red":("🔴 danger","danger",(0,0,255))}
    while True:
        ok,frame=cap.read(); 
        if not ok: break
        h,w=frame.shape[:2]; left,right=frame[:,:w//2].copy(),frame[:,w//2:].copy()
        labL,distL,spdL,tL=detect(left,prevL,tL); labR,distR,spdR,tR=detect(right,prevR,tR)
        prevL,prevR=distL,distR
        nearL,nearR=(0<distL<15),(0<distR<15)
        appL,appR=(distL!=-1 and prevL!=-1 and distL<prevL),(distR!=-1 and prevR!=-1 and distR<prevR)
        if not islock:
            sig,reason=(get_signal_with_slope_priority(labL,nearL,appL,labR,nearR,appR,L,R)
                         if slope else get_signal_no_slope(labL,nearL,appL,labR,nearR,appR))
            if time.time()-lt>holds[sig]: islock=False
            elif sig!="green": islock,lt=True,time.time()
        else:
            if time.time()-lt>holds[sig]: islock=False; reason=""
        if sig=="red" and (not bthr or not bthr.is_alive()):
            stop.clear(); bthr=threading.Thread(target=beep,args=(stop,)); bthr.start()
        elif sig!="red" and bthr and bthr.is_alive(): stop.set()
        txt,vid,col=disp[sig]
        if txt!=lastmsg: print(f"signal: {txt}"); lastmsg=txt
        combo=np.hstack((left,right))
        if distL>0 and distR>0:
            gap=round(distL+distR,2)
            cv2.putText(combo,f"gap: {gap} m",(20,100),0,0.8,(255,255,255),2)
        cv2.rectangle(combo,(10,10),(400,80),(0,0,0),-1)
        cv2.putText(combo,f"signal: {vid}",(20,55),0,1,col,2)
        cv2.imshow("cctv",combo)
        if cv2.waitKey(1)&0xFF==27: break
    stop.set(); cap.release(); cv2.destroyAllWindows()
