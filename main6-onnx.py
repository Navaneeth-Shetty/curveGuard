#!/usr/bin/env python3
# real-time curve‑road safety ai — yolov3 cpu-only

from __future__ import annotations
from dataclasses import dataclass, field
import time, sys, math, collections
import cv2, numpy as np

try:
    import winsound
    _CAN_BEEP = True
except ImportError:
    _CAN_BEEP = False

Y3_CFG   = "yolov3/yolov3.cfg"
Y3_WTS   = "yolov3/yolov3.weights"
Y3_NAMES = "yolov3/coco.names"

@dataclass
class Config:
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    curve_radius_px: int = 400
    caution_zone_px: int = 250
    conf_thresh: float = 0.5
    iou_thresh_nms: float = 0.45
    centroid_match_px: int = 100
    iou_match_thresh: float = 0.30
    history_len: int = 5
    focal_length_px: int = 800
    curve_length_m: float = 3.0
    min_safe_gap_m: float = 85.0
    min_box_w_px: int = 30
    smoothing_len: int = 7
    red_cons_frames: int = 3
    beep_freq: int = 1000
    beep_dur: int = 200
    beep_cooldown: float = 2.0
    heavy_classes: set[str] = field(default_factory=lambda: {"bus", "truck"})
    light_classes: set[str] = field(default_factory=lambda: {"car", "motorbike", "bicycle"})
    vehicle_widths: dict[str, float] = field(default_factory=lambda:{
        "bus":2.5,"truck":2.5,"car":1.8,"motorbike":0.8,"bicycle":0.6})

    def __post_init__(self): self.cx = self.frame_width//2
    @property
    def vehicle_classes(self): return self.heavy_classes|self.light_classes

def bbox_iou(a,b):
    x1a,y1a,wa,ha=a; x2a,y2a=x1a+wa,y1a+ha
    x1b,y1b,wb,hb=b; x2b,y2b=x1b+wb,y1b+hb
    xa,ya=max(x1a,x1b),max(y1a,y1b); xb,yb=min(x2a,x2b),min(y2a,y2b)
    iw,ih=max(0,xb-xa),max(0,yb-ya); inter=iw*ih
    return 0.0 if inter==0 else inter/(wa*ha+wb*hb-inter)

def estimate_distance(width_px:int,cls:str,cfg:Config)->float:
    real=cfg.vehicle_widths.get(cls,1.8)
    return float('inf') if width_px==0 else (real*cfg.focal_length_px)/width_px

def beep(cfg:Config):
    if _CAN_BEEP: winsound.Beep(cfg.beep_freq,cfg.beep_dur)
    else: print("\a",end="",flush=True)

class Tracker:
    def __init__(self,cfg:Config):
        self.cfg=cfg; self.cents={}; self.boxes={}; self.names={}; self.lost={}; self.next_id=0
    def _pair(self,a,b): return np.linalg.norm(a[:,None]-b[None,:],axis=2)
    def update(self,cents,boxes,names):
        cfg=self.cfg
        if not cents:
            for oid in list(self.lost):
                self.lost[oid]+=1
                if self.lost[oid]>cfg.history_len:
                    for d in(self.cents,self.boxes,self.names,self.lost): d.pop(oid,None)
            return
        if not self.cents:
            for c,b,n in zip(cents,boxes,names): self._add(c,b,n); return
        oids=list(self.cents)
        D=self._pair(np.array([self.cents[i] for i in oids]),np.array(cents))
        used_r,used_c=set(),set()
        for r in D.min(1).argsort():
            c=int(D[r].argmin()); oid=oids[r]
            if r in used_r or c in used_c: continue
            if D[r,c]>cfg.centroid_match_px or bbox_iou(self.boxes[oid],boxes[c])<cfg.iou_match_thresh: continue
            self.cents[oid],self.boxes[oid],self.names[oid]=cents[c],boxes[c],names[c]
            self.lost[oid]=0; used_r.add(r); used_c.add(c)
        for r,oid in enumerate(oids):
            if r not in used_r:
                self.lost[oid]+=1
                if self.lost[oid]>cfg.history_len:
                    for d in(self.cents,self.boxes,self.names,self.lost): d.pop(oid,None)
        for c in range(len(cents)):
            if c not in used_c: self._add(cents[c],boxes[c],names[c])
    def _add(self,c,b,n):
        self.cents[self.next_id]=c; self.boxes[self.next_id]=b; self.names[self.next_id]=n; self.lost[self.next_id]=0
        self.next_id+=1

class YoloV3Detector:
    def __init__(self,cfg:Config):
        self.cfg=cfg
        self.net=cv2.dnn.readNet(Y3_WTS,Y3_CFG)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.ln=self.net.getUnconnectedOutLayersNames()
        with open(Y3_NAMES) as f: self.names=[x.strip() for x in f]
    def __call__(self,frame):
        h,w=frame.shape[:2]
        blob=cv2.dnn.blobFromImage(frame,1/255,(416,416),swapRB=True)
        self.net.setInput(blob)
        outs=self.net.forward(self.ln)
        boxes,cents,names,confs=[],[],[],[]
        for out in outs:
            for det in out:
                scores=det[5:]; cid=int(np.argmax(scores)); conf=scores[cid]
                if conf<self.cfg.conf_thresh: continue
                n=self.names[cid]
                if n not in self.cfg.vehicle_classes: continue
                cx,cy,bw,bh=(det[0:4]*np.array([w,h,w,h])).astype(int)
                if bw<self.cfg.min_box_w_px: continue
                boxes.append([int(cx-bw/2),int(cy-bh/2),int(bw),int(bh)])
                cents.append((cx,cy)); names.append(n); confs.append(float(conf))
        idxs=cv2.dnn.NMSBoxes(boxes,confs,self.cfg.conf_thresh,self.cfg.iou_thresh_nms)
        if len(idxs):
            idxs=idxs.flatten()
            boxes=[boxes[i] for i in idxs]; cents=[cents[i] for i in idxs]; names=[names[i] for i in idxs]
        else:
            boxes,cents,names=[],[],[]
        return cents,boxes,names

def main(cfg:Config):
    cap=cv2.VideoCapture(1)
    if not cap.isOpened(): sys.exit(f"could not open camera {cfg.camera_index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,cfg.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,cfg.frame_height)
    detect,tracker=YoloV3Detector(cfg),Tracker(cfg)

    gap_hist=collections.deque(maxlen=cfg.smoothing_len)
    last_beep,state,red_frames=0.0,"green",0
    colours={"green":(0,255,0),"yellow":(0,255,255),"red":(0,0,255)}

    while True:
        ok,frame=cap.read()
        if not ok: break
        frame=cv2.resize(frame,(cfg.frame_width,cfg.frame_height))

        cents,boxes,names=detect(frame)
        tracker.update(cents,boxes,names)
        dist={oid:estimate_distance(tracker.boxes[oid][2],tracker.names[oid],cfg)
              for oid in tracker.cents}

        left,right=set(),set()
        for oid,(x,_) in tracker.cents.items():
            (left if x<cfg.cx else right).add(oid)

        valid_left=[dist[i] for i in left if dist[i]<200]
        valid_right=[dist[i] for i in right if dist[i]<200]

        avg_left=np.median(valid_left) if valid_left else None
        avg_right=np.median(valid_right) if valid_right else None

        gap=(avg_left+avg_right+cfg.curve_length_m) if avg_left and avg_right else None
        if gap: gap_hist.append(gap)
        smoothed_gap=np.median(gap_hist) if gap_hist else None

        if not valid_left and not valid_right:
            red_frames=0; state="green"
        elif smoothed_gap and smoothed_gap<cfg.min_safe_gap_m:
            red_frames+=1
            state="red" if red_frames>=cfg.red_cons_frames else "yellow"
        else:
            red_frames=0; state="yellow"

        if state=="red" and time.time()-last_beep>cfg.beep_cooldown:
            beep(cfg); last_beep=time.time()

        cv2.line(frame,(cfg.cx,0),(cfg.cx,cfg.frame_height),(200,200,200),1)
        for oid,box in tracker.boxes.items():
            x,y,w,h=box
            col=(0,0,255) if tracker.names[oid] in cfg.heavy_classes else (255,0,0)
            cv2.rectangle(frame,(x,y),(x+w,y+h),col,2)
            cv2.putText(frame,f"{oid}:{dist[oid]:.1f}m",(x,y-8),cv2.FONT_HERSHEY_SIMPLEX,0.6,col,2)
        if smoothed_gap:
            col=(0,0,255) if state=="red" else (0,255,255)
            cv2.putText(frame,f"gap {smoothed_gap:.1f} m",(20,cfg.frame_height-20),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,col,2)
        cv2.rectangle(frame,(10,10),(150,60),colours[state],-1)
        cv2.putText(frame,state.upper(),(20,45),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,0),3)
        cv2.imshow("curve‑road traffic monitor",frame)
        if cv2.waitKey(1)&0xFF==27: break

    cap.release(); cv2.destroyAllWindows()

if __name__=="__main__":
    main(Config())
