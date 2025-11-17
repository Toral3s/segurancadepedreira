# Monitor de zona da pedreira - detecta pessoas e veículos, calcula distância e dispara alerta

import os
import time
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from shapely.geometry import Point, Polygon
import pygame

#criei varias classes pra tentar diferenciar ao maximo (estou tentando achar como diferencia uma arvore)
VIDEO_SOURCE = 0
YOLO_MODEL = "yolov8n.pt"
CLASSES_WANTED = ["person", "car", "truck", "bus", "motorbike"]

GEOFENCES = {
    "ZonaSeguranca": [(200, 100), (800, 100), (800, 600), (200, 600)]
}

CALIBRATION = {
    "known_object_height_m": 1.7,
    "pixels_at_known_distance": 300,
    "known_distance_m": 10.0
}

DISTANCE_ALERT_THRESHOLD_M = 6.0
SIREN_FILE = os.path.join(os.path.dirname(__file__), "sirene.mp3")
LOG_CSV = os.path.join(os.path.dirname(__file__), "log_eventos.csv")
ALERT_FRAME_FOLDER = os.path.join(os.path.dirname(__file__), "frames_alerta")
os.makedirs(ALERT_FRAME_FOLDER, exist_ok=True)

MIN_CONTOUR_AREA = 400

#tentei no maximo pegar o mais preciso contagem desde altura, distancia e quantidade, mas é o maximo que consegui
def create_polygons(geofences_dict):
    polys = {name: Polygon(pts) for name, pts in geofences_dict.items()}
    return polys

def point_in_geofences(pt, polygons):
    p = Point(pt)
    return [name for name, poly in polygons.items() if poly.contains(p)]

def estimate_focal_length(known_height_m, pixels_at_known_distance, known_distance_m):
    if pixels_at_known_distance <= 0:
        return None
    return (pixels_at_known_distance * known_distance_m) / known_height_m

def estimate_distance_m(real_height_m, focal_length, observed_pixels):
    if observed_pixels <= 0 or focal_length is None:
        return None
    return (real_height_m * focal_length) / observed_pixels

class CentroidTracker:
    def __init__(self, max_disappeared=40, max_distance=120):
        self.nextObjectID = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        self.objects.pop(objectID, None)
        self.disappeared.pop(objectID, None)

    def update(self, rects):
        if len(rects) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.objects

        inputCentroids = np.array(rects)
        if len(self.objects) == 0:
            for c in inputCentroids:
                self.register(tuple(c))
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = np.linalg.norm(np.array(objectCentroids)[:, None] - inputCentroids[None, :], axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            usedRows, usedCols = set(), set()
            for r, c in zip(rows, cols):
                if r in usedRows or c in usedCols: continue
                if D[r, c] > self.max_distance: continue
                oid = objectIDs[r]
                self.objects[oid] = tuple(inputCentroids[c])
                self.disappeared[oid] = 0
                usedRows.add(r)
                usedCols.add(c)
            for r in set(range(D.shape[0])) - usedRows:
                oid = objectIDs[r]
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            for c in set(range(D.shape[1])) - usedCols:
                self.register(tuple(inputCentroids[c]))
        return self.objects


polygons = create_polygons(GEOFENCES)
model = YOLO(YOLO_MODEL)
ct = CentroidTracker()
focal = estimate_focal_length(CALIBRATION["known_object_height_m"],
                              CALIBRATION["pixels_at_known_distance"],
                              CALIBRATION["known_distance_m"])

if not os.path.exists(LOG_CSV):
    pd.DataFrame(columns=["timestamp","object_id","class","cx","cy","bbox_w","bbox_h","distance_m","geofence","event"]).to_csv(LOG_CSV,index=False)

pygame.mixer.init()
SIREN_AVAILABLE = os.path.exists(SIREN_FILE)
if SIREN_AVAILABLE:
    print("[INFO] Sirene pronta")
else:
    print("[WARN] Sirene não encontrada, alerta será visual")

cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print("Erro ao abrir vídeo"); exit(1)

W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
writer = None
OUTPUT_VIDEO = None
if OUTPUT_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (W,H))

id_to_class = {}
log_rows = []
alert_count = 0

print("[INFO] Monitoramento iniciado. Pressione 'q' para sair")

#as vezes da bug que as person não pegam a box
while True:
    ret, frame = cap.read()
    if not ret: break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.predict(source=frame_rgb, imgsz=640, conf=0.35, verbose=False)

    boxes, centroids, classes, confidences = [], [], [], []

    if len(results) > 0:
        r = results[0]
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names.get(cls_id,str(cls_id))
            if label not in CLASSES_WANTED: continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx, cy = int((x1+x2)/2), int((y1+y2)/2)
            boxes.append((x1,y1,x2,y2))
            centroids.append((cx,cy))
            classes.append(label)
            confidences.append(conf)

    objects = ct.update(centroids)
    alert_this_frame = False

    for oid, centroid in objects.items():
        matched_idx = None
        if len(centroids) > 0:
            dists = [np.linalg.norm(np.array(centroid)-np.array(c)) for c in centroids]
            idx = int(np.argmin(dists))
            if dists[idx] < 150: matched_idx = idx

        detected_class = id_to_class.get(oid,"unknown")
        bbox = None
        distance_m = None

        if matched_idx is not None:
            detected_class = classes[matched_idx]
            bbox = boxes[matched_idx]
            id_to_class[oid] = detected_class

        if bbox is not None:
            observed_pixels = bbox[3]-bbox[1]
            real_h = 1.7 if detected_class=="person" else (1.8 if detected_class=="car" else 3.0)
            distance_m = estimate_distance_m(real_h,focal,observed_pixels)

        zones = point_in_geofences(centroid, polygons)
        event = ""
        if zones: event += "ENTER_GEOFENCE:" + ",".join(zones) + ";"
        if distance_m and distance_m < DISTANCE_ALERT_THRESHOLD_M:
            event += f"DISTANCE_LOW:{distance_m:.1f}m;"

        if event:
            alert_count += 1
            alert_this_frame = True
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            if bbox is not None:
                x1,y1,x2,y2 = bbox
                frame_name = f"{ALERT_FRAME_FOLDER}/alert_{timestamp}_ID{oid}.jpg"
                cv2.imwrite(frame_name, frame)
            log_rows.append({"timestamp":time.strftime("%Y-%m-%d %H:%M:%S"),
                             "object_id":oid,"class":detected_class,"cx":centroid[0],
                             "cy":centroid[1],"bbox_w":bbox[2]-bbox[0] if bbox else None,
                             "bbox_h":bbox[3]-bbox[1] if bbox else None,"distance_m":distance_m,
                             "geofence":",".join(zones),"event":event})
            if SIREN_AVAILABLE:
                try: pygame.mixer.music.load(SIREN_FILE); pygame.mixer.music.play()
                except: pass

        color = (0,255,0) if not distance_m or distance_m>DISTANCE_ALERT_THRESHOLD_M else (0,0,255)
        if bbox: x1,y1,x2,y2 = bbox; cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
        cv2.putText(frame,f"{detected_class} {oid}",(centroid[0]-20,centroid[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
        if distance_m: cv2.putText(frame,f"{distance_m:.1f}m",(centroid[0]-20,centroid[1]+15),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
        cv2.circle(frame,(centroid[0],centroid[1]),4,(255,0,0),-1)

    for name, pts in GEOFENCES.items():
        pts_np = np.array(pts,np.int32).reshape((-1,1,2))
        cv2.polylines(frame,[pts_np],isClosed=True,color=(255,0,0),thickness=2)
        mc = np.array(pts).mean(axis=0).astype(int)
        cv2.putText(frame,name,tuple(mc),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),2)

    if alert_this_frame:
        cv2.putText(frame," ALERTA", (W//2-100,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,(0,0,255),3)

    cv2.putText(frame,f"Objetos rastreados: {len(objects)} | Alertas: {alert_count}",(10,30),
                cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

    if writer: writer.write(frame)
    cv2.imshow("Monitor Pedreira",frame)

    if len(log_rows)>=5:
        df = pd.read_csv(LOG_CSV)
        df_new = pd.DataFrame(log_rows)
        pd.concat([df,df_new],ignore_index=True).to_csv(LOG_CSV,index=False)
        log_rows=[]

    if cv2.waitKey(1)&0xFF == ord('q'): break

if log_rows:
    df = pd.read_csv(LOG_CSV)
    df_new = pd.DataFrame(log_rows)
    pd.concat([df,df_new],ignore_index=True).to_csv(LOG_CSV,index=False)

cap.release()
if writer: writer.release()
cv2.destroyAllWindows()
