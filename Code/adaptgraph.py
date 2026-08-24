import cv2
import numpy as np
import time
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import psutil
from threading import Thread
from tracker import Tracker


# =========================
# CAMERA THREAD
# =========================

class CameraThread:

    def __init__(self, cap):

        self.cap = cap
        self.ret, self.frame = cap.read()
        self.running = True

        Thread(target=self.update, daemon=True).start()

    def update(self):

        while self.running:
            self.ret, self.frame = self.cap.read()

    def read(self):

        return self.ret, self.frame


# =========================
# COCO CLASSES
# =========================

class_names = [
"person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat",
"traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
"dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
"umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
"kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
"bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
"sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake",
"chair","sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop",
"mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink",
"refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]


# =========================
# TRACKER + OPENCV SETTINGS
# =========================

tracker = Tracker(max_lost=20)

cv2.setUseOptimized(True)
cv2.setNumThreads(2)


# =========================
# LOAD TENSORRT ENGINE
# =========================

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

with open("yolov5n.engine", "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

inputs = []
outputs = []
bindings = []
stream = cuda.Stream()

for binding in engine:

    shape = engine.get_binding_shape(binding)
    size = trt.volume(shape)

    dtype = trt.nptype(engine.get_binding_dtype(binding))

    device_mem = cuda.mem_alloc(size * dtype().nbytes)

    bindings.append(int(device_mem))

    if engine.binding_is_input(binding):
        inputs.append(device_mem)
    else:
        outputs.append(device_mem)


# =========================
# CAMERA INIT
# =========================

cap = cv2.VideoCapture(0)
cam = CameraThread(cap)

INPUT_SIZE = 320
DISPLAY_SIZE = (512, 384)


# =========================
# VARIABLES
# =========================

frame_count = 0
skip_frames = 1

mode = "QUALITY"

conf_thresh = 0.25
target_conf_thresh = 0.25

latency_avg = 0
temp_avg = 50

last_detections = np.empty((0, 6))


# =========================
# MODE CONTROLLER TIMER
# =========================

mode_last_changed = time.time()
MIN_MODE_DURATION = 5


# =========================
# FPS COUNTER
# =========================

fps_timer = time.time()
fps_counter = 0
display_fps = 0


# =========================
# TEMP SENSOR
# =========================

def get_temp():

    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read()) / 1000.0
    except:
        return 0


# =========================
# WINDOW INIT
# =========================


# =========================
# RESET GRAPH LOG FILES
# =========================

open("adaptive_fps.txt","w").close()
open("adaptive_latency.txt","w").close()


cv2.namedWindow("FINAL SYSTEM", cv2.WINDOW_NORMAL)
cv2.resizeWindow("FINAL SYSTEM", DISPLAY_SIZE[0], DISPLAY_SIZE[1])


# =========================
# MAIN LOOP
# =========================

while True:

    ret, frame = cam.read()

    if not ret:
        break

    start_time = time.time()

    cpu = psutil.cpu_percent()
    temp = get_temp()

    temp_avg = 0.9 * temp_avg + 0.1 * temp


    # =========================
    # ADAPTIVE CONTROLLER
    # =========================

    current_time = time.time()

    allow_switch = (current_time - mode_last_changed) > MIN_MODE_DURATION

    new_mode = mode


    if temp_avg > 58:

        new_mode = "COOLING"


    elif latency_avg > 65 or cpu > 80:

        new_mode = "PERFORMANCE"


    elif latency_avg > 45 or cpu > 40 or temp_avg > 55:

        new_mode = "BALANCED"


    else:

        new_mode = "QUALITY"


    if new_mode != mode and allow_switch:

        mode = new_mode
        mode_last_changed = current_time


    # =========================
    # APPLY MODE SETTINGS
    # =========================

    if mode == "QUALITY":

        skip_frames = 0
        target_conf_thresh = 0.25


    elif mode == "BALANCED":

        skip_frames = 1
        target_conf_thresh = 0.30


    elif mode == "PERFORMANCE":

        skip_frames = 2
        target_conf_thresh = 0.35


    elif mode == "COOLING":

        skip_frames = 3
        target_conf_thresh = 0.40


    conf_thresh = 0.8 * conf_thresh + 0.2 * target_conf_thresh


    run_detection = (frame_count % (skip_frames + 1) == 0)

    detections = []


    # =========================
    # INFERENCE
    # =========================

    if run_detection:

        img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))

        img = np.expand_dims(img, axis=0)
        img = np.ascontiguousarray(img)

        context.set_binding_shape(0, img.shape)

        cuda.memcpy_htod_async(inputs[0], img, stream)

        context.execute_async_v2(bindings=bindings,
                                 stream_handle=stream.handle)

        output_shape = context.get_binding_shape(1)
        output_shape = tuple(1 if x == -1 else x for x in output_shape)

        output = np.empty(output_shape, dtype=np.float32)

        cuda.memcpy_dtoh_async(output, outputs[0], stream)
        stream.synchronize()

        output = output.reshape(-1, 85)

        h, w = frame.shape[:2]


        boxes = []
        scores = []
        class_ids = []

        for row in output:

            conf = row[4]

            if conf < conf_thresh:
                continue

            class_id = np.argmax(row[5:])
            class_conf = row[5 + class_id]

            score = conf * class_conf

            if score < conf_thresh:
                continue

            cx, cy, bw, bh = row[:4]

            x1 = int((cx - bw/2) * w / INPUT_SIZE)
            y1 = int((cy - bh/2) * h / INPUT_SIZE)
            x2 = int((cx + bw/2) * w / INPUT_SIZE)
            y2 = int((cy + bh/2) * h / INPUT_SIZE)

            boxes.append([x1, y1, x2-x1, y2-y1])
            scores.append(score)
            class_ids.append(class_id)


        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, 0.45)

        detections = []

        if len(indices) > 0:

            for i in indices.flatten():

                x, y, bw, bh = boxes[i]

                detections.append([
                    x,
                    y,
                    x + bw,
                    y + bh,
                    scores[i],
                    class_ids[i]
                ])

        detections = np.array(detections) if len(detections) > 0 else np.empty((0,6))

        last_detections = detections

    else:

        detections = last_detections


    # =========================
    # TRACKER UPDATE
    # =========================

    track_input = detections[:, :5] if len(detections) > 0 else np.empty((0,5))

    tracked_objects = tracker.update(track_input)


    # =========================
    # CLASS LOOKUP CACHE
    # =========================

    cls_lookup = {}

    for det in detections:

        x1, y1, _, _, _, cls = det

        cls_lookup[(int(x1), int(y1))] = class_names[int(cls)]


    # =========================
    # DRAW LABEL ALWAYS ON TOP
    # =========================

    for obj in tracked_objects:

        x1, y1, x2, y2, obj_id = obj
        x1, y1, x2, y2 = map(int, [x1,y1,x2,y2])

        cls_name = cls_lookup.get((x1,y1), "")

        label = f"{cls_name} | ID {int(obj_id)}"

        font_scale = 0.7
        thickness = 2

        (tw, th), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness
        )

        label_y = y1 - 6

        if label_y - th - 6 < 0:
            label_y = y1 + th + 10

        cv2.rectangle(frame,
                      (x1, label_y - th - 6),
                      (x1 + tw + 6, label_y),
                      (0,255,0),
                      -1)

        cv2.rectangle(frame,
                      (x1,y1),
                      (x2,y2),
                      (0,255,0),
                      2)

        cv2.putText(frame,
                    label,
                    (x1+3,label_y-3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0,0,0),
                    thickness)


    latency = (time.time() - start_time) * 1000
    latency_avg = 0.8 * latency_avg + 0.2 * latency


    fps_counter += 1

    if time.time() - fps_timer >= 1:

        display_fps = fps_counter

        with open("adaptive_fps.txt","a") as f:
            f.write(f"{display_fps}\n")

        with open("adaptive_latency.txt","a") as f:
            f.write(f"{latency_avg}\n")

        fps_counter = 0
        fps_timer = time.time()




    cv2.putText(frame,f"FPS: {display_fps}",(10,25),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

    cv2.putText(frame,f"Latency: {latency_avg:.1f}",(10,50),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)

    cv2.putText(frame,f"CPU: {cpu:.1f}%",(10,75),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)

    cv2.putText(frame,f"TEMP: {temp_avg:.1f}",(10,100),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)

    cv2.putText(frame,f"MODE: {mode}",(10,125),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)


    frame = cv2.resize(frame, DISPLAY_SIZE)

    cv2.imshow("FINAL SYSTEM", frame)

    frame_count += 1

    if cv2.waitKey(1) == 27:
        break


cap.release()
cv2.destroyAllWindows()

