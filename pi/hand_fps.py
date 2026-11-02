import time
import cv2
import mediapipe as mp
import numpy as np

N = 200 #number of frames to measure FPS over
cap = cv2.VideoCapture(2)#open the default camera (index 0)
#1 will open the webcam or CAMO extension ?
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    print("Error opening video stream or file")
    exit()
mp_hands = mp.solutions.hands#hand tracking module
mp_drawing = mp.solutions.drawing_utils#tool for drawing hand landmarks on the video frame

def ram_mem():
    for line in open('/proc/self/status'):#read the current process's memory usage from the system
        if line.startswith('VmRSS:'):
            return int(line.split()[1])/1024  # return the resident set size in megabytes
    return 0  # return 0 if VmRSS is not found

def temp_cpu():
    # read the CPU temperature from the system
    return int(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000
#variables for tracking FPS and system metrics
proc = []#store how long each frame takes to process
seen = 0#number of frames seen
last = None#timestamp of the last frame

start = time.perf_counter()#timestamp of the start of the FPS measurement

with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands:
    for _ in range(N):
        worked, frame = cap.read()#read a frame from the camera
        if not worked:
            continue
        t0 = time.perf_counter()
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))#process the frame with the hand tracking module
        proc.append(time.perf_counter() - t0)
        if res.multi_hand_landmarks:
            seen += 1
            mp_drawing.draw_landmarks(frame, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
        last = frame
total = time.perf_counter() - start
cap.release()
if last is not None:
    cv2.imwrite("last_frame.jpg", last)
avg = sum(proc) / len(proc)
print(f"frames: {len(proc)}  hand found in {seen}")
print(f"MediaPipe only: {avg*1000:.1f} ms/frame (~{1/avg:.1f} FPS)")
print(f"End to end: {len(proc)/total:.1f} FPS")
#print(f"RAM used: {ram_mem():.0f} MB   CPU temp: {temp_cpu():.1f} C")
