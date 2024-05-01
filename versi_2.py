import threading
import winsound

import cv2
import imutils
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

_, start_frame = cap.read()
start_frame = imutils.resize(start_frame, width=500)
start_frame = cv2.cvtColor(start_frame, cv2.COLOR_BGR2GRAY)
start_frame = cv2.GaussianBlur(start_frame, (21, 21), 0)

alarm = False
alarm_mode = False
alarm_counter = 0

def beep_alarm():
    global alarm
    for _ in range(5):
        if not alarm_mode:
            break
        print("ALARM")
        winsound.Beep(2500, 1000)
    alarm = False

while True:
    _, frame = cap.read()
    frame = imutils.resize(frame, width=500)
    frame_copy = frame.copy()

    if alarm_mode:
        frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_bw = cv2.GaussianBlur(frame_bw, (5, 5), 0)

        difference = cv2.absdiff(frame_bw, start_frame)
        threshold = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)[1]
        start_frame = frame_bw

        if threshold.sum() > 300:
            print(threshold.sum())
            alarm_counter += 1
        else:
            if alarm_counter > 0:
                alarm_counter -= 1

        cv2.imshow("Motion Detection", threshold)
    else:
        cv2.imshow("Motion Detection", frame)

    # Create matrix radar
    radar = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
    cv2.circle(radar, (frame.shape[1] // 2, frame.shape[0] // 2), min(frame.shape[0] // 2, frame.shape[1] // 2), (0, 255, 0), 2)
    cv2.imshow("Radar", radar)

    # Blend matrix radar with original frame
    alpha = 0.5
    beta = 1.0 - alpha
    blended_frame = cv2.addWeighted(frame_copy, alpha, radar, beta, 0.0)
    cv2.imshow("Original Frame with Radar", blended_frame)

    if alarm_counter > 50:
        if not alarm:
            alarm = True
            threading.Thread(target=beep_alarm).start()

    key_pressed = cv2.waitKey(30)
    if key_pressed == ord("t"):
        alarm_mode = not alarm_mode
        alarm_counter = 0
    if key_pressed == ord("q"):
        alarm_mode = False
        break

cap.release()
cv2.destroyAllWindows()
