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
alarm_mode = True
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

    # Create cursor frame
    cursor_frame = np.zeros_like(frame)
    cursor_x = int(frame.shape[1] / 2)  # Initialize cursor x position
    cursor_y = int(frame.shape[0] / 2)  # Initialize cursor y position

    # Update cursor position based on motion
    if alarm_mode:
        contours, _ = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            (x, y), _ = cv2.minEnclosingCircle(max_contour)
            cursor_x = int(x)
            cursor_y = int(y)

    # Draw cursor on cursor frame
    cv2.circle(cursor_frame, (cursor_x, cursor_y), 10, (0, 255, 0), -1)

    # Blend cursor frame with original frame
    alpha = 0.5
    beta = 1.0 - alpha
    blended_frame = cv2.addWeighted(frame_copy, alpha, cursor_frame, beta, 0.0)
    cv2.imshow("Original Frame with Cursor", blended_frame)

    if alarm_counter > 10:
        if not alarm:
            alarm = True
            threading.Thread(target=beep_alarm).start()

    key_pressed = cv2.waitKey(30)
    if key_pressed == ord("q"):
        alarm_mode = False
        break

cap.release()
cv2.destroyAllWindows()
