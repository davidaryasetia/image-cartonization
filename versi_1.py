import threading
import winsound

import cv2
import imutils

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

_, start_frame = cap.read()
start_frame = imutils.resize(start_frame, width=500)
start_frame = cv2.cvtColor(start_frame, cv2.COLOR_BGR2GRAY)
start_frame = cv2.GaussianBlur(start_frame, (21, 21), 0)

alarm = False
alarm_mode = True  # Set alarm_mode to True initially for automatic activation
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
    original_frame = frame.copy()  # Copy original frame for display

    if alarm_mode:
        frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_bw = cv2.GaussianBlur(frame_bw, (5, 5), 0)

        difference = cv2.absdiff(frame_bw, start_frame)
        threshold = cv2.threshold(difference, 15, 255, cv2.THRESH_BINARY)[1]  # Decreased threshold value for more sensitivity
        start_frame = frame_bw

        motion_pixels = cv2.countNonZero(threshold)

        if motion_pixels > 1500:  # Adjust this threshold according to your requirement
            alarm_counter += 1
        else:
            if alarm_counter > 0:
                alarm_counter -= 1

        cv2.imshow("Alarm", threshold)
    else:
        cv2.imshow("Alarm", frame)  # Show original frame if alarm is not active

    cv2.imshow("Original", original_frame)  # Show original frame

    if alarm_counter > 10:  # Adjust this value according to your requirement
        if not alarm: 
            alarm = True
            threading.Thread(target=beep_alarm).start()

    key_pressed = cv2.waitKey(30)
    if key_pressed == ord("q"):
        alarm_mode = False
        break

cap.release()
cv2.destroyAllWindows()
