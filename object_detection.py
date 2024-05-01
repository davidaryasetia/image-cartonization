import threading
import winsound

import cv2
import imutils
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

alarm = False
alarm_mode = True
alarm_counter = 0

# Inisialisasi frame sebelumnya
prev_frame = None

def beep_alarm():
    global alarm
    for _ in range(5):
        if not alarm_mode:
            break
        print("ALARM")
        winsound.Beep(2500, 1000)
    alarm = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=500)
    original_frame = frame.copy()
    frame_copy = frame.copy()

    if alarm_mode:
        # Memastikan prev_frame tidak None sebelum menggunakannya
        if prev_frame is not None:
            # Hitung perbedaan antara dua frame
            fg_mask = cv2.absdiff(prev_frame, frame)

            if np.sum(fg_mask) > 1000:  # Mengurangi ambang deteksi untuk objek yang lebih kecil
                print(np.sum(fg_mask))
                alarm_counter += 1
            else:
                if alarm_counter > 0:
                    alarm_counter -= 1

            cv2.imshow("Motion Detection", fg_mask)

        # Simpan frame saat ini sebagai frame sebelumnya
        prev_frame = frame.copy()
    else:
        cv2.imshow("Motion Detection", frame)

    # Object detection
    fg_mask = bg_subtractor.apply(frame)  # Menggunakan background subtractor untuk mendapatkan fg_mask
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) > 200:  # Mengurangi ukuran minimal objek yang terdeteksi
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"({x}, {y})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display frames
    cv2.imshow("Object Detection", frame)

    #Original 
    cv2.imshow("Original Frame", original_frame)
    

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
