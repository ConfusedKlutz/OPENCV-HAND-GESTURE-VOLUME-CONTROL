import cv2 as cv
import time
import numpy as np
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import mediapipe as mp

wCam, hCam = 1288, 720  # Set the size of the screen

cap = cv.VideoCapture(0)
cap.set(3, wCam)  # Set width (id at 3)
cap.set(4, hCam)  # Set height (id at 4)
pTime = 0

# We want the hand detector to work better, so we'll change the detection confidence
detector = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
vol_bar = 400
vol_per = 0

while True:
    success, img = cap.read()
    img = cv.flip(img, 1)  # Flip the image horizontally for a more intuitive experience
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = detector.process(imgRGB)
    lmList = []

    if results.multi_hand_landmarks:
        for handLandmarks in results.multi_hand_landmarks:
            for id, lm in enumerate(handLandmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

        if len(lmList) != 0:
            x1, y1 = lmList[4][1], lmList[4][2]  # Index finger
            x2, y2 = lmList[8][1], lmList[8][2]  # Thumb

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            cv.circle(img, (x1, y1), 15, (255, 0, 255), -1)
            cv.circle(img, (x2, y2), 15, (255, 0, 255), -1)
            cv.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv.circle(img, (cx, cy), 15, (255, 0, 255), -1)

            length = math.hypot(x2 - x1, y2 - y1)
            vol = np.interp(length, [50, 300], [minVol, maxVol])
            vol_bar = np.interp(length, [50, 300], [400, 150])
            vol_per = np.interp(length, [50, 300], [0, 100])

            volume.SetMasterVolumeLevel(vol, None)

            if length < 50:
                cv.circle(img, (cx, cy), 15, (0, 255, 0), -1)

    cv.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv.rectangle(img, (50, int(vol_bar)), (85, 400), (255, 0, 0), -1)
    cv.putText(img, f'{int(vol_per)}%', (40, 450), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv.putText(img, f'FPS: {int(fps)}', (40, 50), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
    cv.imshow("IMG", img)

    if cv.waitKey(1) == 27:  # Press 'Esc' key to exit the program
        break

cap.release()
cv.destroyAllWindows()
