import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture (0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
#method that helps draw all points detected during hand movement
mpDraw = mp.solutions.drawing_utils
pTime = 0
CTime = 0
    

while True:
    success, img = cap.read ()
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)       #essential
    results = hands.process (img_rgb)                   
    
      #function that displays coordinates when a user's hands are detected.
    if results.multi_hand_landmarks:     
        for handLms in results.multi_hand_landmarks:                      #a single hand as in multiple hands in arrays that could be obtained as indices
           #check index numbers of the handlandmarks with their ids
           for id,  lm in enumerate (handLms.landmark):
             #0 for bottom of the finger, 4 for the tip of the finger
             
             #ratio of the image, multiply by the wwidth and the height to obtain the pixel value
             #print (id, lm)
             
             h, w, c  = img.shape 
             
             cx, cy = int (lm.x*w), int (lm.y*h)     #calculated coordinates in image
             print (id, cx, cy)    
             
             #if id ==4:
              # cv.circle (img, (cx, cy), 25, (255,0,255), -1)
            # mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)  #joins the dots to form connections (lines)
             #if id ==3:
              # cv.circle (img, (cx, cy), 25, (0,255,255), -1)
             #mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
             #if id ==2:
              # cv.circle (img, (cx, cy), 25, (255,255,255), -1)
            # mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
        
             cv.circle (img, (cx, cy), 15, (255,255,255), -1)                        #draws over the points on the hand
             mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
             
             #21 values for each hand to be imported as a module (makes it easy)
    
    cTime  = time.time ()
    fps = 1/(cTime-pTime)
    pTime = cTime
    
    cv.putText (img, str (int(fps)), (10,70), cv.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

   
    cv.imshow ("Image",img)
    cv.waitKey (1)
 
    