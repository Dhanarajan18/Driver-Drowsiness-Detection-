import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

EYE_CLOSED_FRAMES = 20
counter = 0

cap = cv2.VideoCapture(0)
print("Camera opened:", cap.isOpened())  # Debugging statement

while True:
    ret, frame = cap.read()
    print("Reading frame:", ret)  # Debugging statement
    
    if not ret:
        print("Failed to capture frame. Exiting...")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))
        
        if len(eyes) == 0:
            counter += 1
            if counter >= EYE_CLOSED_FRAMES:
                cv2.putText(frame, "DROWSINESS ALERT!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            counter = 0
    
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()