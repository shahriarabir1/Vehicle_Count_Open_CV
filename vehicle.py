import cv2
import numpy as np

cap = cv2.VideoCapture("video.mp4")
lineP = 550
min_width = 80
min_height = 80

detect = cv2.bgsegm.createBackgroundSubtractorMOG()

def cph(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    xc = x + x1
    yc = y + y1
    return xc, yc

det = []
offset = 6
counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)
    app = detect.apply(blur)
    delt = cv2.dilate(app, np.ones((5, 5)))
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilstr = cv2.morphologyEx(delt, cv2.MORPH_CLOSE, kern)
    dilstr = cv2.morphologyEx(dilstr, cv2.MORPH_CLOSE, kern)
    countor, h = cv2.findContours(dilstr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.line(frame, (10, lineP), (1800, lineP), (255, 127, 0), 3)
  
    for (i, c) in enumerate(countor):
        (x, y, w, h) = cv2.boundingRect(c)
        validate = (w >= min_width) and (h >= min_height)
        if not validate:
            continue
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        center = cph(x, y, w, h)
        det.append(center)
        cv2.circle(frame, center, 4, (0, 0, 255), -1)
        
        for (cx, cy) in det:
            if cy < (lineP + offset) and cy > (lineP - offset):
                counter += 1
                cv2.line(frame, (40, lineP), (1800, lineP), (0, 127, 255), 3)
                det.remove((cx, cy))
                print("Vehicle Counter: " + str(counter))
        
    cv2.putText(frame, "Vehicle Counter: " + str(counter), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
    
    cv2.imshow("dilate", dilstr)
    cv2.imshow("frame", frame)
    
    if cv2.waitKey(50) == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
