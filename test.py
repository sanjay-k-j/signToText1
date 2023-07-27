import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

cap = cv2.VideoCapture(0)
detector =  HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5","Model/labels.txt")


labels = ["0","1","2","3","4","5","6","7","8","9"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    prediction, index = classifier.getPrediction(img)
    print(prediction, index)
    cv2.putText(imgOutput, labels[index],cv2.FONT_HERSHEY_COMPLEX,2,(255,0,0),2)
    cv2.imshow("Image",imgOutput)
    cv2.waitKey(1)
