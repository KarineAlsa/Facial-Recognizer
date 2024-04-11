import os
import cv2

data_path = './Data'
image_path = os.listdir(data_path)
print(image_path)

face_recognizer = cv2.face.LBPHFaceRecognizer.create()

face_recognizer.read('my_face_recognizer_model.xml')

cap = cv2.VideoCapture(0, cv2.CAP_ANY)

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if ret == False: break
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    aux_frame = gray.copy()

    faces = face_classifier.detectMultiScale(gray,1.3,5)

    # aca va lo del cuadro de la persona
    for (x,y,w,h) in faces:
        face = aux_frame[y:y+h,x:x+w]
        face = cv2.resize(face, (720,720), interpolation=cv2.INTER_CUBIC)
        result = face_recognizer.predict(face)


        if result[1] < 9:
            cv2.putText(frame, "{}".format(image_path[result[0]]), (x,y-25), 2,1.1,(0,255,0),1,cv2.LINE_AA)
            cv2.rectangle(frame,(x,y),(x+w ,y+h),(0,255,0),2)
        
    cv2.imshow('frame',frame)
    k=cv2.waitKey(1)
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()