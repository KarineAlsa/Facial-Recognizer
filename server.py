from flask import Flask, render_template, Response
import cv2

app = Flask(__name__, static_folder='static')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detector')
def detector():
    return render_template('init.html')

def recognize_faces():
    face_recognizer = cv2.face.LBPHFaceRecognizer.create()

    face_recognizer.read('my_face_recognizer_model.xml')

    cap = cv2.VideoCapture(0)

    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        aux_frame = gray.copy()
        for (x, y, w, h) in faces:
            face = aux_frame[y:y+h,x:x+w]
            face = cv2.resize(face, (720,720), interpolation=cv2.INTER_CUBIC)
            result = face_recognizer.predict(face)


            if result[1] < 18:
                cv2.rectangle(frame,(x,y),(x+w ,y+h),(0,255,0),2)
                cv2.putText(frame, "yo", (x,y-25), 2,1.1,(0,255,0),1,cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(recognize_faces(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)