import cv2
import dlib

#Capturing video from webcam and importing detector and predictor
cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
#Predictor .dat file taken from https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    _,frame = cap.read()
    
    #Converting frame to grayscale for processing
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    #Detecting faces using dlib frontal face detector
    faces = detector(gray)
    
    #Printing number of faces detected
    l = len(faces)
    if l == 1:
        print("1 face detected.")
    else:
        print(str(l) + " faces detected.")
        
    #Printing facial landmarks for all detected faces
    for face in faces:
        landmarks = predictor(gray,face)
         
        for n in range(0,68):
             x = landmarks.part(n).x
             y = landmarks.part(n).y
             
             cv2.circle(frame,(x,y),3,(0,0,255),-1)
    
    #Showing the output video with facial landmarks         
    cv2.imshow("Out",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Releasing the webcam
cap.release()
cv2.destroyAllWindows()
