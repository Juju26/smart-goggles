import numpy as np
from cv2 import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle
import os
import imutils

curr_path = os.getcwd()

#print("Loading face detection model")
proto_path = os.path.join(curr_path, 'model', 'deploy.prototxt')
model_path = os.path.join(curr_path, 'model', 'res10_300x300_ssd_iter_140000.caffemodel')
face_detector = cv2.dnn.readNetFromCaffe(prototxt=proto_path, caffeModel=model_path)

#print("Loading face recognition model")
recognition_model = os.path.join(curr_path, 'model', 'openface_nn4.small2.v1.t7')
face_recognizer = cv2.dnn.readNetFromTorch(model=recognition_model)

data_base_path = os.path.join(curr_path, 'database')

filenames = []
for path, subdirs, files in os.walk(data_base_path):
    for name in files:
        filenames.append(os.path.join(path, name))   #adding avail img to list

face_embeddings = []
face_names = []
co=1
for (i, filename) in enumerate(filenames):
    print(str(co)+"Processing image {}".format(filename))
    #print(co)
    co+=1

    image = cv2.imread(filename)
    image = imutils.resize(image, width=600)

    (h, w) = image.shape[:2]

    image_blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)

    face_detector.setInput(image_blob)
    face_detections = face_detector.forward()

    i = np.argmax(face_detections[0, 0, :, 2])
    confidence = face_detections[0, 0, i, 2]

    if confidence >= 0.5:

        box = face_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        face = image[startY:endY, startX:endX]
        try:
            face_blob = cv2.dnn.blobFromImage(face, 1.0/255, (96,96), (0, 0), True, False)
            face_recognizer.setInput(face_blob)
            face_recognitions = face_recognizer.forward()

            name = filename.split(os.path.sep)[-2]

            face_embeddings.append(face_recognitions.flatten())
            face_names.append(name)
        except cv2.error:
            print('Face not suitable')
            pass


data = {"embeddings": face_embeddings, "names": face_names}  #dictionary with face embeddings along with name

le = LabelEncoder()
labels = le.fit_transform((data["names"]))

recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

f = open('recognizer.pickle', "wb")
f.write(pickle.dumps(recognizer))   #img data pickled to recognizer
f.close()

f = open("le.pickle", "wb")
f.write(pickle.dumps(le))  #labels pickled as le
f.close()