import dlib
import openface
import numpy as np
from cv2 import cv2
from imutils import face_utils
video_capture = cv2.VideoCapture(0)
shape_predictor_model = "shape_predictor_68_face_landmarks.dat"
face_rec_128 = "dlib_face_recognition_resnet_model_v1.dat"
face_Unique_feature = dlib.face_recognition_model_v1(face_rec_128)
face_detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor(shape_predictor_model)
face_aligner = openface.AlignDlib(shape_predictor_model)
win = dlib.image_window()
def help(name):
    image = cv2.imread(name)
    face_2 = face_detector(image, 1)
    pose_landmarks_1 = face_pose_predictor(image, face_2[0])
    return face_Unique_feature.compute_face_descriptor(image, pose_landmarks_1)
path = "C:\\Users\\Abdul Jalil\\face\\3.jpg"
data = np.load('data.npy')
while True:
    ret, frame = video_capture.read()
    frame = cv2.flip(frame,1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_1 = face_detector(gray, 1)
    win.set_image(gray)
    for i, face_rect in enumerate(face_1):  
        win.add_overlay(face_rect)
        pose_landmarks = face_pose_predictor(frame, face_rect)
        win.add_overlay(pose_landmarks) 
        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(face_rect)
        cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Use openface to calculate and perform the face alignment
        alignedFace = face_aligner.align(534, frame, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)  
        face_features = face_Unique_feature.compute_face_descriptor(frame, pose_landmarks)
        #face_chip = dlib.get_face_chip(frame, pose_landmarks) 
        #face_descriptor_from_prealigned_image = face_Unique_feature.compute_face_descriptor(face_chip) 
        dist = []
        for i in data:
            dist.append(np.linalg.norm(np.array(face_features)-np.array(i)))
        print(max(dist),"---",np.average(dist))  
        if (np.average(dist) < 0.55):
            cv2.putText(gray, "Abdul Jalil - Verified".format(i + 1), (x - 10, y - 10),
		    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(gray, "- NOT Verified".format(i + 1), (x - 10, y - 10),
		    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        win.clear_overlay()
        cv2.imshow('Video', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):break
video_capture.release()
cv2.destroyAllWindows()