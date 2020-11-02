import dlib
import face_recognition
from cv2 import cv2
import os
import numpy as np

shape_predictor_model = "shape_predictor_68_face_landmarks.dat"
face_rec_128 = "dlib_face_recognition_resnet_model_v1.dat"
face_Unique_feature = dlib.face_recognition_model_v1(face_rec_128)
face_detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor(shape_predictor_model)

def load_images_from_folder(folder):
    features = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            face_2 = face_detector(img, 1)
            pose_landmarks_1 = face_pose_predictor(img, face_2[0])
            features.append(face_Unique_feature.compute_face_descriptor(img, pose_landmarks_1))
    return features
data = np.array(load_images_from_folder("C:\\Users\\Abdul Jalil\\face"))
np.save('data.npy', data)