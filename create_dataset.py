import os
import cv2
import pickle
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


model_path = "hand_landmarker.task"

base_options = python.BaseOptions(model_asset_path=model_path)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2,
)

hand_landmarker = vision.HandLandmarker.create_from_options(options)


DATA_DIR = "./data"

data = []
labels = []


for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        
        data_aux = []
        
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert to MediaPipe Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=img_rgb
        )

        # Run detection
        results = hand_landmarker.detect(mp_image)

        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                for landmark in hand_landmarks:
                    data_aux.append(landmark.x)
                    data_aux.append(landmark.y)

            data.append(data_aux)
            labels.append(dir_)


f = open('data.pickle', 'wb')
pickle.dump({"data": data, "labels": labels}, f)
f.close()

print("Dataset created successfully!")