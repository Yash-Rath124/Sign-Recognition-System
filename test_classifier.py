import cv2
import pickle
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# Load trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Alphabet mapping
labels_dict = {
0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',
10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',
18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'
}

# Load MediaPipe hand model
model_path = "hand_landmarker.task"

base_options = python.BaseOptions(model_asset_path=model_path)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

hand_landmarker = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    result = hand_landmarker.detect(mp_image)

    if result.hand_landmarks:

        for hand_landmarks in result.hand_landmarks:

            data_aux = []
            x_ = []
            y_ = []

            # Collect coordinates
            for landmark in hand_landmarks:
                x_.append(landmark.x)
                y_.append(landmark.y)

            # Normalize coordinates
            for landmark in hand_landmarks:
                data_aux.append(landmark.x - min(x_))
                data_aux.append(landmark.y - min(y_))

            if len(data_aux) == 42:

                prediction = model.predict([np.asarray(data_aux)])

                predicted_letter = labels_dict[int(prediction[0])]

                h, w, _ = frame.shape
                x1 = int(min(x_) * w)
                y1 = int(min(y_) * h)

                cv2.putText(
                    frame,
                    predicted_letter,
                    (x1, y1 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0,0,255),
                    4
                )

    cv2.imshow("ASL Hand Sign Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()