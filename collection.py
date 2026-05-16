import os
import cv2
import time

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

letters = 7  # A-Z
dataset_size = 100

cap = cv2.VideoCapture(0)
for letter in range(letters):
    letter_dir = os.path.join(DATA_DIR, str(letter)) 
    if not os.path.exists(letter_dir):
        os.makedirs(letter_dir)

    print(f'Collecting data for letter: {chr(65 + letter)}')
    time.sleep(5)  # Wait before starting

    for img_num in range(dataset_size):
        ret, frame = cap.read()
        if not ret:
            continue

        img_path = os.path.join(letter_dir, f'{img_num + 100}.jpg')
        cv2.imwrite(img_path, frame)

        frame = cv2.flip(frame, 1)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(200) & 0xFF == ord('q'):
            break

    print(f'Finished collecting data for letter: {chr(65 + letter)}')

cap.release()
cv2.destroyAllWindows()