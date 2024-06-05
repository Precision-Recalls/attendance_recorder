# importing the necessary libraries
import datetime
import os
import pickle
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras.backend as K
# from facenet_pytorch.models.mtcnn import MTCNN
from keras.models import load_model
from mtcnn import MTCNN
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load necessary models and objects
from src.utils.FaceEmbeddingModel import load_face_embedding_model
from src.utils.RecordAttendance import takeAttendance

person_rep = pickle.load(open(r'assets/names_mapping.pkl', 'rb'))
vgg_face_model_weight_file_path = 'assets/models/vgg_face_weights.h5'
vgg_face_model = load_face_embedding_model(vgg_face_model_weight_file_path)
classifier_model = load_model('assets/models/face_classifier.h5')

next_start = datetime.datetime(2024, 6, 4, 8, 0, 0)
unknown_counter = 0
start_time = time.time()
# Creating a VideoCapture object to read the video
cap = cv2.VideoCapture('VID_20210316_093644.mp4')

# Loop until the end of the video
while cap.isOpened():

    # Capture frame-by-frame
    ret, frame = cap.read()
    # cv2.imshow("Face Detection Tutorial: ", frame)
    dtn = datetime.datetime.now()
    if dtn >= next_start:
        next_start += datetime.timedelta(1)  # 1 day
        unknown_counter = 0

    # create the detector, using default weights
    detector = MTCNN(steps_threshold=[0.80, 0.85, 0.9])
    # detect faces in the image
    results = detector.detect_faces(frame)
    # extract the bounding box from the first face
    if results:
        for i, result in enumerate(results):
            x1, y1, width, height = result['box']
            # bug fix
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height

            # extract the face
            face = frame[y1:y2, x1:x2]
            # save crop image with person name as image name
            cropped_file_name = f'output/cropped_{i}.jpg'
            cv2.imwrite(cropped_file_name, face)

            # Get Embeddings
            crop_img = load_img(cropped_file_name, target_size=(224, 224))
            crop_img = img_to_array(crop_img)
            crop_img = np.expand_dims(crop_img, axis=0)
            crop_img = preprocess_input(crop_img)
            img_encode = vgg_face_model(crop_img)

            # Make Predictions
            embed = K.eval(img_encode)
            person = classifier_model.predict(embed)
            print(f'-----{person}---------')
            if (person > 0.7).any():
                name = person_rep.get(np.argmax(person))
                os.remove(cropped_file_name)
                print(f'----------{name}-------')
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                frame = cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 0,
                                    cv2.LINE_AA)
                # Save images with bounding box,name and accuracy
                if name == 'unknown':
                    name = name + f'{unknown_counter}'
                    unknown_counter += 1
                attendance_time = takeAttendance(name)
                final_image_name = f'output/recorded_faces/{name}_{attendance_time}.jpg'
                cv2.imwrite(final_image_name, face)
                # plt.figure(figsize=(8, 4))
                # plt.imshow(face)
                # plt.axis('off')
                # plt.show()
            else:
                print('Not a valid face detected!')
    # define q as the exit button
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# release the video capture object
cap.release()
# Closes all the windows currently opened.
cv2.destroyAllWindows()
end_time = time.time()
print(f'total processing time for this video is :- {end_time-start_time}')
