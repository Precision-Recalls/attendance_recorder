# importing the necessary libraries
import datetime
import os
import pickle
import time

import cv2
import numpy as np
import tensorflow.keras.backend as K
from keras.models import load_model
from matplotlib import pyplot as plt
from mtcnn import MTCNN
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load necessary models and objects
from src.training.train import load_face_embedding_model
from src.utils.commonUtils import takeAttendance
from src.utils.config_loader import load_config

# Load the configuration
config = load_config('config.ini')

person_rep_path = config['paths']['person_rep_path']
person_rep = pickle.load(open(person_rep_path, 'rb'))
vgg_face_model_weight_file_path = config['paths']['vgg_face_model_weight_file_path']
vgg_face_model = load_face_embedding_model(vgg_face_model_weight_file_path)
classifier_model_path = config['paths']['classifier_model_path']
classifier_model = load_model(classifier_model_path)

next_start = datetime.datetime(2024, 6, 4, 8, 0, 0)
unknown_counter = 0
start_time = time.time()

# Desired frame rate (in frames per second)
desired_frame_rate = 3

# Creating a VideoCapture object to read the video
cap = cv2.VideoCapture('test/VID_20210316_093644.mp4')

# Calculate the delay between frames in milliseconds
frame_delay = 1 / desired_frame_rate * 1000

# Get the original frame rate of the video
original_frame_rate = cap.get(cv2.CAP_PROP_FPS)
print(f"Original frame rate: {original_frame_rate} FPS")

frame_count = 0

try:
    # Loop until the end of the video
    while cap.isOpened():

        # Capture frame-by-frame
        ret, frame = cap.read()

        # Check if frame is read correctly
        if not ret:
            break

        # Control frame rate
        elapsed_time = time.time() - start_time
        time_to_sleep = max(0, (1 / desired_frame_rate) - elapsed_time)
        time.sleep(time_to_sleep)
        start_time = time.time()

        # Perform operations based on time condition
        dtn = datetime.datetime.now()
        if dtn >= next_start:
            next_start += datetime.timedelta(1)  # 1 day
            unknown_counter = 0

        if frame_count % 30 == 0:
            # Create the detector, using default weights
            detector = MTCNN(steps_threshold=[0.80, 0.85, 0.9])

            # Detect faces in the frame
            results = detector.detect_faces(frame)

            # Process each detected face
            if results:
                for i, result in enumerate(results):
                    x1, y1, width, height = result['box']
                    x1, y1 = abs(x1), abs(y1)
                    x2, y2 = x1 + width, y1 + height

                    # Extract the face region
                    face = frame[y1:y2, x1:x2]

                    # Save cropped image with person name as image name
                    cropped_file_name = f'output/cropped_{i}.jpg'
                    cv2.imwrite(cropped_file_name, face)

                    # Get embeddings
                    crop_img = load_img(cropped_file_name, target_size=(224, 224))
                    crop_img = img_to_array(crop_img)
                    crop_img = np.expand_dims(crop_img, axis=0)
                    crop_img = preprocess_input(crop_img)
                    img_encode = vgg_face_model(crop_img)

                    # Make predictions
                    embed = K.eval(img_encode)
                    person = classifier_model.predict(embed)

                    if (person > 0.7).any():
                        name = person_rep.get(np.argmax(person))

                        # Remove cropped image
                        os.remove(cropped_file_name)

                        # Display bounding box and name
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        frame = cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 0, cv2.LINE_AA)

                        # Save image with bounding box, name, and accuracy
                        if name == 'unknown':
                            name = name + f'{unknown_counter}'
                            unknown_counter += 1
                        attendance_time = takeAttendance(name)
                        final_image_name = f'output/recorded_faces/{name}_{attendance_time}.jpg'
                        cv2.imwrite(final_image_name, face)

                    else:
                        print('Not a valid face detected!')

        frame_count += 1

        # Display the frame
        cv2.imshow("Face Detection Tutorial", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"Total frames processed: {frame_count}")

except Exception as e:
    print(f"Error occurred: {e}")

finally:
    # Release the video capture object
    cap.release()

    # Close all windows
    cv2.destroyAllWindows()

    # Calculate and print total processing time
    end_time = time.time()
    print(f'Total processing time: {end_time - start_time} seconds')
