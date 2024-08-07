import logging
import os
import time
from datetime import timedelta
from os.path import isfile

import cv2
import face_recognition
import numpy as np
from config_loader import load_config
from mtcnn import MTCNN


# Initialize logging
from src.training.train import train_face_classifier
from src.utils.dataUtils import import_image_files, process_files
from src.utils.imageUtils import create_images, create_train_test_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    try:
        # Load the configuration
        config = load_config('config.ini')
        logger.info("Configuration loaded successfully.")
        # Extract configuration settings
        source_image_folder = config['paths']['source_image_folder']
        image_destination_folder = config['paths']['image_destination_folder']
        num_of_images = int(config['parameters']['num_of_images'])
        training_data_directory = config['paths']['training_data_directory']
        training_cropped_data_directory = config['paths']['training_cropped_data_directory']
        test_data_directory = config['paths']['test_data_directory']
        testing_cropped_data_directory = config['paths']['testing_cropped_data_directory']
        classifier_model_path = config['paths']['classifier_model_path']
        person_rep_path = config['paths']['person_rep_path']
        person_encodings_filepath = config['paths']['person_encodings_filepath']
        # Setup service
        setup_service(source_image_folder, image_destination_folder, num_of_images,
                      training_data_directory, training_cropped_data_directory,
                      test_data_directory, testing_cropped_data_directory,
                      classifier_model_path, person_rep_path, person_encodings_filepath)

    except Exception as e:
        logger.error(f"Error occurred: {e}")


def setup_service(source_image_folder, image_destination_folder, num_of_images,
                  training_data_directory, training_cropped_data_directory,
                  test_data_directory, testing_cropped_data_directory,
                  classifier_model_path, person_rep_path, person_encodings_filepath):
    try:
        # Import image files from the source folder of the client
        import_image_files(source_image_folder, image_destination_folder)
        logger.info('Images imported successfully!')

        # Create more images if we have lesser image per person
        create_images(image_destination_folder, num_of_images)
        logger.info('Additional images created successfully!')

        # Crop faces from training and test images
        process_files(training_data_directory, training_cropped_data_directory, person_rep_path)
        logger.info('Training images cropped and saved successfully!')

        process_files(test_data_directory, testing_cropped_data_directory)
        logger.info('Testing images cropped and saved successfully!')

        # Get embedding of all the faces
        x_train, y_train, x_test, y_test = create_train_test_data(training_cropped_data_directory,
                                                                  testing_cropped_data_directory,
                                                                  person_encodings_filepath)
        logger.info('Training and testing data_v2 prepared successfully!')
        # Train the face classifier model
        train_face_classifier(x_train, y_train, x_test, y_test, classifier_model_path)
        logger.info('Face classifier model trained successfully!')

    except Exception as e:
        logger.error(f"Error in setup_service: {e}")
        raise


def load_known_faces(known_faces_dir):
    known_face_encodings = []
    known_face_names = []

    for person_name in os.listdir(known_faces_dir):
        person_dir = os.path.join(known_faces_dir, person_name)
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(person_name)

    return known_face_encodings, known_face_names


def new_verify(video_path):
    video_capture = cv2.VideoCapture(0)
    while True:
        start_time = time.time()

        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Convert the image from BGR color (which OpenCV uses) to RGB color
        rgb_frame = frame[:, :, ::-1]

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Initialize an array for the name of the detected face
        face_names = []

        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            start_time = time.perf_counter()
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            duration = timedelta(seconds=time.perf_counter() - start_time)
            print(f'time taken for comparing one face :- {duration}')
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Ensure processing time per frame is below 1 second
        elapsed_time = time.time() - start_time
        if elapsed_time < 1.0:
            time.sleep(1.0 - elapsed_time)

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()
    return face_names


if __name__ == "__main__":
    # main()
    known_face_encodings, known_face_names = load_known_faces('faces')
    detected_face_names = new_verify('test/WhatsApp Video 2024-06-20 at 17.45.47.mp4')
    print(detected_face_names)
