import logging
import os
import pickle
import random
from concurrent.futures import ThreadPoolExecutor
from os import listdir
from os.path import isdir

import cv2
import numpy as np
import tensorflow.keras.backend as K
from PIL import Image
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing.image import load_img, img_to_array
from mtcnn import MTCNN
from scipy.spatial.distance import cosine
from tensorflow.keras.applications.imagenet_utils import preprocess_input

# Load necessary models and objects
from config_loader import load_config
from src.training.train import load_face_embedding_model
from src.utils.commonUtils import take_attendance, get_pickle_obj

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the configuration
config = load_config('config.ini')

person_rep_path = config['paths']['person_rep_path']
person_rep = get_pickle_obj(person_rep_path)

person_encodings_filepath = config['paths']['person_encodings_filepath']
person_encodings = get_pickle_obj(person_encodings_filepath)

vgg_face_model_weight_file_path = os.path.abspath(config['paths']['vgg_face_model_weight_file_path'])
vgg_face_model = load_face_embedding_model(vgg_face_model_weight_file_path)

classifier_model_path = os.path.abspath(config['paths']['classifier_model_path'])
if os.path.isfile(classifier_model_path):
    classifier_model = load_model(classifier_model_path)

detector = MTCNN(steps_threshold=[0.80, 0.85, 0.9])

output_directory = config['paths']['output_directory']
recorded_faces_directory = config['paths']['recorded_faces_directory']
file_prefix = config['parameters']['file_prefix']


def augment_images(input_image, output_image_directory, image_prefix, total_images):
    try:
        image = load_img(input_image)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        aug = ImageDataGenerator(
            rotation_range=30,
            zoom_range=0.15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            horizontal_flip=True,
            fill_mode="nearest"
        )

        total = 0
        imageGen = aug.flow(image, batch_size=1, save_to_dir=output_image_directory,
                            save_prefix=image_prefix + '_aug', save_format="jpg")
        for _ in imageGen:
            total += 1
            if total == total_images:
                break
    except Exception as e:
        logger.error(f'Error :- {e} in image augmentation for image: {input_image}')


def create_images(input_image_directory, num_of_images):

    for subdir in listdir(input_image_directory):
        try:
            input_image_filepath = os.path.join(input_image_directory, subdir)
            if not isdir(input_image_filepath):
                continue
            for filename in listdir(input_image_filepath):
                input_image_filename = os.path.join(input_image_filepath, filename)
                augment_images(input_image_filename, input_image_filepath, subdir, num_of_images)
        except Exception as e:
            logger.error(f'Error :- {e} creating images for :- {subdir}')


def detect_faces_in_images(file_path):
    try:
        img = Image.open(file_path)
        if img is None or img.size == 0:
            logger.error(f"Please check image path {file_path} or some error occurred!")
            return [], []
        image = img.convert('RGB')
        pixels = np.asarray(image)
        results = detector.detect_faces(pixels)
        return pixels, results
    except Exception as e:
        logger.error(f"There is some error :- {e} in face detection from the images :- {file_path}")


def crop_faces(pixels, result, cropped_file_name):
    try:
        x1, y1, width, height = result['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = pixels[y1:y2, x1:x2]
        cv2.imwrite(cropped_file_name, face)
        return face
    except Exception as e:
        logger.error(f"Some error :- {e} in face cropping for the file :- {cropped_file_name}")


def get_facial_encodings(cropped_file_name):
    try:
        crop_img = load_img(cropped_file_name, target_size=(224, 224))
        crop_img = img_to_array(crop_img)
        crop_img = np.expand_dims(crop_img, axis=0)
        crop_img = preprocess_input(crop_img)
        img_encode = vgg_face_model(crop_img)
        embed = K.eval(img_encode)
        return embed
    except Exception as e:
        logger.error(f"Error :- {e} in encoding the file :- {cropped_file_name}")


def get_person_name(embed):
    person = classifier_model.predict(embed)
    print(f"Person classification probability score are as follows:- {person}")
    name = 'unknown'
    if (person > 0.85).any():
        name = person_rep.get(np.argmax(person))
    return name


# determine if a candidate face is a match for a known face
def is_match(known_embedding, candidate_embedding, thresh=0.40):
    # calculate distance between embeddings
    score = cosine(known_embedding, candidate_embedding)
    print(f"------{score}-------")
    return score <= thresh


def save_faces(name, face):
    if name == 'unknown':
        random_num = random.randint(5000, 10000)
        name = f'{name}{random_num}'
    attendance_time = take_attendance(name, config)
    final_image_name = os.path.join(recorded_faces_directory, f'{name}_{attendance_time}.jpg')
    cv2.imwrite(final_image_name, face)


def process_face(file_path):
    pixels, results = detect_faces_in_images(file_path)
    print('face got detected!')
    if pixels is not None and results is not None:
        for result in results:
            random_num = random.randint(15000, 20000)
            cropped_file_name = os.path.join(output_directory, f'{file_prefix}_{random_num}.jpg')
            face = crop_faces(pixels, result, cropped_file_name)
            embed = get_facial_encodings(cropped_file_name)
            print('embeddings created!')
            os.remove(cropped_file_name)
            name = get_person_name(embed)
            print(f"------original name is :----{name}-----")
            original_person_embeddings = person_encodings[name]
            matching_flag = is_match(original_person_embeddings, embed)
            name = name if matching_flag else 'unknown'
            print(f"-------{name}---")
            save_faces(name, face)


def image_processor(file_paths):
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_face, file_path) for file_path in file_paths]
        for future in futures:
            future.result()


def create_encodings(cropped_images_directory, person_encodings_filepath=''):
    X, Y = [], []
    person_rep = dict()
    person_encodings = dict()
    for i, person in enumerate(listdir(cropped_images_directory)):
        person_rep[i] = person
        if person not in person_encodings:
            person_encodings[person] = []
        image_names = listdir(os.path.join(cropped_images_directory, person))
        for image_name in image_names:
            cropped_file_name = os.path.join(cropped_images_directory, person, image_name)
            image_encodings = get_facial_encodings(cropped_file_name)
            X.append(np.squeeze(image_encodings).tolist())
            Y.append(i)
        person_encodings[person] = [np.mean(k) for k in zip(*X)]
    if person_encodings_filepath:
        with open(person_encodings_filepath, 'wb') as f:
            pickle.dump(person_encodings, f)
    return X, Y


def create_train_test_data(cropped_training_images_directory, cropped_testing_images_directory,
                           person_encodings_filepath):
    x_train, y_train = create_encodings(cropped_training_images_directory, person_encodings_filepath)
    x_test, y_test = create_encodings(cropped_testing_images_directory)
    return x_train, y_train, x_test, y_test
