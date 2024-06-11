import datetime
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
from mtcnn import MTCNN
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing.image import load_img, img_to_array

# Load necessary models and objects
from src.training.train import load_face_embedding_model
from src.utils.commonUtils import takeAttendance

vgg_face_model_weight_file_path = 'assets/models/vgg_face_weights.h5'
detector = MTCNN(steps_threshold=[0.80, 0.85, 0.9])
vgg_face_model = load_face_embedding_model(vgg_face_model_weight_file_path)
classifier_model = load_model('assets/models/face_classifier.h5')
person_rep = pickle.load(open(r'assets/names_mapping.pkl', 'rb'))


def augment_images(input_image, output_image, image_prefix, total_images):
    # load the input image, convert it to a NumPy array, and then
    # reshape it to have an extra dimension
    print("[INFO] loading example image...")
    image = load_img(input_image)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    # construct the image generator for data augmentation then
    # initialize the total number of images generated thus far
    aug = ImageDataGenerator(
        rotation_range=30,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")
    total = 0

    # construct the actual Python generator
    print("[INFO] generating images...")
    imageGen = aug.flow(image, batch_size=1, save_to_dir=output_image,
                        save_prefix=image_prefix + '_aug', save_format="jpg")
    # loop over examples from our image data augmentation generator
    for image in imageGen:
        # increment our counter
        total += 1
        # if we have reached the specified number of examples, break
        # from the loop
        if total == total_images:
            break


def create_images(input_image_directory, num_of_images):
    # enumerate folders, on per class
    for subdir in listdir(input_image_directory):
        # path
        input_image_filepath = input_image_directory + subdir + '/'
        # skip any files that might be in the dir
        if not isdir(input_image_filepath):
            continue
        for filename in listdir(input_image_filepath):
            # path
            input_image_filename = input_image_filepath + filename
            augment_images(input_image_filename, input_image_filepath, subdir, num_of_images)


def detect_faces_in_images(file_path):
    # load image from file
    img = Image.open(file_path)
    results = []
    pixels = []
    if img is None or img.size is 0:
        print("Please check image path or some error occurred !")
        return pixels, results
    else:
        # convert to RGB, if needed
        image = img.convert('RGB')
        # convert to array
        pixels = np.asarray(image)
        # detect faces in the image
        results = detector.detect_faces(pixels)
        print("Faces got detected in the images!")
    return pixels, results


def crop_faces(pixels, result, cropped_file_name):
    # extract the bounding box from the first face
    x1, y1, width, height = result['box']
    # bug fix
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height

    # extract the face
    face = pixels[y1:y2, x1:x2]
    cv2.imwrite(cropped_file_name, face)
    return face


def get_facial_encodings(cropped_file_name):
    # Get Embeddings
    crop_img = load_img(cropped_file_name, target_size=(224, 224))
    crop_img = img_to_array(crop_img)
    crop_img = np.expand_dims(crop_img, axis=0)
    crop_img = preprocess_input(crop_img)
    img_encode = vgg_face_model(crop_img)
    embed = K.eval(img_encode)
    return embed


def get_person_name(embed):
    # Make Predictions
    person = classifier_model.predict(embed)
    name = person_rep.get(np.argmax(person))
    return name


def get_distances_between_faces(face_encoding_person1, face_encoding_person2):
    if len(face_encoding_person1) > 0 and len(face_encoding_person2) > 0:
        euclidean_distance = np.linalg.norm(face_encoding_person1[0] - face_encoding_person2[0])

        # Normalize Euclidean distance
        embedding_dimension = face_encoding_person1.shape[1]
        normalized_distance = euclidean_distance / np.sqrt(embedding_dimension)
        print("Euclidean Distance:", normalized_distance)

        cosine_distance = cosine_similarity([face_encoding_person1], [face_encoding_person2])[0][0]
        print("Cosine Similarity:", cosine_distance)
        return normalized_distance, cosine_distance
    else:
        print('Either of the one face encoding is empty! Please check.')
        return -1, -1


def save_faces(name, face):
    # Save images with bounding box,name and accuracy
    if name == 'unknown':
        random_num = random.randint(5000, 10000)
        name = name + f'{random_num}'
    attendance_time = takeAttendance(name)
    final_image_name = f'output/recorded_faces/{name}_{attendance_time}.jpg'
    cv2.imwrite(final_image_name, face)
    print('Faces saved successfully!')


def process_face(file_path):
    pixels, results = detect_faces_in_images(file_path)
    if pixels is not None and results is not None:
        for result in results:
            # save crop image with person name as image name
            random_num = random.randint(15000, 20000)
            cropped_file_name = f'output/cropped_{random_num}.jpg'
            face = crop_faces(pixels, result, cropped_file_name)
            embed = get_facial_encodings(cropped_file_name)
            os.remove(cropped_file_name)
            name = get_person_name(embed)
            save_faces(name, face)


def image_processor(file_paths):
    # Use ThreadPoolExecutor for concurrent processing
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_face, file_path) for i, file_path in enumerate(file_paths)]
        for future in futures:
            future.result()


def create_encodings(cropped_images_directory):
    X = []
    Y = []
    person_rep = dict()
    person_folders = os.listdir(cropped_images_directory)
    for i, person in enumerate(person_folders):
        person_rep[i] = person
        image_names = os.listdir(cropped_images_directory + person + '/')
        for image_name in image_names:
            cropped_file_name = cropped_images_directory + person + '/' + image_name
            image_encodings = get_facial_encodings(cropped_file_name)
            X.append(np.squeeze(image_encodings).tolist())
            Y.append(i)
    return X, Y


def create_train_test_data(cropped_training_images_directory, cropped_testing_images_directory):
    x_train, y_train = create_encodings(cropped_training_images_directory)
    x_test, y_test = create_encodings(cropped_testing_images_directory)
    return x_train, y_train, x_test, y_test
