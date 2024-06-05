# Prepare Train/Test Data
import os

import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import tensorflow.keras.backend as K


def dataProcessor(vgg_face, cropped_images_directory):
    X = []
    Y = []
    person_rep = dict()
    person_folders = os.listdir(cropped_images_directory)
    for i, person in enumerate(person_folders):
        person_rep[i] = person
        image_names = os.listdir(cropped_images_directory + person + '/')
        for image_name in image_names:
            img = load_img(cropped_images_directory + person + '/' + image_name, target_size=(224, 224))
            img = img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            img_encode = vgg_face(img)
            X.append(np.squeeze(K.eval(img_encode)).tolist())
            Y.append(i)
    return X, Y


def encodeFaces(vgg_face, cropped__training_images_directory, cropped__testing_images_directory):
    x_train, y_train = dataProcessor(vgg_face, cropped__training_images_directory)
    x_test, y_test = dataProcessor(vgg_face, cropped__testing_images_directory)
    return x_train, y_train, x_test, y_test

