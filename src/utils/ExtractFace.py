import os
import pickle
from os import listdir
from os.path import isdir

import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from PIL import Image


def extract_face(input_filename, output_filename):
    # load image from file
    image = Image.open(input_filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = np.asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    if results:
        x1, y1, width, height = results[0]['box']
        # bug fix
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = pixels[y1:y2, x1:x2]
        # save crop image with person name as image name
        cv2.imwrite(output_filename, face)


def process_image_files(input_directory, output_directory):
    # enumerate folders, on per class
    person_rep = dict()
    for i, subdir in enumerate(listdir(input_directory)):
        if i not in person_rep:
            person_rep[i] = subdir
        # path
        input_file_path = input_directory + subdir + '/'
        # skip any files that might be in the dir
        if not isdir(input_file_path):
            continue
        count = 0
        final_output_directory = output_directory + subdir + '/'
        if not os.path.exists(final_output_directory):
            os.makedirs(final_output_directory)

        for filename in listdir(input_file_path):
            input_filename = input_file_path + filename
            output_filename = final_output_directory + f'{subdir}_{count}.jpg'
            # get face
            extract_face(input_filename, output_filename)
            count += 1
    pickle.dump(person_rep, open(r'assets/names_mapping.pkl', 'wb'))

