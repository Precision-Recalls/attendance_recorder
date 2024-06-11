import os
import pickle
import shutil
from os import listdir
from os.path import isdir

from src.utils.imageUtils import detect_faces_in_images, crop_faces


def check_or_create_directory(directory_name):
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    print('Directory got created!')


def move_image_files(image_folder, obj_type, image_destination_folder):
    for i, filename in enumerate(listdir(image_folder), 1):
        orig_file_path = os.path.join(image_folder, filename)
        if obj_type == 'file':
            person_name = filename.split('_')[0]
        else:
            image_folder = image_folder.replace('\\', "/")
            person_name = image_folder.split('/')[-1]
        # 20% images will be copied into test folder for validation
        if i % 5 == 0:
            image_destination_folder = image_destination_folder + '_test'
        image_final_destination = image_destination_folder + '/' + person_name + '/'
        check_or_create_directory(image_final_destination)
        shutil.copy(orig_file_path, image_final_destination)


def import_image_files(source_image_folder, image_destination_folder):
    for i, obj in enumerate(listdir(source_image_folder)):
        if isdir(os.path.join(source_image_folder, obj)):
            obj_type = 'dir'
            source_image_folder = os.path.join(source_image_folder, obj)
            # skip any files that might be in the dir
            if not isdir(source_image_folder):
                continue
            move_image_files(source_image_folder, obj_type, image_destination_folder)
        else:
            obj_type = 'file'
            move_image_files(source_image_folder, obj_type, image_destination_folder)


def process_files(input_directory, output_directory):
    check_or_create_directory(input_directory)
    check_or_create_directory(output_directory)
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
        check_or_create_directory(final_output_directory)

        for filename in listdir(input_file_path):
            input_filename = input_file_path + filename
            output_filename = final_output_directory + f'{subdir}_{count}.jpg'
            # get face
            pixels, results = detect_faces_in_images(input_filename)
            # extract the bounding box from the first face
            if results:
                crop_faces(pixels, results[0], output_filename)
            count += 1
    pickle.dump(person_rep, open(r'assets/names_mapping.pkl', 'wb'))
