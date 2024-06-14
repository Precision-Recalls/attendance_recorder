import os
import pickle
import shutil
import logging
from os import listdir
from os.path import isdir, join
from src.utils.imageUtils import detect_faces_in_images, crop_faces

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_or_create_directory(directory_name):
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
        logger.info(f'Directory {directory_name} created!')


def move_image_files(image_folder, obj_type, image_destination_folder):
    for i, filename in enumerate(listdir(image_folder), 1):
        orig_file_path = join(image_folder, filename)
        person_name = filename.split('_')[0] if obj_type == 'file' else os.path.basename(image_folder)

        if i % 5 == 0:
            destination_folder = f'{image_destination_folder}_test'
        else:
            destination_folder = image_destination_folder

        image_final_destination = join(destination_folder, person_name)
        check_or_create_directory(image_final_destination)
        shutil.copy(orig_file_path, image_final_destination)
        logger.debug(f'Copied {orig_file_path} to {image_final_destination}')


def import_image_files(source_image_folder, image_destination_folder):
    for obj in listdir(source_image_folder):
        obj_path = join(source_image_folder, obj)
        if isdir(obj_path):
            move_image_files(obj_path, 'dir', image_destination_folder)
        else:
            move_image_files(source_image_folder, 'file', image_destination_folder)


def process_files(input_directory, output_directory, person_mapping_filepath=''):
    check_or_create_directory(output_directory)

    person_rep = {}
    for i, subdir in enumerate(listdir(input_directory)):
        subdir_path = join(input_directory, subdir)
        if not isdir(subdir_path):
            continue

        person_rep[i] = subdir
        final_output_directory = join(output_directory, subdir)
        check_or_create_directory(final_output_directory)

        for count, filename in enumerate(listdir(subdir_path)):
            input_filename = join(subdir_path, filename)
            output_filename = join(final_output_directory, f'{subdir}_{count}.jpg')

            try:
                pixels, results = detect_faces_in_images(input_filename)
                if results:
                    crop_faces(pixels, results[0], output_filename)
                    logger.debug(f'Cropped face from {input_filename} to {output_filename}')
            except Exception as e:
                logger.error(f'Error processing file {input_filename}: {e}')

    if person_mapping_filepath:
        with open(person_mapping_filepath, 'wb') as f:
            pickle.dump(person_rep, f)
        logger.info(f'Person mapping saved to {person_mapping_filepath}')
