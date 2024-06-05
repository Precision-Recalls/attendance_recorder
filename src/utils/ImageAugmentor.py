from os import listdir
from os.path import isdir

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing.image import load_img, img_to_array


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
                        save_prefix=image_prefix+'_aug', save_format="jpg")
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
