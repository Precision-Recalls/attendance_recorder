from src.utils.ExtractFace import process_image_files
from src.utils.FaceClassifier import train_face_classifier
from src.utils.FaceEmbeddingModel import load_face_embedding_model
from src.utils.ImageAugmentor import create_images
from src.utils.TrainingDataProcessor import encodeFaces


def prepareFaceData():
    # # Create more images if we have lesser image per person
    # input_image_directory = 'data/images/'
    # num_of_images = 5
    # create_images(input_image_directory, num_of_images)
    # print('Images successfully created!')
    # # ----------------------------------------------------------#

    # crop faces from training and test images
    training_data_directory = 'data/images/'
    training_cropped_data_directory = 'data/images_crop/'
    process_image_files(training_data_directory, training_cropped_data_directory)
    print('Training images got cropped and saved successfully!')

    test_data_directory = 'data/images_test/'
    testing_cropped_data_directory = 'data/images_test_crop/'
    process_image_files(test_data_directory, testing_cropped_data_directory)
    print('Testing images got cropped and saved successfully!')
    # --------------------------------------------------------------------------#

    # Get embedding of all the faces
    vgg_face_model_weight_file_path = 'assets/models/vgg_face_weights.h5'
    vgg_face_model = load_face_embedding_model(vgg_face_model_weight_file_path)
    cropped__training_images_directory = 'data/images_crop/'
    cropped__testing_images_directory = 'data/images_test_crop/'
    x_train, y_train, x_test, y_test = encodeFaces(vgg_face_model, cropped__training_images_directory,
                                                   cropped__testing_images_directory)
    # --------------------------------------------------------------------------------#

    # Train the face classifier model
    train_face_classifier(x_train, y_train)
    # ---------------------------------------------------------------------------------#
