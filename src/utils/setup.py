from src.training.train import train_face_classifier
from src.utils.dataUtils import process_files, import_image_files
from src.utils.imageUtils import create_images, create_train_test_data


def setupService():
    # Import image files from the source folder of the client
    source_image_folder = ''
    image_destination_folder = 'data/images'
    import_image_files(source_image_folder, image_destination_folder)
    print('Images imported successfully!')
    # Create more images if we have lesser image per person
    num_of_images = 5
    create_images(image_destination_folder, num_of_images)
    print('Additional Images successfully created!')
    # ----------------------------------------------------------#
    # crop faces from training and test images
    training_data_directory = 'data/images/'
    training_cropped_data_directory = 'data/images_crop/'
    process_files(training_data_directory, training_cropped_data_directory)
    print('Training images got cropped and saved successfully!')

    test_data_directory = 'data/images_test/'
    testing_cropped_data_directory = 'data/images_test_crop/'
    process_files(test_data_directory, testing_cropped_data_directory)
    print('Testing images got cropped and saved successfully!')
    # --------------------------------------------------------------------------#
    # Get embedding of all the faces
    cropped_training_images_directory = 'data/images_crop/'
    cropped_testing_images_directory = 'data/images_test_crop/'
    x_train, y_train, x_test, y_test = create_train_test_data(cropped_training_images_directory,
                                                              cropped_testing_images_directory)
    print('Training and testing data got prepared!')
    # --------------------------------------------------------------------------------#
    # Train the face classifier model
    train_face_classifier(x_train, y_train)
    print('Face classifier model got trained!')
    # ---------------------------------------------------------------------------------#
