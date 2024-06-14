import logging
from src.training.train import train_face_classifier
from src.utils.dataUtils import process_files, import_image_files
from src.utils.imageUtils import create_images, create_train_test_data
from src.utils.configUtils import load_config

# Initialize logging
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
        num_of_images = config['parameters']['num_of_images']
        training_data_directory = config['paths']['training_data_directory']
        training_cropped_data_directory = config['paths']['training_cropped_data_directory']
        test_data_directory = config['paths']['test_data_directory']
        testing_cropped_data_directory = config['paths']['testing_cropped_data_directory']
        classifier_model_path = config['paths']['classifier_model_path']
        person_rep_path = config['paths']['person_rep_path']

        # Setup service
        setup_service(source_image_folder, image_destination_folder, num_of_images,
                      training_data_directory, training_cropped_data_directory,
                      test_data_directory, testing_cropped_data_directory,
                      classifier_model_path, person_rep_path)

    except Exception as e:
        logger.error(f"Error occurred: {e}")


def setup_service(source_image_folder, image_destination_folder, num_of_images,
                  training_data_directory, training_cropped_data_directory,
                  test_data_directory, testing_cropped_data_directory,
                  classifier_model_path, person_rep_path):
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
                                                                  testing_cropped_data_directory)
        logger.info('Training and testing data prepared successfully!')

        # Train the face classifier model
        train_face_classifier(x_train, y_train, classifier_model_path)
        logger.info('Face classifier model trained successfully!')

    except Exception as e:
        logger.error(f"Error in setup_service: {e}")
        raise


if __name__ == "__main__":
    main()
