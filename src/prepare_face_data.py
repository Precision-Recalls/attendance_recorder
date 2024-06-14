from src.utils.ExtractFace import process_image_files
from src.utils.FaceClassifier import train_face_classifier
from src.utils.FaceEmbeddingModel import load_face_embedding_model
from src.utils.ImageAugmentor import create_images
from src.utils.TrainingDataProcessor import encodeFaces
from src.utils.config_loader import load_config

# Load the configuration
config = load_config('config.ini')

def prepareFaceData():
    
    # Extract values from the config
    training_data_directory = config['paths']['training_data_directory']
    training_cropped_data_directory = config['paths']['training_cropped_data_directory']
    test_data_directory = config['paths']['test_data_directory']
    testing_cropped_data_directory = config['paths']['testing_cropped_data_directory']
    vgg_face_model_weight_file_path = config['paths']['vgg_face_model_weight_file_path']
    cropped_training_images_directory = config['paths']['cropped_training_images_directory']
    cropped_testing_images_directory = config['paths']['cropped_testing_images_directory']
    

    # Crop faces from training and test images
    process_image_files(training_data_directory, training_cropped_data_directory)
    print('Training images got cropped and saved successfully!')

    process_image_files(test_data_directory, testing_cropped_data_directory)
    print('Testing images got cropped and saved successfully!')

    # Get embedding of all the faces
    vgg_face_model = load_face_embedding_model(vgg_face_model_weight_file_path)
    x_train, y_train, x_test, y_test = encodeFaces(vgg_face_model, cropped_training_images_directory, cropped_testing_images_directory)

    # Train the face classifier model
    train_face_classifier(x_train, y_train)
    print('Face classifier model trained successfully!')

if __name__ == '__main__':
    prepareFaceData()
