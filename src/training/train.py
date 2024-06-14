import tensorflow as tf
from tensorflow.keras.layers import Dropout, Flatten, Activation, Dense, BatchNormalization
from tensorflow.keras.layers import ZeroPadding2D, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for hyper parameters
EPOCHS = 100
BATCH_SIZE = 10
DROPOUT_RATE_1 = 0.5
DROPOUT_RATE_2 = 0.2
DROPOUT_RATE_3 = 0.3

def faceEmbeddingModel():
    # Define VGG_FACE_MODEL architecture
    model = Sequential([
        ZeroPadding2D((1, 1), input_shape=(224, 224, 3)),
        Conv2D(64, (3, 3), activation='relu'),
        ZeroPadding2D((1, 1)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2), strides=(2, 2)),
        ZeroPadding2D((1, 1)),
        Conv2D(128, (3, 3), activation='relu'),
        ZeroPadding2D((1, 1)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2), strides=(2, 2)),
        ZeroPadding2D((1, 1)),
        Conv2D(256, (3, 3), activation='relu'),
        ZeroPadding2D((1, 1)),
        Conv2D(256, (3, 3), activation='relu'),
        ZeroPadding2D((1, 1)),
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D((2, 2), strides=(2, 2)),
        ZeroPadding2D((1, 1)),
        Conv2D(512, (3, 3), activation='relu'),
        ZeroPadding2D((1, 1)),
        Conv2D(512, (3, 3), activation='relu'),
        ZeroPadding2D((1, 1)),
        Conv2D(512, (3, 3), activation='relu'),
        MaxPooling2D((2, 2), strides=(2, 2)),
        ZeroPadding2D((1, 1)),
        Conv2D(512, (3, 3), activation='relu'),
        ZeroPadding2D((1, 1)),
        Conv2D(512, (3, 3), activation='relu'),
        ZeroPadding2D((1, 1)),
        Conv2D(512, (3, 3), activation='relu'),
        MaxPooling2D((2, 2), strides=(2, 2)),
        Conv2D(4096, (7, 7), activation='relu'),
        Dropout(DROPOUT_RATE_1),
        Conv2D(4096, (1, 1), activation='relu'),
        Dropout(DROPOUT_RATE_1),
        Conv2D(2622, (1, 1)),
        Flatten(),
        Activation('softmax')
    ])
    return model

def load_face_embedding_model(model_weight_file):
    model = faceEmbeddingModel()
    # Load VGG Face model weights
    model.load_weights(model_weight_file)
    # Remove last Softmax layer and get model up to last flatten layer with outputs 2622 units
    vgg_face = Model(inputs=model.input, outputs=model.layers[-2].output)
    return vgg_face

def faceClassificationModel():
    # Softmax regressor to classify images based on encoding
    classifier_model = Sequential([
        Dense(units=100, kernel_initializer='glorot_uniform', input_shape=(2622,)),
        BatchNormalization(),
        Activation('tanh'),
        Dropout(DROPOUT_RATE_3),
        Dense(units=10, kernel_initializer='glorot_uniform'),
        BatchNormalization(),
        Activation('tanh'),
        Dropout(DROPOUT_RATE_2),
        Dense(units=6, kernel_initializer='he_uniform'),
        Activation('softmax')
    ])
    return classifier_model

def train_face_classifier(x_train, y_train, classifier_model_path):
    face_clf = faceClassificationModel()
    face_clf.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])

    # Callbacks for early stopping and saving the best model
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(classifier_model_path, save_best_only=True, monitor='val_loss')

    # Fit the Keras model on the dataset
    face_clf.fit(
        x_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=[early_stopping, model_checkpoint]
    )

    # Evaluate the Keras model
    _, accuracy = face_clf.evaluate(x_train, y_train)
    logger.info(f'Accuracy: {accuracy * 100:.2f}%')
