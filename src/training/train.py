import pickle

import tensorflow as tf
from tensorflow.keras.layers import Dropout, Flatten, Activation, Dense, BatchNormalization
from tensorflow.keras.layers import ZeroPadding2D, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
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
    try:
        model = faceEmbeddingModel()
        # Load VGG Face model weights
        model.load_weights(model_weight_file)
        # Remove last Softmax layer and get model up to last flatten layer with outputs 2622 units
        vgg_face = Model(inputs=model.input, outputs=model.layers[-2].output)
        return vgg_face
    except Exception as e:
        logger.error(f"There is some issue with vgg_face model loading :- {e}")


def faceClassificationModel(num_classes):
    # Softmax regressor to classify images based on encoding
    classifier_model = Sequential([
        # First layer: Dense layer with 512 units, Xavier/Glorot uniform initializer, and input shape of 2622
        Dense(units=512, kernel_initializer='glorot_uniform', input_shape=(2622,)),
        BatchNormalization(),
        Activation('relu'),
        Dropout(DROPOUT_RATE_3),

        # Second layer: Dense layer with 256 units, Xavier/Glorot uniform initializer
        Dense(units=256, kernel_initializer='glorot_uniform'),
        BatchNormalization(),
        Activation('relu'),
        Dropout(DROPOUT_RATE_2),

        # Third layer: Dense layer with 128 units, Xavier/Glorot uniform initializer
        Dense(units=128, kernel_initializer='glorot_uniform'),
        BatchNormalization(),
        Activation('relu'),
        Dropout(DROPOUT_RATE_1),

        # Fourth layer: Dense layer with 64 units, Xavier/Glorot uniform initializer
        Dense(units=64, kernel_initializer='glorot_uniform'),
        BatchNormalization(),
        Activation('relu'),
        Dropout(DROPOUT_RATE_1),

        # Output layer: Dense layer with 9 units, He uniform initializer
        Dense(units=num_classes, kernel_initializer='he_uniform'),
        Activation('softmax')
    ])
    return classifier_model


def compile_face_classifier():
    face_clf = faceClassificationModel(num_classes=9)

    # Compile the model
    optimizer = Adam(learning_rate=0.001)
    face_clf.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Summary of the model
    face_clf.summary()
    return face_clf


def train_face_classifier(x_train, y_train, x_val, y_val, classifier_model_path):
    try:
        model = compile_face_classifier()

        for epoch in range(EPOCHS):
            print(f'Epoch {epoch + 1}/{EPOCHS}')

            # Train the model on the training data_v2
            model.fit(
                x_train, y_train,
                batch_size=BATCH_SIZE,
                verbose=1
            )

            # Evaluate the model on the validation data_v2
            val_loss, val_accuracy = model.evaluate(x_val, y_val, verbose=0)
            print(f'Validation loss: {val_loss:.4f} - Validation accuracy: {val_accuracy:.4f}')
        # Save the model
        model.save(classifier_model_path)
        print(f"Model saved to {classifier_model_path}")
    except Exception as e:
        logger.error(f"There is some error in face classifier training :- {e}")
