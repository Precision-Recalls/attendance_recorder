# Tensorflow version == 2.0.0
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Flatten, Activation, Dense, BatchNormalization
from tensorflow.keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D
from tensorflow.keras.models import Sequential, Model


def faceEmbeddingModel():
    # Define VGG_FACE_MODEL architecture
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))

    return model


def load_face_embedding_model(model_weight_file):
    model = faceEmbeddingModel()
    # Load VGG Face model weights
    model.load_weights(model_weight_file)
    # Remove last Softmax layer and get model upto last flatten layer #with outputs 2622 units
    vgg_face = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
    return vgg_face


def faceClassificationModel():
    # Softmax regressor to classify images based on encoding
    classifier_model = Sequential()
    classifier_model.add(Dense(units=100,  kernel_initializer='glorot_uniform'))
    classifier_model.add(BatchNormalization())
    classifier_model.add(Activation('tanh'))
    classifier_model.add(Dropout(0.3))
    classifier_model.add(Dense(units=10, kernel_initializer='glorot_uniform'))
    classifier_model.add(BatchNormalization())
    classifier_model.add(Activation('tanh'))
    classifier_model.add(Dropout(0.2))
    classifier_model.add(Dense(units=6, kernel_initializer='he_uniform'))
    classifier_model.add(Activation('softmax'))
    return classifier_model


def train_face_classifier(x_train, y_train):
    face_clf = faceClassificationModel()
    face_clf.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer='adam'
                     , metrics=['accuracy'])
    # fit the keras model on the dataset
    face_clf.fit(x_train, y_train, epochs=100, batch_size=10)
    face_clf.save('assets/models/face_classifier.h5')
    # evaluate the keras model
    _, accuracy = face_clf.evaluate(x_train, y_train)
    print('Accuracy: %.2f' % (accuracy * 100))
