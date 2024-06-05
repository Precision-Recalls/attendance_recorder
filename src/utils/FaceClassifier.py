import tensorflow as tf
from tensorflow.keras.layers import Dropout, Activation, Dense, BatchNormalization
from tensorflow.keras.models import Sequential


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
