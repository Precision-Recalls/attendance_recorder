import os
import pickle
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import tensorflow.keras.backend as K
import yt_dlp
from PIL import Image
from keras.models import load_model
from mtcnn import MTCNN
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from yt_dlp.utils import download_range_func

from src.utils.FaceEmbeddingModel import load_face_embedding_model
from src.utils.RecordAttendance import takeAttendance
from src.utils.config_loader import load_config

# Load the configuration
config = load_config('config.ini')

# Extract values from the config
vgg_face_model_weight_file_path = config['paths']['vgg_face_model_weight_file_path']
classifier_model_path = config['paths']['classifier_model_path']
person_rep_path = config['paths']['person_rep_path']

detector = MTCNN(steps_threshold=[0.80, 0.85, 0.9])
vgg_face_model = load_face_embedding_model(vgg_face_model_weight_file_path)
classifier_model = load_model(classifier_model_path)
person_rep = pickle.load(open(person_rep_path, 'rb'))

def download_video(video_link):
    start_time = 0  # accepts decimal value like 2.3
    end_time = 50

    yt_opts = {
        'verbose': True,
        'download_ranges': download_range_func(None, [(start_time, end_time)]),
        'force_keyframes_at_cuts': True,
    }

    with yt_dlp.YoutubeDL(yt_opts) as ydl:
        ydl.download(video_link)


def frame_processor(file_path, unknown_counter):
    # load image from file
    img = Image.open(file_path)
    if img is None or img.size is 0:
        print("Please check image path or some error occurred !")
    else:
        def process_face(pixels, result, unknown_counter, i):
            x1, y1, width, height = result['box']
            # bug fix
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height

            # extract the face
            face = pixels[y1:y2, x1:x2]
            # save crop image with person name as image name
            cropped_file_name = f'output/cropped_{i}.jpg'
            cv2.imwrite(cropped_file_name, face)

            # Get Embeddings
            crop_img = load_img(cropped_file_name, target_size=(224, 224))
            # target_size = (224, 224)
            # crop_img = np.array(Image.fromarray(face.astype(np.uint8)).resize(target_size)).astype('float32')
            crop_img = img_to_array(crop_img)
            crop_img = np.expand_dims(crop_img, axis=0)
            crop_img = preprocess_input(crop_img)
            img_encode = vgg_face_model(crop_img)

            # Make Predictions
            embed = K.eval(img_encode)
            person = classifier_model.predict(embed)
            name = person_rep.get(np.argmax(person))
            os.remove(cropped_file_name)

            cv2.rectangle(pixels, (x1, y1), (x2, y2), (0, 255, 0), 2)
            pixels = cv2.putText(pixels, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 0,
                                 cv2.LINE_AA)
            # Save images with bounding box,name and accuracy
            if name == 'unknown':
                name = name + f'{unknown_counter}'
                unknown_counter += 1
            attendance_time = takeAttendance(name)
            final_image_name = f'output/recorded_faces/{name}_{attendance_time}.jpg'
            cv2.imwrite(final_image_name, face)
            return unknown_counter

        # convert to RGB, if needed
        image = img.convert('RGB')
        # convert to array
        pixels = np.asarray(image)

        # detect faces in the image
        results = detector.detect_faces(pixels)
        # extract the bounding box from the first face
        if results:
            # Use ThreadPoolExecutor for concurrent processing
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(process_face, pixels, result, unknown_counter, i) for i, result in enumerate(results)]
                for future in futures:
                    unknown_counter = future.result()
    return unknown_counter

