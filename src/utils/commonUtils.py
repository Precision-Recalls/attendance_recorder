import datetime
import logging

# Initialize logging
import os.path
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_next_reset_time(next_start):
    dtn = datetime.datetime.now()
    if dtn >= next_start:
        next_start += datetime.timedelta(days=1)
    return next_start


def get_pickle_obj(file_path):
    file_path = os.path.abspath(file_path)
    # Check if the person_rep_path file exists
    if os.path.isfile(file_path):
        try:
            with open(file_path, 'rb') as f:
                person_rep = pickle.load(f)
                logger.info("Person rep/encodings loaded successfully.")
            return person_rep
        except Exception as e:
            logger.error(f"Error loading person rep: {e}")
    else:
        logger.info(f"File does not exist: {file_path}")


def take_attendance(name, conf_obj):
    file_extension = conf_obj['attendance-parameters']['file_extension']
    attendance_file_name_prefix = conf_obj['attendance-parameters']['attendance_file_name_prefix']

    # Get the current date
    today = datetime.datetime.now().strftime('%d_%m_%Y')
    attendance_file_name = f'{attendance_file_name_prefix}_{today}{file_extension}'

    # Create the file if it doesn't exist
    with open(attendance_file_name, 'a') as f:
        pass

    # Read the CSV file and check if the name is already present
    try:
        with open(attendance_file_name, 'r') as f:
            data = f.readlines()
            names = {line.split(',')[0].strip() for line in data}
    except FileNotFoundError:
        names = set()

    # Check if the name is in the list 'names', if not then add it
    if name not in names:
        current_time = datetime.datetime.now().strftime('%H:%M:%S')
        with open(attendance_file_name, 'a') as f:
            f.write(f"{name}, {current_time}\n")
            logger.info(f"Added {name} to the attendance file at {current_time}")

    return today
