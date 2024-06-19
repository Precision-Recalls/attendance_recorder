# Import required libraries
import datetime
import logging
import time

from src.utils.imageUtils import image_processor

logging.basicConfig(level=logging.CRITICAL)
logger = logging.getLogger(__name__)

# Initialize start time
start_time = time.time()

# Set initial reset time
next_start = datetime.datetime(2024, 6, 4, 8, 0, 0)

# Define file paths
file_paths = [
    'test/jim_corbett.jpeg',
    'test/th.jpg'
]

time_array = []

try:
    # Iterate for 5 times
    for i in range(5):
        # Process images
        image_processor(file_paths)
        # Measure elapsed time and append to time_array
        elapsed_time = time.time() - start_time
        time_array.append(elapsed_time)
        # Update start_time for next iteration
        start_time += elapsed_time
    logger.info(f'Total time taken is: {time_array}')
except Exception as e:
    logger.error(f"There is some error in processing :- {e}")
