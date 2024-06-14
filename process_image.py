# Import required libraries
import datetime
import time

from src.utils.imageUtils import image_processor
from src.utils.commonUtils import get_next_reset_time

# Initialize start time
start_time = time.time()

# Set initial reset time
next_start = datetime.datetime(2024, 6, 4, 8, 0, 0)

# Define file paths
file_paths = [
    'data/images/ben_afflek/ben_afflek_aug_0_25.jpg',
    'test/DSC_3240.JPG',
    'test/test.jpg'
]

time_array = []

# Iterate for 5 times
for i in range(5):
    # Calculate next reset time
    next_start = get_next_reset_time(next_start)

    # Process images
    image_processor(file_paths)

    # Measure elapsed time and append to time_array
    elapsed_time = time.time() - start_time
    time_array.append(elapsed_time)

    # Update start_time for next iteration
    start_time += elapsed_time

print(f'Total time taken is: {time_array}')
