# Import required libraries
import datetime
import time

from src.utils.imageUtils import image_processor
from src.utils.commonUtils import get_next_reset_time

start_time = time.time()
next_start = datetime.datetime(2024, 6, 4, 8, 0, 0)
file_paths = ['data/images/ben_afflek/ben_afflek_aug_0_25.jpg', 'test/DSC_3240.JPG','test/test.jpg']
time_array = []

for i in range(5):
    while True:
        next_start = get_next_reset_time(next_start)
        image_processor(file_paths)
        time_array.append(time.time() - start_time)
        start_time = time.time()
        break
print(f'total time taken is :- {time_array}')
