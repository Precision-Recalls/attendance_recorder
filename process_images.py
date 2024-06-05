# Import required libraries
import datetime
import time

from src.video_utility_functions import frame_processor

start_time = time.time()
next_start = datetime.datetime(2024, 6, 4, 8, 0, 0)
unknown_counter = 0
file_paths = ['DSC_3240.JPG']  # ,'data/images/ben_afflek/ben_afflek_aug_0_25.jpg'
time_array = []

for i in range(5):
    while True:
        dtn = datetime.datetime.now()
        if dtn >= next_start:
            next_start += datetime.timedelta(1)  # 1 day
            unknown_counter = 0

        for file_path in file_paths:
            unknown_counter = frame_processor(file_path, unknown_counter)
        break
    time_array.append(time.time() - start_time)
    start_time = time.time()
print(f'total time taken is :- {time_array}')
