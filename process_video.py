# importing the necessary libraries
import os
import time
import cv2

from src.utils.imageUtils import process_face

start_time = time.time()

# Creating a VideoCapture object to read the video
cap = cv2.VideoCapture(r"test/Live by Night Official Trailer 1 (2016) - Ben Affleck Movie.mp4")
frame_count = 0

try:
    # Loop until the end of the video
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Check if frame is read correctly
        if not ret:
            break

        if frame_count % 30 == 0:
            temp_image_file_name = f'frame_{frame_count}.jpg'
            cv2.imwrite(temp_image_file_name, frame)
            process_face(temp_image_file_name)
            os.remove(temp_image_file_name)
            # Display the frame
            # cv2.imshow("Face Detection Tutorial", frame)
        frame_count += 1
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print(f"Total frames processed: {frame_count}")
except Exception as e:
    print(f"Error occurred: {e}")

finally:
    # Release the video capture object
    cap.release()

    # Close all windows
    cv2.destroyAllWindows()

    # Calculate and print total processing time
    end_time = time.time()
    print(f'Total processing time: {end_time - start_time} seconds')
