import cv2 as cv
import numpy as np
import os
from time import time

# change the working directory to the folder this script is in.
# Doing this since the images from the video will be captured in this folder
os.chdir( os.path.dirname( os.path.abspath(__file__) ) )
cascade_directory = os.path.join( os.path.dirname(__file__), 
                                 'cascade')

loop_time = time()
counter = 0
video_capture = cv.VideoCapture("./17 December 2024.mp4")
while (True):
    # Capture frame-by-frame
    my_ret, my_frame = video_capture.read()
    # if frame is read correctly ret is True
    if not my_ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # print('FPS {}'.format(1 / (time() - loop_time)))
    # print('FPS {}'.format(int(loop_time)))
    print('counter{}'.format(counter))
    loop_time = time()
    counter = counter + 1

    # Display the resulting frame
    cv.imshow('frame', my_frame)
    if cv.waitKey(1) == ord('s'):
        # save frame to directory
        output_fileName = os.path.join( cascade_directory, 
                                        'output' + str(counter) + '.png') 
        cv.imwrite(output_fileName,my_frame)
    if cv.waitKey(1) == ord('q'):
        break

# when everything done, release the capture
video_capture.release()
cv.destroyAllWindows()
print("Done ...")




