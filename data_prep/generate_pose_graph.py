import os
import sys

import cv2

def video_to_images(video_path, output_path, output_name=None):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    success = True
    while True:
        success, image = vidcap.read()
        if not success :
            break
        cv2.imwrite("%s/%s-frame%d.jpg" % (output_path, str(output_name), count), image)  # save frame as JPEG file
        if cv2.waitKey(10) == 27:  # exit if Escape is hit
            break
        count += 1

def videos_to_images(video_folder_path, output_path):
    for file in os.listdir(video_folder_path):
        print("processing %s" % file)
        video_to_images(os.path.join(video_folder_path, file), output_path, os.path.splitext(file)[0])

if __name__ == '__main__':
    videos_to_images(sys.argv[1], sys.argv[2])