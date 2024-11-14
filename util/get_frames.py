import cv2
import os

for file in os.listdir("C:\\Users\\simonguo\\Downloads\\openpose-1.6.0-binaries-win64-only_cpu-flir-3d\\openpose\\360"):
    print(file)
    filename = os.fsdecode(file).split(".")[0]
    vidcap = cv2.VideoCapture(file)
    success,image = vidcap.read()
    if not success:
        print("failed to load: ", file)
        break
    count = 0
    while success:
      cv2.imwrite("%s-frame%d.png" % (filename, count), image)     # save frame as JPEG file
      success,image = vidcap.read()
      print('Read a new frame: ', success)
      count += 1