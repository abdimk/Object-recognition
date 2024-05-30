import cv2
import numpy as np
import os

# Threshold to detect object
thres = 0.70
nms_threshold = 0.5

# Path to the image file
#image_path = r'D:\AI\cat.jpg'
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not read image file.")
    exit()

# Import the class names
classNames = []
classFile = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'config_files', 'coco.names'))

# Read object classes
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Import the config and weights file
configPath = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'config_files', 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'))
weightsPath = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'config_files', 'frozen_inference_graph.pb'))

# Set relevant parameters
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Create a named window and set the window size
cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Output", 900, 900)

# Resize frame to speed up detection
resized_image = cv2.resize(image, (320, 320))

# Detect objects in the frame
classIds, confs, bbox = net.detect(resized_image, confThreshold=thres)

if len(classIds) != 0:
    for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
        # Adjust the box coordinates based on the resize scale
        scale_x = image.shape[1] / 320
        scale_y = image.shape[0] / 320
        box = [int(box[0] * scale_x), int(box[1] * scale_y), int(box[2] * scale_x), int(box[3] * scale_y)]

        cv2.rectangle(image, box, color=(0, 255, 0), thickness=1)
        cv2.putText(image, classNames[classId - 1], (box[0] + 10, box[1] + 30),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

# Show output
cv2.imshow("Output", image)

# Wait for key press to close
cv2.waitKey(0)

# Release resources
cv2.destroyAllWindows()
