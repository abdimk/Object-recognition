import cv2
import numpy as np
import os

# Read in the image file
thres = 0.70  # Threshold to detect object
nms_threshold = 0.5  # NMS
path = r'/home/abdisa/Desktop/AI/oj/Piassa to 4Kilo , Addis Ababa Walking Tour 2023.mp4'
cap = cv2.VideoCapture(path)

cap.set(3, 500)  # Set width
cap.set(4, 600)  # Set height

# Import the class names
classNames = []
classFile = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'config_files', 'coco.names'))

# Read object classes
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Import the config and weights file
os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'config_files', 'coco.names'))
configPath = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'config_files', 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'))
weightsPath = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'config_files', 'frozen_inference_graph.pb'))  # Weights derived from training on large objects dataset

# Set relevant parameters
net = cv2.dnn_DetectionModel(weightsPath, configPath)

# These are some suggested settings from the tutorial, others are fine but this can be used as a baseline
net.setInputSize(320, 320)  #
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    # Read frame from the webcam
    success, image = cap.read()

    # Resize the frame to 400x400q
    image = cv2.resize(image, (650, 500))

    # Detect objects in the frame
    classIds, confs, bbox = net.detect(image, confThreshold=thres)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(image, box, color=(0, 255, 0), thickness=1)
            cv2.putText(image, classNames[classId - 1], (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

    # Show output
    cv2.imshow("Output", image)

    # Wait for key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
