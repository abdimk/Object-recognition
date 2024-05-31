import cv2
import numpy as np
import os

# Threshold to detect object
thres = 0.70
nms_threshold = 0.5

# Path to the video file
path = 'src/test_cat.mp4' #path to the file
cap = cv2.VideoCapture(path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Skip frames to speed up the video
frame_skip = 5  # Adjust this to skip more frames (e.g., 5 means process every 5th frame)

# Import the class names
classNames = []
# classFile = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'config_files', 'coco.names'))

classFile = r'src/coco.names'

# Read object classes
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Import the config and weights file
#configPath = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'config_files', 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'))
configPath = r'src/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
#weightsPath = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'config_files', 'frozen_inference_graph.pb'))
weightsPath = r'src/frozen_inference_graph.pb'
# Set relevant parameters
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

frame_count = 0

# Create a named window and set the window size
cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Output", 900, 900)

while True:
    # Read frame from the video file
    success, image = cap.read()
    if not success:
        print("Error: Could not read frame from video file.")
        break

    # Skip frames
    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

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
            
            cv2.rectangle(image, box, color=(255, 255, 255), thickness=2)
            cv2.putText(image,f"{classNames[classId - 1]} {str(round(confidence, 2))}", (box[0] + 10, box[1] + 19),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            #cv2.putText(image, , (box[0] + 10, box[1] + 30),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    # Show output
    cv2.imshow("Output", image)

    # Wait for key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
