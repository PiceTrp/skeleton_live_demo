import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from ultralytics import YOLO
import cv2
import numpy as np
from cfg import *

# Load the YOLO model
model = YOLO(conf.MODEL_PATH)

def inference_camera_YOLO():
    results = model(source=0, show=True, conf=0.3)
    return results


def inference_camera():
    # Initialize the video capture from the default camera
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("YOLO Output", cv2.WINDOW_NORMAL) # Create a named window with the ability to resize
    # # Get screen dimensions
    # screen_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # screen_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # # Set the window size
    # cv2.resizeWindow("YOLO Output", screen_width, screen_height)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, conf=0.3)
        frame = results[0].plot()
        cv2.imshow("YOLO Output", frame)

        # Set the window to a specific size or fullscreen
        # For fullscreen, uncomment the following line:
        cv2.setWindowProperty("YOLO Output", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()


def inference_model(image):
    # Perform inference
    results = model(image)
    annotated_image = results[0].plot()
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB) # Convert the image to RGB (Gradio expects RGB images)
    return annotated_image_rgb


if __name__ == "__main__":
    # work_dir = os.getcwd()
    # model = YOLO("./models/yolo11n-pose.pt")
    # results = model(source="./image.png", show=True, conf=0.3)
    # print(results)

    # model.export(format="onnx")
    # inference_model()
    pass