import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from ultralytics import YOLO
import cv2
from cfg import *

# Load the YOLO model
model = YOLO(conf.MODEL_PATH)

def inference_camera():
    results = model(source=0, show=True, conf=0.3)
    return results

def inference_model(image):
    # Perform inference
    results = model(image)
    annotated_image = results[0].plot()
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB) # Convert the image to RGB (Gradio expects RGB images)
    return annotated_image_rgb

def export_to_onnx(model):
    model.export(format="onnx")


if __name__ == "__main__":
    # work_dir = os.getcwd()
    # model = YOLO("./models/yolo11n-pose.pt")
    # results = model(source="./image.png", show=True, conf=0.3)
    # print(results)

    # model.export(format="onnx")
    # inference_model()
    pass