# from ultralytics import YOLO
# import gradio as gr
# from ui.demo_yolo_pose import demo_ui_pose
from demo.yolo_pose import inference_camera
import cProfile

def main():
    inference_camera()
    # demo_ui_pose()

if __name__ == "__main__":
    print("Hi")
    cProfile.run('main()')