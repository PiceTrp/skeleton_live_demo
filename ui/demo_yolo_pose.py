import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import gradio as gr
from models.yolo_pose import inference_model

# Define the Gradio interface
def demo_ui_pose():
    demo_interface = gr.Interface(
        fn=inference_model,  # The function to process the input
        inputs=gr.Image(type="numpy"),  # Input component for image upload
        outputs=gr.Image(type="numpy"),  # Output component to display the processed image
        title="YOLO Pose Estimation",
        description="Upload an image to see the pose estimation results."
    )
    demo_interface.launch()


# Launch the interface
if __name__ == "__main__":
    pass
    # interface.launch()