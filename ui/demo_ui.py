import gradio as gr

def greet(name):
    return "Hello " + name + "!"

def demo_webapp():
    demo = gr.Interface(fn=greet, inputs="text", outputs="text")
    demo.launch()