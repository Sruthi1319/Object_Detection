# gradio_app.py  – YOLOv5 object detection (webcam or upload)

import torch, numpy as np, gradio as gr
from PIL import Image

# ➊  Load a YOLOv5 model — swap in your own .pt if you’ve trained one:
model = torch.hub.load("ultralytics/yolov5", "yolov5s", trust_repo=True)  # or path="runs/train/exp/weights/best.pt"

# ➋  Inference function (returns a PIL image with boxes drawn)
def detect_objects(img: Image.Image) -> Image.Image:
    result = model(np.array(img))         # run inference
    rendered = result.render()[0]         # draw boxes
    return Image.fromarray(rendered)

# ➌  Gradio UI
demo = gr.Interface(
    fn=detect_objects,
    inputs=gr.Image(
        sources=["webcam", "upload"],     # 👈 NEW: list form
        type="pil",
        label="Webcam capture or image upload",
    ),
    outputs=gr.Image(type="pil", label="Detected objects"),
    title="YOLOv5 Object Detection",
    description="Click the camera icon or browse for an image, then let YOLOv5 detect objects.",
)

if __name__ == "__main__":
    demo.launch()
