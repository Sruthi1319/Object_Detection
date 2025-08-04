import torch
import cv2
import tkinter as tk
from PIL import Image, ImageTk

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)

# Create main Tkinter window
window = tk.Tk()
window.title("YOLOv5 Live Detection")
window.geometry("800x600")

# Label to display the video
video_label = tk.Label(window)
video_label.pack()

# Capture video from webcam
cap = cv2.VideoCapture(0)

def update_frame():
    ret, frame = cap.read()
    if not ret:
        return

    # Run YOLOv5 detection
    results = model(frame)
    results.render()  # In-place render with bounding boxes

    # Convert result image to Tkinter-compatible format
    frame_rgb = cv2.cvtColor(results.ims[0], cv2.COLOR_RGB2BGR)
    img_pil = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img_pil)

    # Update the label with the image
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    # Call this function again after 10ms
    window.after(10, update_frame)

# Start updating frames
update_frame()

# Run the GUI loop
window.mainloop()

# Release resources on close
cap.release()
cv2.destroyAllWindows()
