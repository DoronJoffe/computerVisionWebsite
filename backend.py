from flask import Flask, request, send_file, render_template
import cv2
import numpy as np
import io
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # You can use 'yolov8s.pt' or your custom model

@app.route('/')
def home():
    return render_template('index.html')  # Serves your HTML file from templates/

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['image']
    img = Image.open(file.stream).convert("RGB")
    img = np.array(img)

    # Run detection
    results = model(img)

    # Use .plot() to draw boxes on the image
    frame = results[0].plot()  # This returns a NumPy array

    # Convert to JPEG and send back
    _, jpeg = cv2.imencode('.jpg', frame)
    return send_file(io.BytesIO(jpeg.tobytes()), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=10000)
