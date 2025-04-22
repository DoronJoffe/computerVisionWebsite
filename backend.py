from flask import Flask, request, send_file, render_template
import cv2
import numpy as np
import io
from PIL import Image

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # Serves the HTML

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['image']
    img = Image.open(file.stream)
    img = np.array(img)

    # Dummy object detection (draw a red box)
    cv2.rectangle(img, (50, 50), (200, 200), (255, 0, 0), 2)

    # Convert back to JPEG
    _, jpeg = cv2.imencode('.jpg', img)
    return send_file(io.BytesIO(jpeg.tobytes()), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
