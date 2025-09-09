from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import cv2
import torch
import numpy as np
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

# Load YOLOv5 model
model = None

def load_model():
    global model
    try:
        # Load YOLOv5s model (you can change to yolov5m, yolov5l, yolov5x for better accuracy)
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def detect_objects(image_path):
    global model
    if model is None:
        return None, "Model not loaded"
    
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return None, "Could not load image"
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Run inference
        results = model(img_rgb)
        
        # Get predictions
        predictions = results.pandas().xyxy[0].to_dict(orient="records")
        
        # Draw bounding boxes
        img_with_boxes = img_rgb.copy()
        
        for pred in predictions:
            x1, y1, x2, y2 = int(pred['xmin']), int(pred['ymin']), int(pred['xmax']), int(pred['ymax'])
            confidence = pred['confidence']
            class_name = pred['name']
            
            if confidence > 0.5:  # Confidence threshold
                # Draw bounding box
                cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"{class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(img_with_boxes, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(img_with_boxes, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Save result image
        result_filename = 'result_' + os.path.basename(image_path)
        result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
        cv2.imwrite(result_path, cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR))
        
        # Filter predictions with confidence > 0.5
        filtered_predictions = [pred for pred in predictions if pred['confidence'] > 0.5]
        
        return {
            'predictions': filtered_predictions,
            'result_image': result_filename,
            'total_objects': len(filtered_predictions)
        }, None
        
    except Exception as e:
        return None, str(e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Perform object detection
        result, error = detect_objects(filepath)
        
        if error:
            return jsonify({'error': error})
        
        return jsonify({
            'success': True,
            'predictions': result['predictions'],
            'result_image': result['result_image'],
            'total_objects': result['total_objects']
        })
    
    return jsonify({'error': 'Invalid file type. Please upload an image file.'})

@app.route('/results/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

@app.route('/uploads/<filename>')
def original_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    print("Loading YOLO model...")
    load_model()
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)
