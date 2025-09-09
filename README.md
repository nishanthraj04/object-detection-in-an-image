Object Detection with YOLOv5 and Flask

This project is a simple Flask web application for performing real-time object detection using YOLOv5.
Users can upload an image, and the app will return a result image with bounding boxes and labels for detected objects.

📂 Project Structure
.
├── app.py               # Flask backend
├── yolov5s.pt           # YOLOv5 model weights (custom or pretrained)
├── templates/
│   └── index.html       # Frontend HTML file
├── uploads/             # Uploaded images
├── results/             # Processed images with detections
└── static/              # Static assets (CSS, JS, etc.)

⚙️ Requirements

Make sure you have Python 3.8+ installed.
Install dependencies with:

pip install -r requirements.txt


requirements.txt (example):

flask
torch
torchvision
opencv-python
numpy
pillow

🚀 Running the App

Clone the repository:

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name


Start the Flask server:

python app.py


Open your browser and go to:

http://127.0.0.1:5000

🧠 Model

By default, the app loads yolov5s.pt from the project folder.

You can replace it with your own custom trained YOLOv5 model.

Model loading code (in app.py):

model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt')

📸 Usage

Upload an image via the web UI.

YOLOv5 runs inference on the image.

The output image with bounding boxes and labels will be displayed.

You’ll also get detection results in JSON format, including:

Object class

Confidence score

Bounding box coordinates

📊 Example Output

Input Image → uploaded via browser

Result Image → saved in results/ with bounding boxes drawn

JSON Response:

{
  "success": true,
  "predictions": [
    {"xmin": 34, "ymin": 50, "xmax": 200, "ymax": 300, "confidence": 0.87, "name": "person"}
  ],
  "result_image": "result_sample.jpg",
  "total_objects": 1
}

📌 Notes

Maximum upload size is 16MB.

Supported file formats: jpg, jpeg, png, bmp, gif.

Results are saved in the results/ folder.

🛠️ Future Improvements

Add support for video streams
