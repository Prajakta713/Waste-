from flask import Flask, render_template, request, redirect, url_for
import torch
import cv2
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Define the bin categories and colors
BIN_CATEGORIES = {
    'Plastic': {'color': 'Red', 'objects': ['Plastic bag', 'bottle', 'plastic containers']},
    'Cans and Tetrapacks': {'color': 'Yellow', 'objects': ['cans', 'tetrapacks']},
    'Paper and Cardboard': {'color': 'Blue', 'objects': ['paper', 'book', 'cardboard']},
    'Glass': {'color': 'Green', 'objects': ['glass', 'glass bottles', 'glass cups', 'mirror']},
    'E-waste': {'color': 'Black', 'objects': ['cell phone', 'laptop', 'charger', 'wires', 'earbuds', 'earphones', 'computer']} ,
    'Human': {'color': 'not racist', 'objects': ['person', 'human']}
}

def get_bin_info(object_name):
    """
    Get the bin category and color based on the detected object.
    """
    for category, info in BIN_CATEGORIES.items():
        if object_name in info['objects']:
            return category, info['color']
    return 'Unknown', 'Unknown'


# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

UPLOAD_FOLDER = 'static/uploads/'

# Ensure the directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allow only specific file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
    
        try:
            # Perform object detection
            img = cv2.imread(filepath)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = model(img_rgb)
            
            # Render the results on the image
            output_img = results.render()[0]
            output_filename = 'detected_' + filename
            output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            cv2.imwrite(output_filepath, output_img)

            # Get results as a pandas DataFrame
            detections = results.pandas().xyxy[0]
            
            results_list = []
            for _, row in detections.iterrows():
                object_name = row['name']
                bin_category, bin_color = get_bin_info(object_name)  # Get bin category and color
                
                results_list.append({
                    'bin_category': f"{bin_category} ({bin_color})",  # Include color in category
                    'location': (row['xmin'], row['ymin'])
                })

            return render_template('result.html', results=results_list, image_url='/static/uploads/' + output_filename)

        except Exception as e:
            return f"Error processing image with YOLOv5: {e}"
    else:
        return 'Invalid file type'


if __name__ == '__main__':
    app.run(debug=True)
