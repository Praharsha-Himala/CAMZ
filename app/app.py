import os
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from datetime import datetime
from utils import process_video_with_model, write_predictions_to_csv, CNNModel, load_checkpoint

UPLOAD_FOLDER = 'uploads'  # Make sure this folder exists
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB limit

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return "No file part", 400

    video_file = request.files['video']
    if video_file.filename == '':
        return "No selected file", 400

    # Save uploaded video temporarily
    video_filename = secure_filename(video_file.filename)
    uploaded_folder = 'uploads'
    os.makedirs(uploaded_folder, exist_ok=True)

    # Create result folder
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_folder = os.path.join(uploaded_folder, f"result_{timestamp}")
    os.makedirs(result_folder, exist_ok=True)

    # Save the uploaded video inside the result folder
    video_path = os.path.join(result_folder, video_filename)
    video_file.save(video_path)

    # Load model
    model = load_checkpoint(r"D:\Praharsha\code\CAMZ\models\model_history\1.0-CNN\CNN_checkpoint.pth.tar", CNNModel)

    # Process and save results inside result folder
    predictions = process_video_with_model(video_path, model, trim_seconds=10, output_dir=result_folder)
    write_predictions_to_csv(predictions, video_path, result_folder) # This will save in the same folder as video_path


    return jsonify({
        'message': 'Processing completed successfully!'
    }), 200


if __name__ == '__main__':
    app.run(debug=True)