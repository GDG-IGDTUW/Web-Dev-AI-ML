import os
from flask import Flask, render_template, request, jsonify, send_from_directory
import face_recognition
from PIL import Image, ImageDraw

app = Flask(__name__)

# Folder to save uploaded images
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Route to display the upload form
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and face recognition
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Save the uploaded image
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Load the uploaded image using face_recognition
    image = face_recognition.load_image_file(file_path)
    face_locations = face_recognition.face_locations(image)

    # Convert image to RGB for displaying and saving
    pil_image = Image.open(file_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.load_default()

    # Draw rectangles around detected faces
    #Issue #8 Image labelling functionality
    detected_faces = []
    for i, (top, right, bottom, left) in enumerate(face_locations):
        person_lbl = f"Person {i+1}"
        draw.rectangle([left, top, right, bottom], outline="red", width=5)
        draw.text((left, top-10), person_lbl, fill="blue", font=font)
        detected_faces.append({'name':person_lbl, 'box':[top, right, bottom, left]})

    # Save the image with faces highlighted
    result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_' + file.filename)
    pil_image.save(result_image_path)

    # Return the result image path
    return jsonify({
        'message': 'File uploaded and faces detected!',
        'image': f'/uploads/{os.path.basename(result_image_path)}'
        'faces':detected_faces
    })

# Route to serve uploaded and processed images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
