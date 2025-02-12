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

#Issue #7: Implement Preprocessing for Input Images 

def preprocess_image(file_path):
    image = cv2.imread(file_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # grayscale
    
    # Applying Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Sharpening filter to enhance edges
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(blurred, -1, kernel)
    
    # Save preprocessed image
    processed_path = file_path.replace('.jpg', '_processed2.jpg')
    cv2.imwrite(processed_path, sharpened)
    
    return processed_path  # Returns new image path

def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Save the uploaded image
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    
    # Preprocess Image Before Face Detection
    processed_path = preprocess_image(file_path)

    # Load the uploaded image using face_recognition
    image = face_recognition.load_image_file(processed_path)
    face_locations = face_recognition.face_locations(image)

    # Convert image to RGB for displaying and saving
    pil_image = Image.open(file_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    # Draw rectangles around detected faces
    for face_location in face_locations:
        top, right, bottom, left = face_location
        draw.rectangle([left, top, right, bottom], outline="red", width=5)

    # Save the image with faces highlighted
    result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_' + file.filename)
    pil_image.save(result_image_path)

    # Return the result image path
    return jsonify({
        'message': 'File uploaded and faces detected!',
        'image': f'/uploads/{os.path.basename(result_image_path)}'
    })

# Route to serve uploaded and processed images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
