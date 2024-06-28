from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from utils import extract_features, sampling
import cv2
import tempfile
import os

app = Flask(__name__)
model = joblib.load('classification_model_1.pkl')  # Save and load your trained model
# pca = joblib.load('pca.pkl')  # Save and load PCA

# Create the sharpening kernel 
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
categories = ['Building', 'Glacier', 'Sea', 'Mountains', 'Streets', 'Forest']

@app.route('/', methods=['GET'])
def index():
    return "hello"

@app.route('/classification', methods=['GET', 'POST'])
def upload_file():
    images = []
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        try:
            # Save the uploaded file to a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            file.save(temp_file.name)

            temp_file.close()

            # Read the saved image using OpenCV
            cv_image = cv2.imread(temp_file.name)
            print(cv_image)

            # delete temporary file
            os.remove(temp_file.name)
            
            if cv_image is None:
                return jsonify({'error': 'Failed to read the uploaded image'}), 500

            # increase the quality of image
            resized_image = cv2.resize(cv_image, (128, 128))
            sharpened_image = cv2.filter2D(resized_image, -1, kernel) 

            # images.append(sharpened_image)


            features = extract_features(sharpened_image)
            print(features)
    
            prediction = model.predict(features)
            print(prediction)
            category = categories[prediction[0]]

            file_path = os.path.join('static/uploads', file.filename)
            file.save(file_path)
            print(file_path)
            file_path = file_path.split('\\')[-1]

            print(file_path)
                
            return render_template('index.html', category=category, image_path=file_path)
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return render_template('index.html')
            

if __name__ == '__main__':
    app.run(debug=True, port=5000)
