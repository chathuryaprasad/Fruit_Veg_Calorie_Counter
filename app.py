from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# Initialize the Flask app
app = Flask(__name__)

# Load your custom-trained model (ensure it's in the correct path)
model = tf.keras.models.load_model('models/custom_mobilenet_model.h5')

# Define your custom class labels
class_labels = ['apple', 'banana', 'beetroot']

# Preprocess the image for MobileNetV2
def preprocess_image(img_path):
    # Load image using PIL
    img = Image.open(img_path)
    img = img.resize((224, 224))  # Resize the image
    img_array = np.array(img)  # Convert image to NumPy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Normalize the image (MobileNetV2 requires pixel values between -1 and 1)
    img_array = img_array / 255.0  # Scale values to [0, 1]
    img_array = img_array - 0.5  # Center the data around 0
    img_array = img_array * 2  # Scale values to [-1, 1]
    
    return img_array

# Function to classify the image
def classify_image(img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    
    # Convert predictions to float values to avoid JSON serialization issues
    top_indices = predictions[0].argsort()[-3:][::-1]  # Top 3 predictions
    top_predictions = [(class_labels[i], float(predictions[0][i])) for i in top_indices]
    return top_predictions

# Route for image classification
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if an image file is provided in the request
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']

        # If no file is selected
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Save the uploaded file
        img_path = os.path.join('uploads', file.filename)
        file.save(img_path)

        # Classify the image using the model
        top_predictions = classify_image(img_path)

        # Prepare the response with the top predictions
        response = {
            'predictions': [
                {'label': label, 'probability': round(score, 2)} for label, score in top_predictions
            ]
        }

        return jsonify(response)
    
    except Exception as e:
        # Handle errors and return the exception message
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    # Create an 'uploads' folder to store the uploaded images
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    app.run(debug=True, host='0.0.0.0', port=5000)
