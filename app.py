from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image, ImageEnhance
import os
import cv2

# Initialize the Flask app
app = Flask(__name__)

# Load your custom-trained model (ensure it's in the correct path)
model = tf.keras.models.load_model('./models/custom_mobilenet_model.h5', compile=False)

# Define your custom class labels
class_labels = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot',
    'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger',
    'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange',
    'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish',
    'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon'
]

calories_per_100g = {
    'apple': 52,
    'banana': 89,
    'beetroot': 43,
    'bell pepper': 31,
    'cabbage': 25,
    'capsicum': 31,
    'carrot': 41,
    'cauliflower': 25,
    'chilli pepper': 40,
    'corn': 86,
    'cucumber': 16,
    'eggplant': 25,
    'garlic': 149,
    'ginger': 80,
    'grapes': 69,
    'jalepeno': 29,
    'kiwi': 61,
    'lemon': 29,
    'lettuce': 15,
    'mango': 60,
    'onion': 40,
    'orange': 47,
    'paprika': 282,
    'pear': 57,
    'peas': 81,
    'pineapple': 50,
    'pomegranate': 83,
    'potato': 77,
    'raddish': 16,
    'soy beans': 446,
    'spinach': 23,
    'sweetcorn': 86,
    'sweetpotato': 86,
    'tomato': 18,
    'turnip': 28,
    'watermelon': 30
}


# Preprocess the image for MobileNetV2
def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')

    width, height = img.size
    new_width, new_height = min(width, height), min(width, height)
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height
    img = img.crop((left, top, right, bottom))

    img = img.resize((224, 224))  # Resize the image to 224x224

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.1) 


    img = ImageEnhance.Sharpness(img).enhance(1.2)  # Slightly sharper


    

    
    img_array = np.array(img)  # Convert image to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Normalize the image to [-1, 1] as expected by MobileNetV2
    img_array = img_array / 255.0  # Scale to [0, 1]
    img_array = img_array - 0.5    # Center to [-0.5, 0.5]
    img_array = img_array * 2      # Scale to [-1, 1]

    return img_array

# Function to classify the image and return top prediction
def classify_image(img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    top_index = predictions[0].argmax()
    top_prediction = (class_labels[top_index], float(predictions[0][top_index]))
    return top_prediction

# Route for image classification
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Save uploaded file
        if not os.path.exists('uploads'):
            os.makedirs('uploads')
        img_path = os.path.join('uploads', file.filename)
        file.save(img_path)

        label, score = classify_image(img_path)

        # Optional: remove the uploaded file to save space
        os.remove(img_path)

        # Get calories
        calories = calories_per_100g.get(label, "Unknown")


        response = {
            'prediction': {'label': label, 'probability': round(score, 2), "calories_per_100g": calories}
        }
        print(response)

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
