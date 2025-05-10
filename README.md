# Fruit and Vegetable Calorie Counter

This application is designed to classify and estimate the calorie content of fruits and vegetables through image recognition using a custom-trained MobileNet model. The app is powered by Flask for backend processing and Streamlit for a user-friendly interface. 

## Project Structure

- **models/**: Contains the custom-trained MobileNet model used for classifying fruits and vegetables.
- **uploads/**: Directory where test images are uploaded for classification.
- **app.py**: Flask backend that serves the model and handles classification requests. Run using `python app.py`.
- **main.py**: Streamlit app that provides a simple UI for users to interact with the classification system. Run using `streamlit run main.py`.
- **requirements.txt**: List of Python dependencies for the project.

## Features

- **Flask Backend**: The backend API is built using Flask and handles image classification requests. It uses the custom-trained MobileNet model to predict the calorie content of fruits and vegetables.
  
- **Streamlit UI**: The front-end interface is built using Streamlit, allowing users to upload images and receive calorie estimations directly.

## Installation

To run this application locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fruit-veg-calorie-counter.git
2. Navigate to the project directory:
   ```bash
   cd fruit-veg-calorie-counter
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

## Usage
1. Run Flask API (Backend)
To start the Flask backend, run the following command:
    ```bash
    python app.py
This will start the Flask server and the model will be ready to receive classification requests.

2. To start the Flask backend, run the following command:
    ```bash
    python app.py
This will open the Streamlit app in your web browser where you can upload an image of a fruit or vegetable, and the app will predict the calorie content.

## Model Details

The classification model used in this project is a custom-trained MobileNet model, designed for efficient performance on mobile and web platforms. 
It is trained on a dataset of fruits and vegetables, with labels corresponding to various types of produce.


### 1. Uploading an Image to the Flask API (via Postman)

To upload an image file for classification via the Flask API, follow these steps in Postman:

1. **Set the Request URL**:
   - Use the URL `http://127.0.0.1:5000/predict` for the Flask backend.

2. **Set the Request Method**:
   - Choose **POST** as the method.

3. **Set up the Body**:
   - Go to the **Body** tab in Postman.
   - Select **form-data**.
   - For the **Key**, type `file` (this is the key Flask expects for image uploads).
   - For the **Value**, click **Select Files** and choose the image you want to upload.

4. **Send the Request**:
   - Click the **Send** button to submit the image.
   - The response will return predictions based on the image provided.
   
### Sample Response:
After sending the request, the response will contain predictions, such as:
```json
{
    "predictions": [
        {
            "label": "beetroot",
            "probability": 1.0
        },
        {
            "label": "apple",
            "probability": 0.0
        },
        {
            "label": "banana",
            "probability": 0.0
        }
    ]
}


![image](https://github.com/user-attachments/assets/6f1ab84c-0f49-449b-a46a-6d9ca8aa16d4)



