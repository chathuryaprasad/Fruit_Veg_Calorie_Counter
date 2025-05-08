import streamlit as st
import requests

# Title
st.title("Fruit & Veg Calories Counter")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

# Function to check the API response
def check_api_response(image):
    url = "http://127.0.0.1:5000/predict"  # Your API endpoint
    files = {"file": image}
    try:
        response = requests.post(url, files=files)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json(), response.status_code
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error occurred: {http_err}")  # Specific HTTP error
        return None, 500
    except Exception as err:
        st.error(f"Other error occurred: {err}")  # General error
        return None, 500

# Check the API response if an image is uploaded
if uploaded_file is not None:
    # Check the API response and status code
    json_response, status_code = check_api_response(uploaded_file)
    
    if json_response is not None:
        # Display the status code
        st.write(f"API Status Code: {status_code}")
        
        # If status code is 200, display the label, otherwise show an error
        if status_code == 200:
            # Extract and display only the label under predictions[0]
            if 'predictions' in json_response and len(json_response['predictions']) > 0:
                label = json_response['predictions'][0]['label']
                st.write(f"Predicted Label: {label}")
            else:
                st.error("No predictions found in the API response.")
        else:
            # Show error message if status code is not 200
            st.error(f"Error: API returned status code {status_code}. Please check the server.")
