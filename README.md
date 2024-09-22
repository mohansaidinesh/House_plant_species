# 🌿 Plant Species Detection using Flask and OpenCV
This project is designed to detect and classify 47 different species of house plants using a deep learning model. It utilizes a Flask web server for receiving images, and OpenCV for capturing images from a mobile device camera or webcam. The model processes the image and predicts the plant species, which is then returned to the user.

## 🚀 Features
Mobile Camera Integration: Capture plant images using your mobile camera.
Deep Learning Model: Utilizes a trained model for detecting 47 plant species.
Flask Backend: Handles image uploads and returns predictions.
OpenCV for Image Capture: Captures images directly from the mobile device or webcam.
Mobile-Friendly Interface: Designed to work seamlessly on mobile devices.
## 🛠️ Technologies Used
Python 3
Flask
OpenCV
Keras for model loading
Mobile camera access
HTML5 & JavaScript (for optional web interface)
## 📂 Project Structure
bash

house_plant_species/
│
├── app.py               # Flask server handling predictions

├── camera_capture.py     # Script for capturing images via OpenCV

├── model/

│   └── your_model.h5     # Pre-trained plant species detection model

└── templates/

    └── index.html        # Optional, simple HTML interface for web interaction
    
## ⚙️ Setup Instructions
Follow the steps below to get the project running on your local machine or mobile device:

Prerequisites
Python 3.7+
OpenCV
Flask
Keras & TensorFlow (for loading the deep learning model)
1. Clone the Repository
bash
git clone https://github.com/your-username/house-plant-detection.git
cd house-plant-detection

2. Install Dependencies
Use pip to install the necessary dependencies:

bash
Copy code
pip install -r requirements.txt
If you don't have a requirements.txt file, install the dependencies manually:

bash
Copy code
pip install flask opencv-python keras tensorflow numpy requests

3. Run the Flask App
Make sure your trained model is saved in the model/ directory, then start the Flask server:

bash
Copy code
python app.py
This will start the Flask server locally on http://127.0.0.1:5000. You can now upload plant images for prediction.

4. Capture Images from Mobile (Optional)
If you want to capture plant images using your mobile device:

a) Install Termux on Android
Install Termux from Google Play Store or F-Droid.

b) Install OpenCV and Python in Termux
Open Termux and run:

bash
Copy code
pkg update
pkg install python
pkg install opencv
pip install opencv-python requests numpy
c) Run Camera Capture Script
In Termux, create and run the camera_capture.py script to capture images from your mobile device and send them to the Flask server for prediction:

bash
Copy code
python camera_capture.py
Ensure that the Flask app is accessible on the same network, or use services like ngrok if needed.

5. Mobile-Friendly Web Interface (Optional)
If you'd prefer to use a browser-based interface:

Open http://<your-server-ip>:5000 on your mobile device.
Capture and upload an image of a plant for species detection.

## 📊 Model Information
The model used in this project is a MobileNetV2 architecture trained on images of 47 house plant species.
Make sure to place the model file (.h5) in the model/ directory before running the Flask app.

## 📋 API Endpoints
/predict
Method: POST
Description: Receives an image file and returns the predicted plant species.
Example: Upload an image using the form or programmatically using an HTTP client.
python
Copy code
files = {'file': open('plant_image.jpg', 'rb')}
response = requests.post('http://127.0.0.1:5000/predict', files=files)
print(response.json())

## 🖼️ Example Output
After uploading or capturing an image, the Flask app will return a prediction in the following format:

json
Copy code
{
  "prediction": "Aloe Vera"
}

## 📱 Running on Mobile Devices
Ensure both your mobile device and the Flask server are on the same network (Wi-Fi).
You can use services like ngrok or a public IP to expose your Flask server to the internet for remote access.

## 🔧 Future Improvements
Implement live video feed for continuous prediction.
Add more plant species to the dataset.
Improve the user interface for better mobile experience.

## 🤝 Contributing
Feel free to submit pull requests or open issues if you have any suggestions or find any bugs.

## 📜 License
This project is licensed under the NONE License - see the LICENSE file for details.
