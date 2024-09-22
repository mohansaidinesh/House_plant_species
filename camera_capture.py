import cv2
import requests
import numpy as np

# Replace with your mobile IP camera URL
ip_camera_url = 'http://192.168.1.40:8080/video'  # Example URL

# Initialize the camera
cap = cv2.VideoCapture(ip_camera_url)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    # Display the frame
    cv2.imshow('Plant Species Detection - Press "c" to capture', frame)

    # Press 'c' to capture the image or 'q' to quit
    key = cv2.waitKey(1)
    if key % 256 == ord('c'):
        # Save the captured frame to a file
        img_name = "captured_plant_image.jpg"
        cv2.imwrite(img_name, frame)
        print(f"Image saved: {img_name}")

        # Send image to Flask app for prediction
        url = 'http://192.168.1.33:5000/predict'  # Replace with your Flask server's IP
        files = {'file': open(img_name, 'rb')}

        response = requests.post(url, files=files)
        print(f"Response from server: {response.json()['prediction']}")

    elif key % 256 == ord('q'):
        # Exit the loop if 'q' is pressed
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
