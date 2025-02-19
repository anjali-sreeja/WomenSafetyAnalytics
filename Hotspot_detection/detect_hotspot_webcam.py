# Import necessary libraries
import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Load pre-trained CNN model
model = keras.models.load_model('hotspot_detection1.h5')

# Open the default camera (index 0)
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()
    
    # Preprocess the frame
    frame = cv2.resize(frame, (224, 224))  # Resize to 224x224
    frame = frame / 255.0  # Normalize pixel values to [0, 1]
    
    # Convert the frame to a 4D tensor
    frame = frame.reshape((1, -1))  
    
    # Make predictions using the pre-trained CNN model
    predictions = model.predict(frame)
    
    # Get the class probabilities from the predictions
    class_probabilities = tf.nn.softmax(predictions[0])
    
    # Get the class with the highest probability
    class_id = tf.argmax(class_probabilities)
    
    # Print the detected class
    print(f'Detected class: {class_id}')
    
    # Display the output
    cv2.imshow('Camera', frame)
    
    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()