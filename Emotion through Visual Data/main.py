# Import necessary libraries
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

# Load pre-trained Haar Cascade Classifier for face detection
face_classifier = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')

# Load the pre-trained emotion classification model
classifier = load_model(r'model.h5')

# Define emotion labels corresponding to the model's output classes
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Start video capture from the default webcam (0)
cap = cv2.VideoCapture(0)

# Run real-time emotion detection loop
while True:
    # Capture frame-by-frame
    _, frame = cap.read()
    
    # Flip the frame horizontally (mirror effect)
    frame = cv2.flip(frame, 1) 

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_classifier.detectMultiScale(gray)

    # Loop through all detected faces
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

        # Extract the region of interest (ROI) for emotion classification
        roi_gray = gray[y:y+h, x:x+w]

        # Resize ROI to match model input size
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        # Check if ROI is valid (not completely black)
        if np.sum([roi_gray]) != 0:
            # Normalize the pixel values to [0, 1]
            roi = roi_gray.astype('float') / 255.0

            # Convert to array and reshape to match model input
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # Predict emotion
            prediction = classifier.predict(roi)[0]

            # Get the label with the highest confidence
            label = emotion_labels[prediction.argmax()]

            # Display the label on the frame near the face
            label_position = (x, y)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            # Display message if no valid face ROI is found
            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show the video frame with emotion labels
    cv2.imshow('Emotion Detector', frame)

    # Break loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close display window
cap.release()
cv2.destroyAllWindows()
