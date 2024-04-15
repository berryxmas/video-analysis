import cv2
import dlib
import numpy as np
import datetime
import time
import logging

# Set up logging
logging.basicConfig(filename='log.txt', level=logging.INFO, format='%(message)s')

# Load the detector
detector = dlib.get_frontal_face_detector()

# Load the predictor
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Load the face recognition model
face_rec_model = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")

# Load the images and labels
images = [cv2.imread(f"content/{person}{num}.jpeg") for person in ["rutte", "wilders"] for num in range(1, 3)]
labels = ["Rutte", "Rutte", "Wilders", "Wilders"]  # Adjusted to match each image

# Initialize an empty list to store face encodings
face_encodings = []

for image in images:
    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces in the image
    faces = detector(image_rgb)

    for face in faces:
        # Get the facial landmarks for the detected face
        shape = predictor(image_rgb, face)

        # Compute the face encoding
        face_encoding = np.array(face_rec_model.compute_face_descriptor(image_rgb, shape))
        face_encodings.append(face_encoding)

# Initialize video capture
cap = cv2.VideoCapture('content/trimmed_video2.mp4')

# Timing control variables
print_interval = 3  # seconds
last_print_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the frame
    boxes = detector(frame_rgb, 0)

    for box in boxes:
        # Draw a green rectangle around the detected face
        x, y, w, h = box.left(), box.top(), box.width(), box.height()
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Get the facial landmarks for the detected face
        shape = predictor(frame_rgb, box)

        # Compute the face encoding
        face_encoding = np.array(face_rec_model.compute_face_descriptor(frame_rgb, shape))

        # Compare the face encoding to the known face encodings
        matches = [(np.linalg.norm(face_encoding - encoding), label) for encoding, label in zip(face_encodings, labels)]
        matches.sort(key=lambda x: x[0])

        # If the closest match is close enough and it's time to print a new message
        current_time = time.time()
        if matches[0][0] < 0.6 and (current_time - last_print_time) > print_interval:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            message = f"[{timestamp}] Detected: {matches[0][1]}"
            print(message)
            logging.info(message)  # Log the message
            last_print_time = current_time
            
            # Draw a red rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
