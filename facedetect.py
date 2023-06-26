import cv2

# Load the cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load the cascade for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Start capturing webcam video
video_capture = cv2.VideoCapture(0)

while True:
    # Capture a frame from the video
    ret, frame = video_capture.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # For each face detected, plot circles on the eyes
    for (x, y, w, h) in faces:
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) 
        # Select the face region
        face_roi = gray[y:y+h, x:x+w]

        # Detect eyes in the face region
        eyes = eye_cascade.detectMultiScale(face_roi)

        # For each eye detected, plot a circle
        for (ex, ey, ew, eh) in eyes:
            # Calculate the center of the eye
            center = (x + ex + ew//2, y + ey + eh//2)

            # Draw a circle on the center of the eye
            cv2.circle(frame, center, radius=10, color=(0, 255, 0), thickness=2)

    # Show the video with circles plotted on the eyes
    cv2.imshow('Video', frame)

    # Wait for a key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the video window
video_capture.release()
cv2.destroyAllWindows()