import cv2
import tensorflow as tf

# Load the pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the pre-trained gender classification model
model = tf.keras.models.load_model('model.h5')


def classify_gender(face_image):
    # Convert the image to grayscale
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    # Resize the image to the input size expected by the model (width, height)
    face_image = cv2.resize(face_image, (80, 110))
    # Reshape the image to add the batch dimension and channel dimension
    face_image = face_image.reshape(1, 80, 110, 1)  # (batch_size, width, height, channels)
    # Normalize image data to range [0, 1]
    face_image = face_image / 255.0  
    # Make the prediction using the model
    prediction = model.predict(face_image)
    # Return the gender classification based on the prediction
    return "Female" if prediction[0][0] > 0.5 else "Male"

# Initialize gender counters
men_count = 0
women_count = 0

# Open the default camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()  # Capture each frame
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each detected face
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]  # Crop the face from the frame
        gender = classify_gender(face_img)  # Classify gender

        if gender == "Male":
            men_count += 1
        else:
            women_count += 1

        # Draw a rectangle around the face and display gender
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, gender, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow("Live Feed", frame)  # Display the frame with detected faces

    if cv2.waitKey(30) & 0xFF == ord('q'):  # Exit on 'q' key press
        break
    
cap.release()
cv2.destroyAllWindows()