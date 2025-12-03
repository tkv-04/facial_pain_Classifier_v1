import cv2
import numpy as np
import tensorflow as tf

# --- CONFIG ---
MODEL_PATH = 'pain_classifier_mobilenetv2.h5'
IMG_SIZE = (224, 224)
CLASS_NAMES = ['mild', 'moderate', 'none', 'severe']  # Must match your training order
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# --- LOAD MODEL ---
model = tf.keras.models.load_model(MODEL_PATH)

# --- LOAD FACE CASCADE ---
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# --- START VIDEO CAPTURE ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    if len(faces) > 0:
        # Use the largest detected face
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        (x, y, w, h) = largest_face
        face_img = frame[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, IMG_SIZE)
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = face_img.astype(np.float32) / 255.0
        face_img = np.expand_dims(face_img, axis=0)

        preds = model.predict(face_img)
        class_idx = np.argmax(preds[0])
        class_name = CLASS_NAMES[class_idx]
        confidence = preds[0][class_idx]

        label = f"{class_name} ({confidence*100:.1f}%)"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No face detected", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

    cv2.imshow('Pain Detection (press q to quit)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 