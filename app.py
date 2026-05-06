import os
import cv2
import numpy as np
from keras.models import load_model

# ✅ Step 1: Load model
model_path = r"C:\Users\swati\OneDrive\Desktop\emotion_project\mode1\emotion_model.hdf5"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

model = load_model(model_path)

# Emotion labels (must match your training order)
emotion_labels = ["angry","happy","neutral","sad","surprise"]

# ✅ Step 2: Initialize face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ✅ Step 3: Start webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise Exception("Could not open webcam")

# ✅ Step 4: Process frames
while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not working")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48,48))
        face = face / 255.0
        face = np.reshape(face, (1,48,48,1))

        # ✅ Step 5: Predict emotion
        prediction = model.predict(face, verbose=0)
        label = emotion_labels[np.argmax(prediction)]
        prob = np.max(prediction)

        # ✅ Step 6: Draw rectangle & label
        color = (0,255,0)  # Green for all emotions
        cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
        cv2.putText(frame, f"{label} ({prob:.2f})", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

    # Show frame
    cv2.imshow("Emotion Detector", frame)

    # Press ESC to exit
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()