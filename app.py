import streamlit as st
import cv2
from keras.models import model_from_json
import numpy as np

# Load model
with open("facialemotionmodel.json", "r") as f:
    model = model_from_json(f.read())
model.load_weights("emotiondetector.h5")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
labels = {0 : 'angry', 1 : 'disgust', 2 : 'fear', 3 : 'happy', 4 : 'neutral', 5 : 'sad', 6 : 'surprise'}

def extract_features(image):
    image = cv2.resize(image, (48, 48))
    image = image.reshape(1, 48, 48, 1)
    return image / 255.0

st.title("Real-Time Facial Emotion Detector")

run = st.checkbox('Start Camera')
FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        img = extract_features(face_img)
        pred = model.predict(img)
        emotion = labels[pred.argmax()]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

camera.release()
