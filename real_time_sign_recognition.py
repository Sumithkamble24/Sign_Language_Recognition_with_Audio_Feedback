
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pyttsx3
import threading
import queue
import json
import time

model = load_model('sign_language_model.h5')
with open('class_labels.json') as f:
    class_indices = json.load(f)
classes = list(class_indices.keys())

engine = pyttsx3.init()
engine.setProperty('rate', 150)  #(150 words/minute).
speech_queue = queue.Queue()

def speech_worker():
    while True:
        text = speech_queue.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()
        speech_queue.task_done()

speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()

IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.8
STABLE_FRAMES = 15
prev_prediction = ""
stable_counter = 0
last_spoken = ""
last_spoken_time = 0
COOLDOWN_TIME = 2  

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    prediction = ""

    if result.multi_hand_landmarks:
        all_x, all_y = [], []

        for hand_landmarks in result.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                all_x.append(int(lm.x * w))
                all_y.append(int(lm.y * h))

        if all_x and all_y:
            x_min = max(min(all_x) - 20, 0)
            y_min = max(min(all_y) - 20, 0)
            x_max = min(max(all_x) + 20, w)
            y_max = min(max(all_y) + 20, h)

            roi = frame[y_min:y_max, x_min:x_max]

            if roi.size != 0:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                h_roi, w_roi = gray.shape
                diff = abs(h_roi - w_roi)

                if h_roi > w_roi:
                    pad = diff // 2
                    gray = cv2.copyMakeBorder(gray, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=0)
                elif w_roi > h_roi:
                    pad = diff // 2
                    gray = cv2.copyMakeBorder(gray, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=0)

                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray = clahe.apply(gray)

                resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
                rgb_image = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
                normalized = rgb_image / 255.0
                reshaped = normalized.reshape(1, IMG_SIZE, IMG_SIZE, 3)

                pred = model.predict(reshaped, verbose=0)[0]
                max_index = np.argmax(pred)
                confidence = pred[max_index]

                if confidence > CONFIDENCE_THRESHOLD:
                    prediction = classes[max_index]

                    if prediction == prev_prediction:
                        
                        stable_counter += 1
                    else:
                        stable_counter = 0
                        prev_prediction = prediction

                    if stable_counter == STABLE_FRAMES:
                        current_time = time.time()
                        if prediction != last_spoken or (current_time - last_spoken_time) > COOLDOWN_TIME:
                            with speech_queue.mutex:
                                speech_queue.queue.clear()
                            speech_queue.put(prediction)
                            last_spoken = prediction
                            last_spoken_time = current_time
                        stable_counter = 0

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, f'{prev_prediction}', (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Live Sign Language Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

speech_queue.put(None)
cap.release()
cv2.destroyAllWindows()


