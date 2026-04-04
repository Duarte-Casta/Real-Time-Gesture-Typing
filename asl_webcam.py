import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from tensorflow.keras.models import load_model

import time
from collections import Counter

# CONFIG
MODEL_PATH = "models/model_v1.keras"
IMG_SIZE = 128
CONF_THRESHOLD = 0.3

# labels
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "nothing", "space"]

# LOAD MODEL
model = load_model(MODEL_PATH)

# MEDIAPIPE HANDS (NOVA API TASKS)
base_options = python.BaseOptions(model_asset_path='data/hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
detector = vision.HandLandmarker.create_from_options(options)

# WEBCAM
cap = cv2.VideoCapture(0)

# Buffer para estabilizar previsões
start_time = time.time()
INTERVAL = 2
pred_buffer = []
last_letter = ""
output_file = open("output.txt", "w")

# loop principal
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:

            # Bounding box
            x_list = [int(lm.x * w) for lm in hand_landmarks]
            y_list = [int(lm.y * h) for lm in hand_landmarks]

            x_min, x_max = min(x_list), max(x_list)
            y_min, y_max = min(y_list), max(y_list)

            padding = 30
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)

            # Tornar quadrado
            box_w, box_h = x_max - x_min, y_max - y_min
            size = max(box_w, box_h)
            x_max = x_min + size
            y_max = y_min + size
            x_max = min(w, x_max)
            y_max = min(h, y_max)

            # Crop da mão
            hand_img = frame[y_min:y_max, x_min:x_max]

            if hand_img.size != 0:
                # Criar máscara da mão usando landmarks relativos ao crop
                """mask = np.zeros((y_max - y_min, x_max - x_min), dtype=np.uint8)
                pts = np.array([[int((lm.x * w) - x_min), int((lm.y * h) - y_min)] for lm in hand_landmarks])
                cv2.fillPoly(mask, [pts], 255)

                # Blur do fundo dentro do    quadrado
                blurred = cv2.GaussianBlur(hand_img, (25, 25), 0)
                hand_img = np.where(mask[:, :, None] == 255, hand_img, blurred)"""

                # Preprocess para o modelo
                img = cv2.resize(hand_img, (IMG_SIZE, IMG_SIZE))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img / 255.0
                img = np.expand_dims(img, axis=0)

                # Predição
                pred = model.predict(img, verbose=0)
                class_id = np.argmax(pred)
                confidence = np.max(pred)
                label = labels[class_id] if confidence > CONF_THRESHOLD else "..."
                pred_buffer.append(labels[class_id])
                print(f"Predicted: {label} ({confidence:.2f})")

                # Desenhar resultado
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 2)
                cv2.putText(frame, f"{label} ({confidence:.2f})",
                            (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("ASL Detector", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    current_time = time.time()

    if current_time - start_time > INTERVAL and len(pred_buffer) > 0:
        most_common = Counter(pred_buffer).most_common(1)[0][0]

        if most_common != last_letter:
            output_file.write(most_common)
            output_file.flush()
            last_letter = most_common

        pred_buffer = []
        start_time = current_time

output_file.close()

cap.release()
cv2.destroyAllWindows()