import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from tensorflow.keras.models import load_model

# CONFIG
MODEL_PATH = "models/model_v1.keras"
IMG_SIZE = 128
CONF_THRESHOLD = 0.8

# labels
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "nothing", "space"]

# LOAD MODEL
model = load_model(MODEL_PATH)


# MEDIAPIPE HANDS (NOVA API TASKS)
# Configuração do detetor usando o ficheiro .task
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

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # BGR -> RGB (MediaPipe precisa converter)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # A nova API exige que a imagem seja convertida para o formato mp.Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    
    # Processar a deteção
    result = detector.detect(mp_image)

    # A verificação agora é feita através de result.hand_landmarks
    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:

            # BOUNDING BOX
            x_list = []
            y_list = []

            # Na nova API, iteramos diretamente sobre a lista de landmarks
            for lm in hand_landmarks:
                x_list.append(int(lm.x * w))
                y_list.append(int(lm.y * h))

            x_min, x_max = min(x_list), max(x_list)
            y_min, y_max = min(y_list), max(y_list)

            # padding
            padding = 40
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)

            # tornar quadrado
            box_w = x_max - x_min
            box_h = y_max - y_min
            size = max(box_w, box_h)

            x_max = x_min + size
            y_max = y_min + size

            # garantir limites
            x_max = min(w, x_max)
            y_max = min(h, y_max)

            # CROP
            hand_img = frame[y_min:y_max, x_min:x_max]

            if hand_img.size != 0:

                # PREPROCESS (IGUAL AO TREINO)
                img = cv2.resize(hand_img, (IMG_SIZE, IMG_SIZE))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # importante
                img = img / 255.0
                img = np.expand_dims(img, axis=0)

                # PREDIÇÃO
                pred = model.predict(img, verbose=0)
                class_id = np.argmax(pred)
                confidence = np.max(pred)

                if confidence > CONF_THRESHOLD:
                    label = labels[class_id]
                else:
                    label = "..."
                print(f"Predicted: {label} ({confidence:.2f})")
                # DESENHAR RESULTADO
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 2)

                cv2.putText(frame, f"{label} ({confidence:.2f})",
                            (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0,255,0), 2)

    cv2.imshow("ASL Detector", frame)

    # ESC para sair
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()