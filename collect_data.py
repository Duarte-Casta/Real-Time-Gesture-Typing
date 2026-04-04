import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

# CONFIG
SAVE_DIR = "data/train"
IMG_SIZE = 128

# Criar pasta
os.makedirs(SAVE_DIR, exist_ok=True)

# Mediapipe
base_options = python.BaseOptions(model_asset_path='data/hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

current_label = None
counter = 0

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

            x_list = [int(lm.x * w) for lm in hand_landmarks]
            y_list = [int(lm.y * h) for lm in hand_landmarks]

            x_min, x_max = min(x_list), max(x_list)
            y_min, y_max = min(y_list), max(y_list)

            padding = 30
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)

            size = max(x_max - x_min, y_max - y_min)
            x_max = min(w, x_min + size)
            y_max = min(h, y_min + size)

            hand_img = frame[y_min:y_max, x_min:x_max]

            if hand_img.size != 0:
                img = cv2.resize(hand_img, (IMG_SIZE, IMG_SIZE))

                # Guardar se houver label ativa
                if current_label is not None:
                    class_dir = os.path.join(SAVE_DIR, current_label)
                    os.makedirs(class_dir, exist_ok=True)

                    filename = os.path.join(class_dir, f"{counter}.jpg")
                    cv2.imwrite(filename, img)
                    counter += 1

                cv2.imshow("Hand", img)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF

    # Teclas A-Z
    if key >= ord('a') and key <= ord('z'):
        current_label = chr(key).upper()
        counter = 0
        print(f"Capturing: {current_label}")

    # parar captura
    if key == ord(' '):
        current_label = None
        print("Stopped")

    # sair
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()