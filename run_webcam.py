import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

model = load_model("drowsiness_model_colab.h5")
IMG_SIZE = (64, 64)
SEQUENCE_LENGTH = 20
frames = deque(maxlen=SEQUENCE_LENGTH)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, IMG_SIZE)
    img = img / 255.0
    frames.append(img)

    if len(frames) == SEQUENCE_LENGTH:
        X = np.expand_dims(frames, axis=0)   
        pred = model.predict(X)[0][0]
        label = "DROWSY" if pred > 0.5 else "AWAKE"

        color = (0,0,255) if label == "DROWSY" else (0,255,0)
        cv2.putText(frame, label, (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
