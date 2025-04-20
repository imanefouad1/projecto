import cv2
import mediapipe as mp
import numpy as np
from utils.optical_flow import compute_lucas_kanade, compute_farneback
from utils.feature_extraction import extract_histograms
from utils.visualization import draw_flow_arrows, draw_dense_flow

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
cap = cv2.VideoCapture(0)

prev_gray = None
prev_landmarks = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0]
        h, w, _ = frame.shape
        curr_points = np.array([[lm.x * w, lm.y * h] for lm in landmarks.landmark], dtype=np.float32)

        if prev_gray is not None and prev_landmarks is not None:
            good_old, good_new = compute_lucas_kanade(prev_gray, gray, prev_landmarks.reshape(-1, 1, 2))
            flow_vectors = good_new - good_old
            histogram = extract_histograms(flow_vectors)
            print("Flow Histogram:", histogram)

            draw_flow_arrows(frame, good_old, good_new)

            dense_flow = compute_farneback(prev_gray, gray)
            dense_vis = draw_dense_flow(dense_flow)
            cv2.imshow('Farneback Flow', dense_vis)

        prev_landmarks = curr_points
        prev_gray = gray.copy()

    cv2.imshow('Hand Gesture Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
