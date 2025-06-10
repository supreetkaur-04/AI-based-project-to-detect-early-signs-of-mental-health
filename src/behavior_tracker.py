# behaviour_tracker.py

import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque

model = joblib.load('models/behavior_model.pkl')

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)

prev_keypoints = None
movement_history = deque(maxlen=10)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    
    if results.pose_landmarks:
        keypoints = np.array([(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]).flatten()
        
        if prev_keypoints is not None:
            total_movement = np.linalg.norm(keypoints - prev_keypoints)
        else:
            total_movement = 0
        
        prev_keypoints = keypoints
        movement_history.append(total_movement)
        smoothed_movement = np.mean(movement_history)
        
        if smoothed_movement < 0.1:
            inactivity, restlessness = 1, 0
        elif smoothed_movement < 0.3:
            inactivity, restlessness = 0, 0
        elif smoothed_movement < 0.6:
            inactivity, restlessness = 0, 0
        else:
            inactivity, restlessness = 0, 1
        
        features = np.array([[smoothed_movement, inactivity, restlessness]])
        prediction = model.predict(features)[0]
        
        cv2.putText(frame, f'Behavior: {prediction}', (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.imshow('Real-Time Behavior Prediction', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
