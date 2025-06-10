# # distress_detection.py

import threading
import time
import cv2
import torch
import torchvision
import numpy as np
from collections import deque
from PIL import Image
import mediapipe as mp
from alert import trigger_alert  

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
model.fc = torch.nn.Sequential(
    torch.nn.Linear(model.fc.in_features, 512),
    torch.nn.BatchNorm1d(512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.4),
    torch.nn.Linear(512, 7))
model.load_state_dict(torch.load("models/facial_expression_resnet34_best.pth", map_location=device))
model = model.to(device)
model.eval()

class_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

transform = torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(num_output_channels=3),
    torchvision.transforms.Resize((48, 48)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
])

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

facial_emotion = "Neutral"
posture_status = "Unknown"
behavior_prediction = "Normal"
distress_flag = False

lock = threading.Lock()

def facial_emotion_detection(frame):
    global facial_emotion
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]  
        face_pil = Image.fromarray(face) 
        face_tensor = transform(face_pil).unsqueeze(0).to(device) 
        with torch.no_grad():
            output = model(face_tensor)
            _, predicted = torch.max(output, 1)
            with lock:
                facial_emotion = class_labels[predicted.item()]

def posture_detection(frame):
    global posture_status
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        if results.pose_landmarks:
            left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            shoulder_distance = abs(left_shoulder.y - right_shoulder.y)
            posture_status = "Slouching" if shoulder_distance > 0.04 else "Upright"

def behavior_tracking(frame):
    global behavior_prediction
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        if results.pose_landmarks:
            keypoints = np.array([(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]).flatten()
            if hasattr(behavior_tracking, 'prev_keypoints'):
                total_movement = np.linalg.norm(keypoints - behavior_tracking.prev_keypoints)
            else:
                total_movement = 0
            behavior_tracking.prev_keypoints = keypoints
            if not hasattr(behavior_tracking, 'movement_history'):
                behavior_tracking.movement_history = deque(maxlen=10)
            behavior_tracking.movement_history.append(total_movement)
            smoothed_movement = np.mean(behavior_tracking.movement_history)
            if smoothed_movement < 0.1:
                inactivity, restlessness = 1, 0
            elif smoothed_movement < 0.6:
                inactivity, restlessness = 0, 0
            else:
                inactivity, restlessness = 0, 1
            features = np.array([[smoothed_movement, inactivity, restlessness]])
            with lock:
                behavior_prediction = "Restless" if restlessness else "Normal"

def distress_monitoring():
    global facial_emotion, posture_status, behavior_prediction, distress_flag
    while True:
        with lock:
            if (facial_emotion in ["Sad", "Angry"] and 
                posture_status == "Slouching" and 
                behavior_prediction == "Restless"):
                if not distress_flag:
                    trigger_alert("⚠️ Warning: Possible depression detected! (Sad/Angry + Slouching + Restless)")
                    distress_flag = True

            elif (facial_emotion in ["Sad", "Angry"] and 
                  posture_status == "Restless"):
                if not distress_flag:
                    trigger_alert("⚠️ Possible Anxiety: Fidgeting or pacing detected!")
                    distress_flag = True

            elif (facial_emotion in ["Confused", "Fear"] and 
                  posture_status == "Slouching"):
                if not distress_flag:
                    trigger_alert("⚠️ Possible Social Anxiety or Fear detected!")
                    distress_flag = True

            elif (behavior_prediction == "Inactive"):
                if not distress_flag:
                    trigger_alert("⚠️ Possible Depression: Lack of movement detected!")
                    distress_flag = True

            elif (facial_emotion == "Sad" and 
                  posture_status == "Slouching" and 
                  behavior_prediction == "Normal"):
                if not distress_flag:
                    trigger_alert("⚠️ Possible Fatigue: Neutral expression with slouching.")
                    distress_flag = True

            else:
                distress_flag = False

        time.sleep(1)  

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    threading.Thread(target=distress_monitoring, daemon=True).start()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        facial_thread = threading.Thread(target=facial_emotion_detection, args=(frame,), daemon=True)
        posture_thread = threading.Thread(target=posture_detection, args=(frame,), daemon=True)
        behavior_thread = threading.Thread(target=behavior_tracking, args=(frame,), daemon=True)

        facial_thread.start()
        posture_thread.start()
        behavior_thread.start()

        facial_thread.join()
        posture_thread.join()
        behavior_thread.join()

        cv2.putText(frame, f"Emotion: {facial_emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Posture: {posture_status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Behavior: {behavior_prediction}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Real-Time Distress Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()