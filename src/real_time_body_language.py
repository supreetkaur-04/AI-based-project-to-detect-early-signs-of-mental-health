# real_time_body_language.py

import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def detect_pose(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(frame_rgb)

    if not result.pose_landmarks:
        print("⚠️ No body detected")
        return "Unknown"

    left_shoulder = result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

    shoulder_distance = abs(left_shoulder.y - right_shoulder.y)

    slouch_threshold = 0.04  
    upright_threshold = 0.03  

    if shoulder_distance > slouch_threshold:
        return "Slouching"
    elif shoulder_distance < upright_threshold:
        return "Upright"
    else:
        return "Upright"  

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Cannot open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Failed to capture frame")
            break

        posture = detect_pose(frame)

        cv2.putText(frame, f"Posture: {posture}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Real-Time Body Language Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
