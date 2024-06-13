import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

def extract_skeleton_keypoints(image_path):
    image = cv2.imread(image_path)
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            keypoints = []
            for landmark in results.pose_landmarks.landmark:
                keypoints.append((landmark.x, landmark.y))
            return np.array(keypoints).flatten()
        else:
            return None

# Example usage
keypoints = extract_skeleton_keypoints('path_to_frame.jpg')
print(keypoints)
