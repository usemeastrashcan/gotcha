# import cv2
# import mediapipe as mp
import numpy as np
import cv2
import mediapipe as mp

MAX_ANGLE = 20
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]


def return_center_of_iris(landmarks, iris_coordinate_indices):
    anotherArrayToStore_xy_coordinates = [landmarks[i] for i in iris_coordinate_indices]
    return np.mean([(j.x, j.y) for j in anotherArrayToStore_xy_coordinates], axis = 0)



def return_vector_from_left_to_right_corner_of_eye(landmarks, left_corner_index, right_corner_index):
    left = np.array([landmarks[left_corner_index].x, landmarks[left_corner_index].y])
    right = np.array([landmarks[right_corner_index].x, landmarks[right_corner_index].y])
    return right - left



def get_iris_angle(iris_center, eye_center, eye_vector):

    eye_center_array = np.array([eye_center.x, eye_center.y])
    vector_eyecenter_to_irisCenter = iris_center - eye_center_array

    unit_vector_eye = eye_vector/np.linalg.norm(eye_vector)
    unit_vector_iris = vector_eyecenter_to_irisCenter/np.linalg.norm(vector_eyecenter_to_irisCenter)

    dot_product_of_irisVector_and_EyeVector = np.clip(np.dot(unit_vector_eye, unit_vector_iris), -1, 1)

    angle = np.arccos(dot_product_of_irisVector_and_EyeVector)
    angle_in_degrees = np.degrees(angle)
    return angle_in_degrees


def monitor_iris(landmarks):
    left_iris_center = return_center_of_iris(landmarks, LEFT_IRIS)
    right_iris_center = return_center_of_iris(landmarks, RIGHT_IRIS)

    eyeVector_leftEye = return_vector_from_left_to_right_corner_of_eye(landmarks, 33, 133)
    eyeVector_rightEye = return_vector_from_left_to_right_corner_of_eye(landmarks, 362, 263)

    left_eye_angle = get_iris_angle(left_iris_center, landmarks[33], eyeVector_leftEye)
    right_eye_angle = get_iris_angle(right_iris_center, landmarks[362], eyeVector_rightEye)

    return max(left_eye_angle, right_eye_angle) > MAX_ANGLE



mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# For drawing the face landmarks
mp_drawing = mp.solutions.drawing_utils

# Open webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Read frame from webcam
    result, video_frame = video_capture.read()
    if not result:
        print("Error reading from camera.")
        break

    # Convert to RGB (MediaPipe requires RGB)
    rgb_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
    
    # Process with Face Mesh
    results = face_mesh.process(rgb_frame)
    
    # If face detected
    if results.multi_face_landmarks:
        # Draw all face contours
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=video_frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 255, 0),  # Green lines
                    thickness=1,
                    circle_radius=1
                )
            )

    if monitor_iris(results.multi_face_landmarks[0].landmark) == True:
        print("Cheater")
    
    # Display the resulting frame
    cv2.imshow("Face Mesh Detection", video_frame)
    
    # Exit on 'a' key press
    if cv2.waitKey(1) & 0xFF == ord("a"):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()