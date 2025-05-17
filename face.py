import os
import cv2
import numpy as np
import mediapipe as mp

CANVAS_SIZE = (800, 800)
TARGET_EYE_DIST = 120
TARGET_LEFT_EYE_POS = (340, 300)

mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE = [33, 133]
RIGHT_EYE = [362, 263]
NOSE = 1

def get_eye_center(landmarks, indices, image_shape):
    h, w = image_shape[:2]
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    return np.mean(pts, axis=0)

def is_forward_facing(landmarks, image_shape):
    h, w = image_shape[:2]
    left_eye = get_eye_center(landmarks, LEFT_EYE, image_shape)
    right_eye = get_eye_center(landmarks, RIGHT_EYE, image_shape)
    nose = landmarks[NOSE]
    nose_x = nose.x * w

    # Check if nose is roughly centered between eyes
    eye_mid_x = (left_eye[0] + right_eye[0]) / 2
    nose_offset = abs(nose_x - eye_mid_x)
    max_nose_offset = 0.04 * w

    # Check if both eyes are horizontally aligned
    eye_diff_x = abs(left_eye[0] - right_eye[0])
    eye_diff_y = abs(left_eye[1] - right_eye[1])
    is_level = eye_diff_x > eye_diff_y * 3

    # Check eye-nose distance balance (side profiles fail this)
    left_nose_dist = abs(left_eye[0] - nose_x)
    right_nose_dist = abs(right_eye[0] - nose_x)
    eye_balance_ratio = min(left_nose_dist, right_nose_dist) / max(left_nose_dist, right_nose_dist)

    return (nose_offset < max_nose_offset) and is_level and (eye_balance_ratio > 0.65)

def align_face(img, left_eye, right_eye):
    delta = right_eye - left_eye
    angle = np.degrees(np.arctan2(delta[1], delta[0]))
    center = tuple(np.mean([left_eye, right_eye], axis=0))
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]))

    new_left = rot_mat @ np.append(left_eye, 1)
    new_right = rot_mat @ np.append(right_eye, 1)

    current_dist = np.linalg.norm(new_right - new_left)
    scale = TARGET_EYE_DIST / current_dist
    resized = cv2.resize(rotated, (0, 0), fx=scale, fy=scale)

    new_left_scaled = new_left * scale
    dx = TARGET_LEFT_EYE_POS[0] - new_left_scaled[0]
    dy = TARGET_LEFT_EYE_POS[1] - new_left_scaled[1]

    canvas = np.zeros((CANVAS_SIZE[1], CANVAS_SIZE[0], 3), dtype=np.uint8)
    x_offset = int(dx)
    y_offset = int(dy)

    x1 = max(0, x_offset)
    y1 = max(0, y_offset)
    x2 = min(CANVAS_SIZE[0], x_offset + resized.shape[1])
    y2 = min(CANVAS_SIZE[1], y_offset + resized.shape[0])

    src_x1 = max(0, -x_offset)
    src_y1 = max(0, -y_offset)

    canvas[y1:y2, x1:x2] = resized[src_y1:src_y1 + (y2 - y1), src_x1:src_x1 + (x2 - x1)]
    return canvas

def process_faces(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    skipped_log = os.path.join(output_dir, 'skipped_images.txt')
    with open(skipped_log, 'w') as skipped_file:
        with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
            for file in os.listdir(input_dir):
                if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                path = os.path.join(input_dir, file)
                image = cv2.imread(path)
                if image is None:
                    continue
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)
                if not results.multi_face_landmarks:
                    skipped_file.write(file + '\n')
                    continue
                landmarks = results.multi_face_landmarks[0].landmark
                if not is_forward_facing(landmarks, image.shape):
                    skipped_file.write(file + '\n')
                    continue
                left_eye = get_eye_center(landmarks, LEFT_EYE, image.shape)
                right_eye = get_eye_center(landmarks, RIGHT_EYE, image.shape)
                aligned = align_face(image, left_eye, right_eye)
                cv2.imwrite(os.path.join(output_dir, file), aligned)
