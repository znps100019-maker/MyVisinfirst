import cv2
import mediapipe as mp
import os

mp_face = mp.solutions.face_mesh
VIDEO_DIR = os.path.join("sign_language_app", "raw_videos")

face = mp_face.FaceMesh(
    static_image_mode=False, max_num_faces=1,
    min_detection_confidence=0.5,
)

for filename in sorted(os.listdir(VIDEO_DIR)):
    if not filename.endswith(".mp4"):
        continue
    path = os.path.join(VIDEO_DIR, filename)
    cap = cv2.VideoCapture(path)
    total = 0
    detected = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        total += 1
        if total % 3 != 0:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        r = face.process(rgb)
        if r.multi_face_landmarks:
            detected += 1
    cap.release()
    print(f"{filename}: 抽樣 {total//3} 幀, 偵測到臉 {detected} 幀")

face.close()
