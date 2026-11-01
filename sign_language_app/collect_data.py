"""
手語資料收集工具（雙手 + 臉部）
操作：
  - 按一次空白鍵：開始連續記錄
  - 再按一次空白鍵：停止記錄
  - q：離開並存檔
"""
import cv2
import mediapipe as mp
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sign_language_app"))
from landmarks import (
    mediapipe_landmarks_to_list,
    mediapipe_face_to_list,
    build_full_vector,
)

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

MODEL_PATH = os.path.join("sign_language_app", "model.json")

# 每幾幀記錄一筆（數字越大，同樣時間錄的筆數越少）
RECORD_INTERVAL_FRAMES = 3


def main():
    samples = []
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "r", encoding="utf-8") as f:
            samples = json.load(f).get("samples", [])
        print(f"已載入現有 {len(samples)} 筆樣本")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    hands = mp_hands.Hands(
        static_image_mode=False, max_num_hands=2,
        min_detection_confidence=0.7, min_tracking_confidence=0.6,
    )
    face = mp_face.FaceMesh(
        static_image_mode=False, max_num_faces=1,
        min_detection_confidence=0.5,
    )

    label = input("請輸入這個手語的標籤名稱（例如：你好）：").strip()
    if not label:
        print("標籤不能為空")
        return

    count = 0
    is_recording = False
    frame_counter = 0

    print(f"開始收集「{label}」")
    print("操作：按空白鍵開始記錄，再按空白鍵停止，按 q 離開並存檔")

    cv2.namedWindow("Collect Data", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Collect Data", 960, 720)
    cv2.moveWindow("Collect Data", 100, 100)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        hand_results = hands.process(rgb)
        face_results = face.process(rgb)

        hands_list = []
        if hand_results.multi_hand_landmarks:
            for hand in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                hands_list.append(mediapipe_landmarks_to_list(hand))

        face_list = None
        if face_results.multi_face_landmarks:
            face_list = mediapipe_face_to_list(face_results.multi_face_landmarks[0])

        # 錄製中：每隔 RECORD_INTERVAL_FRAMES 幀記錄一筆
        if is_recording and hands_list:
            frame_counter += 1
            if frame_counter >= RECORD_INTERVAL_FRAMES:
                vector = build_full_vector(hands_list, face_list)
                samples.append({"label": label, "vector": vector})
                count += 1
                frame_counter = 0

        # 顯示狀態
        if is_recording:
            cv2.circle(frame, (30, 60), 15, (0, 0, 255), -1)
            cv2.putText(frame, f"● 錄製中  {label}  已錄 {count}", (60, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            cv2.putText(frame, f"○ 待機  {label}  已錄 {count}", (30, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if not hands_list:
            cv2.putText(frame, "未偵測到手", (30, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.putText(frame, "空白鍵: 開始/停止   q: 離開", (30, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Collect Data", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            is_recording = not is_recording
            frame_counter = 0
            print(f"{'開始' if is_recording else '停止'}記錄，目前共 {count} 筆")
        elif key == ord("q"):
            break

    with open(MODEL_PATH, "w", encoding="utf-8") as f:
        json.dump({"samples": samples}, f, ensure_ascii=False)
    print(f"已儲存 {len(samples)} 筆樣本")

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    face.close()


if __name__ == "__main__":
    main()