"""
用 webcam 錄製手語影片
執行：python record_video.py --label 你好
操作：
  - 空白鍵：開始 / 停止錄影
  - q：離開
"""
import argparse
import cv2
import os

VIDEO_DIR = os.path.join("sign_language_app", "raw_videos")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="手語標籤（會當作檔名）")
    parser.add_argument("--camera", type=int, default=0, help="攝影機編號")
    args = parser.parse_args()

    os.makedirs(VIDEO_DIR, exist_ok=True)
    output_path = os.path.join(VIDEO_DIR, f"{args.label}.mp4")

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("無法開啟攝影機")
        return

    fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = None
    is_recording = False

    print(f"標籤：{args.label}")
    print(f"輸出：{output_path}")
    print("空白鍵：開始/停止錄影，q：離開")

    cv2.namedWindow("Record Video", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Record Video", 960, 720)
    cv2.moveWindow("Record Video", 100, 100)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        if is_recording and writer is not None:
            writer.write(frame)
            cv2.circle(frame, (30, 60), 15, (0, 0, 255), -1)
            cv2.putText(frame, "● 錄影中", (60, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "○ 待機", (30, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.putText(frame, "空白鍵: 開始/停止   q: 離開", (30, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Record Video", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            if not is_recording:
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                is_recording = True
                print("開始錄影...")
            else:
                is_recording = False
                if writer is not None:
                    writer.release()
                    writer = None
                print(f"停止錄影，已儲存到 {output_path}")
        elif key == ord("q"):
            break

    if writer is not None:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()
    print("結束")


if __name__ == "__main__":
    main()