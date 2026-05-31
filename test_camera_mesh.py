import cv2
from hand_detector import HandSignRecognizer

def main():
    # 1. 載入我們的手勢與手部偵測器
    recognizer = HandSignRecognizer()
    
    # 2. 開啟攝影機 (預設為 0)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("【測試失敗】無法開啟攝影機，請檢查攝影機連線或權限。")
        recognizer.close()
        return

    print("【測試啟動】攝影機已成功開啟！正在顯示視窗...")
    print("請將手部放入鏡頭內。按下鍵盤 'q' 鍵即可結束測試。")

    try:
        while True:
            # 讀取一幀畫面
            success, frame = cap.read()
            if not success:
                print("無法從攝影機讀取畫面。")
                break

            # 3. 處理影像並偵測手部
            detections = recognizer.process(frame)

            # 4. 繪製手部網格與骨架
            recognizer.draw(frame, detections)

            # 5. 顯示於新視窗
            cv2.imshow("Hand Mesh Test - Press 'q' to Quit", frame)

            # 按下 'q' 鍵結束
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        recognizer.close()
        cv2.destroyAllWindows()
        print("【測試結束】視窗已關閉。")

if __name__ == "__main__":
    main()
