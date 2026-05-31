import cv2
from hand_detector import HandSignRecognizer

def main():
    # 1. 載入我們的手勢與手部偵測器
    recognizer = HandSignRecognizer()
    
    # 2. 開啟攝影機並驗證是否能正常讀取影像
    print("【系統檢查】正在尋找可用的相機裝置...")
    cap = None
    working_index = None
    
    # 優先嘗試預設索引 0
    test_cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if test_cap.isOpened():
        success, _ = test_cap.read()
        if success:
            cap = test_cap
            working_index = 0
        else:
            print("警告：相機 0 雖然可以開啟，但無法讀取畫面（可能被佔用或為未啟動的虛擬相機）。")
            test_cap.release()
            
    # 如果索引 0 無法讀取，則自動嘗試搜尋索引 1 到 5
    if cap is None:
        print("正在嘗試自動尋找其他可用的相機索引（1 到 5）...")
        for index in range(1, 6):
            test_cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if test_cap.isOpened():
                success, _ = test_cap.read()
                if success:
                    cap = test_cap
                    working_index = index
                    break
                test_cap.release()

    if cap is None:
        print("\n【測試失敗】找不到任何可以正常讀取影像畫面的相機！")
        print("請嘗試以下步驟排除問題：")
        print("  1. 關閉所有可能在使用相機的程式（例如：OBS、Zoom、Teams、Discord、瀏覽器、其他 Python 視窗）。")
        print("  2. 到 Windows 設定的「隱私權與安全性 > 相機」，確認「允許桌面應用程式存取您的相機」已開啟。")
        print("  3. 拔除相機 USB 重新插入，或重開機以釋放被佔用的相機資源。）")
        recognizer.close()
        return

    print(f"\n【測試啟動】成功使用相機編號 {working_index} 開啟影像串流！正在顯示視窗...")
    print("請將手部放入鏡頭內。點擊影像視窗後，按下鍵盤的 'q' 鍵即可結束測試。")

    try:
        while True:
            # 讀取一幀畫面
            success, frame = cap.read()
            if not success:
                print("讀取畫面時發生錯誤。")
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
