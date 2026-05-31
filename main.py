import cv2
import pyautogui
# ⭐️ 核心：從我們剛剛寫的檔案裡，把大腦拿進來用
from arm_detector import ArmDetector

# 1. 叫出我們的手臂辨識大腦   
detector = ArmDetector()

# 2. 準備專題的計數變數
counter = 0
stage = None

# 3. 開啟攝影機
cap = cv2.VideoCapture(0)
print("【畢業專題系統】啟動中... 請站至鏡頭前。")

while True:
    success, img = cap.read()
    if not success:
        break

    # 4. 呼叫大腦去抓手臂的三個點
    points = detector.find_arm_points(img)

    if points is not None:
        shoulder, elbow, wrist = points

        # 5. 呼叫大腦去算手臂角度
        angle = detector.calculate_angle(shoulder, elbow, wrist)

        # 6. 專題邏輯判斷與控制
        if angle > 160:
            stage = "down"
        if angle < 45 and stage == 'down':
            stage = "up"
            counter += 1
            # 觸發控制訊號：模擬按下鍵盤空白鍵
            pyautogui.press('space')
            print(f"【觸發成功】目前次數：{counter}")

        # 7. 視覺化呈現：畫出關節與連線
        cv2.circle(img, tuple(shoulder), 12, (0, 0, 255), cv2.FILLED)
        cv2.circle(img, tuple(elbow), 12, (0, 0, 255), cv2.FILLED)
        cv2.circle(img, tuple(wrist), 12, (0, 0, 255), cv2.FILLED)
        cv2.line(img, tuple(shoulder), tuple(elbow), (0, 255, 0), 3)
        cv2.line(img, tuple(elbow), tuple(wrist), (0, 255, 0), 3)

        # 在手肘旁印出當前角度
        cv2.putText(img, f"{angle} deg", (elbow[0] + 15, elbow[1] - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # 左上角專題控制面板
        cv2.rectangle(img, (0, 0), (280, 100), (0, 0, 0), cv2.FILLED)
        cv2.putText(img, f"ARM COUNTER: {counter}", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img, f"SYSTEM STAGE: {stage}", (10, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # 顯示主畫面
    cv2.imshow("Graduation Project - Main Control", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# 紀錄一下
cap.release()
cv2.destroyAllWindows()