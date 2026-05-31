import cv2

def check_available_cameras(max_to_test=10):
    """
    掃描並列出所有可用的相機索引號
    """
    available_cameras = []
    
    print("正在掃描可用相機，請稍候...")
    print("-" * 40)
    
    for index in range(max_to_test):
        # 嘗試開啟對應索引的相機
        # cv2.CAP_DSHOW 是 Windows 系統加速開啟相機用的（DirectShow）
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        
        if cap.isOpened():
            # 嘗試讀取一幀，確保相機真的能正常工作，而不是被其他程式佔用
            ret, frame = cap.read()
            if ret:
                print(f"[成功] 找到可用相機！相機索引號 (Index): {index}")
                # 順便印出解析度
                width = cap.get(cv2.calcBlur_ if hasattr(cv2, 'calcBlur_') else cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                print(f"       預設解析度: {int(width)} x {int(height)}")
                available_cameras.append(index)
            else:
                print(f"[警告] 相機 Index {index} 可以開啟，但無法讀取影像（可能被其他程式佔用）。")
            
            # 測試完畢後釋放相機
            cap.release()
        else:
            # 如果找不到相機，通常代表後續的索引也不會有相機了
            pass

    print("-" * 40)
    if available_cameras:
        print(f"掃描完成！所有可用的相機 Index 清單: {available_cameras}")
        print("提示：在 cv2.VideoCapture(Index) 中填入上方數字即可使用該相機。")
    else:
        print("未偵測到任何可用的相機，請檢查硬體連接或驅動程式。")
        
    return available_cameras

if __name__ == "__main__":
    check_available_cameras()