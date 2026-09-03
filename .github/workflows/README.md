# CI/CD 工作流程說明

## 工作流程

### 1. 語法檢查（syntax-check）
- 檢查所有 Python 檔案的語法
- 在每次 push 和 PR 時運行

### 2. 準確度測試（accuracy-test）
- 運行單一影片的準確度測試
- 確保準確度維持在 99% 以上

### 3. 綜合測試（comprehensive-test）
- 測試所有 9 個影片
- 計算平均準確度

### 4. 儀表板測試（dashboard-test）
- 建立即時視覺化儀表板
- 上傳截圖作為 artifacts

### 5. 效能基準測試（performance-benchmark）
- 測量初始化時間
- 測量處理速度（FPS）

## 觸發條件

- **Push**: 推送到 main 或 develop 分支
- **Pull Request**: 建立或更新 PR
- **定時**: 每週日午夜自動運行

## 查看結果

1. 前往 GitHub 專案頁面
2. 點擊 "Actions" 標籤
3. 查看最新的工作流程運行結果
4. 點擊具體的工作流程查看詳細日誌
