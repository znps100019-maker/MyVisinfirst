/**
 * AI 台灣手語即時辨識系統 - 前端核心引擎
 * 支援 MediaPipe 關節追蹤 + 126 維空間特徵擷取 + 瀏覽器端 Float32Array 加速 KNN 比對
 */

// 全域狀態
let modelSamples = [];
let modelMatrix = null;
let modelLabels = [];
let uniqueVocab = [];
let isCameraRunning = false;
let isMirror = true;
let camera = null;
let handsDetector = null;

// 效能與時間統計
let lastFrameTime = performance.now();
let frameCount = 0;
let fps = 0;
let latency = 0;

// 手語辨識與穩定度狀態
const HISTORY_SIZE = 8;
const STABLE_MIN_COUNT = 4;
let predictionHistory = [];
let sentenceList = [];
let lastAddedSign = null;
let currentCandidate = 'No hand';
let currentConfidence = 0;
let isStable = false;

// DOM 元素快取
const videoElement = document.getElementById('webcamVideo');
const canvasElement = document.getElementById('outputCanvas');
const canvasCtx = canvasElement.getContext('2d');
const loadingOverlay = document.getElementById('loadingOverlay');
const loadingMsg = document.getElementById('loadingMsg');
const modelStatusBadge = document.getElementById('modelStatusBadge');
const vocabCountBadge = document.getElementById('vocabCountBadge');
const toggleCameraBtn = document.getElementById('toggleCameraBtn');
const toggleCameraText = document.getElementById('toggleCameraText');
const flipMirrorBtn = document.getElementById('flipMirrorBtn');
const videoFileInput = document.getElementById('videoFileInput');
const fpsVal = document.getElementById('fpsVal');
const latencyVal = document.getElementById('latencyVal');
const handCountHud = document.getElementById('handCountHud');
const landmarkConfHud = document.getElementById('landmarkConfHud');
const knnTableBody = document.getElementById('knnTableBody');
const heroSignWord = document.getElementById('heroSignWord');
const heroConfidence = document.getElementById('heroConfidence');
const confBarFill = document.getElementById('confBarFill');
const stabilityTag = document.getElementById('stabilityTag');
const decisionStatus = document.getElementById('decisionStatus');
const sentenceContainer = document.getElementById('sentenceContainer');
const copySentenceBtn = document.getElementById('copySentenceBtn');
const clearSentenceBtn = document.getElementById('clearSentenceBtn');
const vocabContainer = document.getElementById('vocabContainer');
const vocabSearchInput = document.getElementById('vocabSearchInput');
const vocabMatchCount = document.getElementById('vocabMatchCount');

// 手指標籤對應
const FINGERS = ['thumb', 'index', 'middle', 'ring', 'pinky'];
const FINGER_ELEMENTS = {
  thumb: document.getElementById('fingerThumb'),
  index: document.getElementById('fingerIndex'),
  middle: document.getElementById('fingerMiddle'),
  ring: document.getElementById('fingerRing'),
  pinky: document.getElementById('fingerPinky')
};

// 1. 初始化載入 KNN 手語模型
async function loadKNNModel() {
  loadingMsg.textContent = '正在載入 519 詞彙手語特徵模型 (model.json)...';
  
  const candidatePaths = ['model.json', '../sign_language_app/model.json'];
  let data = null;

  for (const path of candidatePaths) {
    try {
      const resp = await fetch(path);
      if (resp.ok) {
        data = await resp.json();
        break;
      }
    } catch (e) {
      console.warn(`嘗試載入 ${path} 失敗，嘗試備用路徑...`);
    }
  }

  if (!data || !data.samples || data.samples.length === 0) {
    modelStatusBadge.innerHTML = '<span class="status-dot" style="background:#ef4444;"></span><span>模型載入失敗</span>';
    loadingMsg.textContent = '無法載入 model.json，請確認檔案路徑。';
    return false;
  }

  modelSamples = data.samples;
  const N = modelSamples.length;
  modelMatrix = new Float32Array(N * 126);
  modelLabels = new Array(N);

  const vocabSet = new Set();
  for (let i = 0; i < N; i++) {
    const s = modelSamples[i];
    modelLabels[i] = s.label;
    vocabSet.add(s.label);
    const offset = i * 126;
    for (let d = 0; d < 126; d++) {
      modelMatrix[offset + d] = s.vector[d] || 0.0;
    }
  }

  uniqueVocab = Array.from(vocabSet).sort();
  vocabCountBadge.textContent = `${uniqueVocab.length} 手語詞彙`;
  modelStatusBadge.innerHTML = '<span class="status-dot"></span><span>519 詞彙模型已就緒</span>';
  
  // 建立詞庫速查清單
  renderVocabChips(uniqueVocab);
  return true;
}

// 2. 126 維度空間相對座標正規化 (與 Python 完全對齊)
function normalizeLandmarks(multiHandLandmarks) {
  const vec = new Float32Array(126); // 預設全部補零
  if (!multiHandLandmarks || multiHandLandmarks.length === 0) return vec;

  const hand1 = multiHandLandmarks[0];
  const wrist = hand1[0];
  const middleMcp = hand1[9];

  const dx = middleMcp.x - wrist.x;
  const dy = middleMcp.y - wrist.y;
  const dz = middleMcp.z - wrist.z;
  let scale = Math.sqrt(dx * dx + dy * dy + dz * dz);
  if (scale === 0) scale = 1e-6;

  // 第一隻手 (前 63 維)
  for (let i = 0; i < 21; i++) {
    const lm = hand1[i];
    vec[i * 3 + 0] = (lm.x - wrist.x) / scale;
    vec[i * 3 + 1] = (lm.y - wrist.y) / scale;
    vec[i * 3 + 2] = (lm.z - wrist.z) / scale;
  }

  // 第二隻手 (以第一隻手手腕為中心相對正規化，後 63 維)
  if (multiHandLandmarks.length > 1) {
    const hand2 = multiHandLandmarks[1];
    for (let i = 0; i < 21; i++) {
      const lm = hand2[i];
      vec[63 + i * 3 + 0] = (lm.x - wrist.x) / scale;
      vec[63 + i * 3 + 1] = (lm.y - wrist.y) / scale;
      vec[63 + i * 3 + 2] = (lm.z - wrist.z) / scale;
    }
  }

  return vec;
}

// 3. 高效 Float32Array KNN 距離檢索與 Top-K 加權投票
function classifyKNN(queryVec, numHands, k = 9) {
  if (!modelMatrix || modelLabels.length === 0) {
    return { label: 'Unknown', confidence: 0, topK: [] };
  }

  const N = modelLabels.length;
  const distList = new Array(N);

  for (let i = 0; i < N; i++) {
    const offset = i * 126;
    let sumSq = 0;
    for (let d = 0; d < 126; d++) {
      const diff = modelMatrix[offset + d] - queryVec[d];
      sumSq += diff * diff;
    }
    distList[i] = { index: i, dist: Math.sqrt(sumSq), label: modelLabels[i] };
  }

  // 排序前 K 個最小距離
  distList.sort((a, b) => a.dist - b.dist);
  const topK = distList.slice(0, Math.min(k, N));

  const minDist = topK[0].dist;
  const maxDist = numHands === 1 ? 16.0 : 45.0;

  // 距離過遠超過門檻則判定為無法識別
  if (minDist > maxDist) {
    return { label: 'Unknown', confidence: 0, topK };
  }

  // 距離反比加權投票
  const votes = {};
  let totalWeight = 0;
  for (const item of topK) {
    const weight = 1.0 / (item.dist + 1e-4);
    item.weight = weight;
    votes[item.label] = (votes[item.label] || 0) + weight;
    totalWeight += weight;
  }

  let bestLabel = 'Unknown';
  let bestWeight = 0;
  for (const [lbl, wt] of Object.entries(votes)) {
    if (wt > bestWeight) {
      bestWeight = wt;
      bestLabel = lbl;
    }
  }

  const confidence = totalWeight > 0 ? (bestWeight / totalWeight) : 0;
  if (confidence < 0.20) {
    return { label: 'Unknown', confidence: 0, topK };
  }

  return { label: bestLabel, confidence, topK };
}

// 4. 手指伸直/彎曲幾何判斷 (以第一隻手為主)
function detectFingerStates(handLandmarks) {
  if (!handLandmarks || handLandmarks.length < 21) {
    return { thumb: false, index: false, middle: false, ring: false, pinky: false };
  }

  const wrist = handLandmarks[0];
  const dist = (p1, p2) => Math.hypot(p1.x - p2.x, p1.y - p2.y, (p1.z || 0) - (p2.z || 0));

  // 食、中、無名、小指：Tip 到 Wrist 的距離大於 PIP 到 Wrist
  const indexExt = dist(handLandmarks[8], wrist) > dist(handLandmarks[6], wrist) * 1.05;
  const middleExt = dist(handLandmarks[12], wrist) > dist(handLandmarks[10], wrist) * 1.05;
  const ringExt = dist(handLandmarks[16], wrist) > dist(handLandmarks[14], wrist) * 1.05;
  const pinkyExt = dist(handLandmarks[20], wrist) > dist(handLandmarks[18], wrist) * 1.05;

  // 拇指：Tip 到 Pinky MCP 的距離大於 IP 到 Pinky MCP
  const pinkyMcp = handLandmarks[17];
  const thumbExt = dist(handLandmarks[4], pinkyMcp) > dist(handLandmarks[3], pinkyMcp) * 1.1;

  return { thumb: thumbExt, index: indexExt, middle: middleExt, ring: ringExt, pinky: pinkyExt };
}

// 5. MediaPipe Hands 結果處理回呼
function onHandsResults(results) {
  const now = performance.now();
  latency = Math.round(now - lastFrameTime);
  lastFrameTime = now;
  frameCount++;

  // 調整畫布尺寸以符合影像解析度
  if (canvasElement.width !== videoElement.videoWidth && videoElement.videoWidth > 0) {
    canvasElement.width = videoElement.videoWidth;
    canvasElement.height = videoElement.videoHeight;
  }

  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  
  // 繪製視訊影像至畫布
  canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

  const hands = results.multiHandLandmarks || [];
  const numHands = hands.length;

  if (numHands > 0) {
    handCountHud.textContent = numHands === 1 ? '單手偵測中 (1 Hand)' : `雙手聯合偵測中 (${numHands} Hands)`;
    landmarkConfHud.textContent = `21 關節點鎖定 (${Math.round((results.multiHandedness?.[0]?.score || 0.95) * 100)}%)`;

    // 繪製關節點與骨骼連線
    drawHandSkeletons(hands);

    // 126 維正規化與 KNN 推論
    const queryVec = normalizeLandmarks(hands);
    const { label, confidence, topK } = classifyKNN(queryVec, numHands);

    currentCandidate = label;
    currentConfidence = confidence;

    // 手指狀態偵測與更新
    const fingerStates = detectFingerStates(hands[0]);
    updateFingerUI(fingerStates);

    // 更新 Top-K 候選透視表
    updateKNNTableUI(topK);

    // 更新時間一致性統計
    updateTemporalStability(label);
  } else {
    handCountHud.textContent = '未偵測到手部';
    landmarkConfHud.textContent = '等待手部進入畫面';
    currentCandidate = 'No hand';
    currentConfidence = 0;
    isStable = false;
    updateFingerUI({ thumb: false, index: false, middle: false, ring: false, pinky: false });
    knnTableBody.innerHTML = '<tr><td colspan="4" class="empty-hint">等待手部進入畫面...</td></tr>';
    updateTemporalStability('No hand');
  }

  canvasCtx.restore();

  // 更新辨識結果看板
  updateResultsUI();
}

// 6. 骨架連線與關節彩色繪製
function drawHandSkeletons(hands) {
  const CONNECTIONS = [
    [0,1],[1,2],[2,3],[3,4],        // 拇指
    [0,5],[5,6],[6,7],[7,8],        // 食指
    [5,9],[9,10],[10,11],[11,12],   // 中指
    [9,13],[13,14],[14,15],[15,16], // 無名指
    [13,17],[17,18],[18,19],[19,20],[0,17] // 小指及手掌底
  ];

  hands.forEach((landmarks, handIdx) => {
    const isPrimary = handIdx === 0;
    const strokeColor = isPrimary ? '#00f0ff' : '#a855f7';
    const jointColor = isPrimary ? '#ffffff' : '#ffd700';

    // 繪製連線
    canvasCtx.lineWidth = 3.5;
    canvasCtx.strokeStyle = strokeColor;
    canvasCtx.shadowColor = strokeColor;
    canvasCtx.shadowBlur = 10;

    CONNECTIONS.forEach(([i, j]) => {
      const p1 = landmarks[i];
      const p2 = landmarks[j];
      canvasCtx.beginPath();
      canvasCtx.moveTo(p1.x * canvasElement.width, p1.y * canvasElement.height);
      canvasCtx.lineTo(p2.x * canvasElement.width, p2.y * canvasElement.height);
      canvasCtx.stroke();
    });

    // 繪製關節圓點
    landmarks.forEach((p, idx) => {
      const x = p.x * canvasElement.width;
      const y = p.y * canvasElement.height;
      canvasCtx.beginPath();
      const radius = idx === 0 ? 6.5 : (idx % 4 === 0 ? 5.5 : 4);
      canvasCtx.arc(x, y, radius, 0, 2 * Math.PI);
      canvasCtx.fillStyle = idx === 0 ? '#10b981' : (idx % 4 === 0 ? '#ff007f' : jointColor);
      canvasCtx.fill();
      canvasCtx.lineWidth = 1.5;
      canvasCtx.strokeStyle = '#000000';
      canvasCtx.stroke();
    });
  });
}

// 7. 時間一致性與手語句子累積
function updateTemporalStability(candidate) {
  predictionHistory.push(candidate);
  if (predictionHistory.length > HISTORY_SIZE) {
    predictionHistory.shift();
  }

  if (candidate === 'No hand' || candidate === 'Unknown') {
    isStable = false;
    return;
  }

  // 計算最近連續票數
  let trailingCount = 0;
  for (let i = predictionHistory.length - 1; i >= 0; i--) {
    if (predictionHistory[i] === candidate) {
      trailingCount++;
    } else {
      break;
    }
  }

  const totalCount = predictionHistory.filter(item => item === candidate).length;
  isStable = trailingCount >= STABLE_MIN_COUNT && totalCount >= STABLE_MIN_COUNT;

  // 若確認手勢穩定且與上一單字不同，加入手語語句
  if (isStable && candidate !== lastAddedSign) {
    addWordToSentence(candidate);
    lastAddedSign = candidate;
  }
}

function addWordToSentence(word) {
  sentenceList.push({
    word: word,
    time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
  });
  renderSentenceUI();
}

function renderSentenceUI() {
  if (sentenceList.length === 0) {
    sentenceContainer.innerHTML = '<span class="placeholder-text">比出穩定手勢後，系統會自動將單字組裝為語句...</span>';
    return;
  }

  sentenceContainer.innerHTML = '';
  sentenceList.forEach((item, index) => {
    const tag = document.createElement('span');
    tag.className = 'sentence-word-tag';
    tag.textContent = item.word;
    tag.title = `辨識時間: ${item.time}`;
    sentenceContainer.appendChild(tag);

    if (index < sentenceList.length - 1) {
      const arrow = document.createElement('span');
      arrow.className = 'sentence-arrow';
      arrow.textContent = '➔';
      sentenceContainer.appendChild(arrow);
    }
  });
  sentenceContainer.scrollTop = sentenceContainer.scrollHeight;
}

// 8. 介面各區塊狀態更新
function updateResultsUI() {
  if (currentCandidate === 'No hand') {
    heroSignWord.textContent = '請比出手語...';
    heroSignWord.style.opacity = '0.5';
    heroConfidence.textContent = '0%';
    confBarFill.style.width = '0%';
    stabilityTag.className = 'stability-tag unconfirmed';
    stabilityTag.textContent = '等待手勢';
    decisionStatus.innerHTML = '<span class="status-icon">⏳</span> 畫面無手部';
  } else if (currentCandidate === 'Unknown') {
    heroSignWord.textContent = '無法判讀';
    heroSignWord.style.opacity = '0.7';
    heroConfidence.textContent = `${Math.round(currentConfidence * 100)}%`;
    confBarFill.style.width = `${Math.round(currentConfidence * 100)}%`;
    stabilityTag.className = 'stability-tag unconfirmed';
    stabilityTag.textContent = '特徵距離過遠';
    decisionStatus.innerHTML = '<span class="status-icon">❓</span> 未在 519 詞庫中匹配';
  } else {
    heroSignWord.textContent = currentCandidate;
    heroSignWord.style.opacity = '1.0';
    const confPercent = Math.round(currentConfidence * 100);
    heroConfidence.textContent = `${confPercent}%`;
    confBarFill.style.width = `${confPercent}%`;

    if (isStable) {
      stabilityTag.className = 'stability-tag confirmed';
      stabilityTag.textContent = '已確認 (穩定手勢)';
      decisionStatus.innerHTML = '<span class="status-icon">✅</span> 成功鎖定手語詞彙';
    } else {
      stabilityTag.className = 'stability-tag unconfirmed';
      stabilityTag.textContent = '確認中...';
      decisionStatus.innerHTML = '<span class="status-icon">🔄</span> 手勢確認中';
    }
  }
}

function updateFingerUI(states) {
  FINGERS.forEach(finger => {
    const el = FINGER_ELEMENTS[finger];
    if (el) {
      if (states[finger]) {
        el.classList.add('extended');
        el.querySelector('.chip-state').textContent = '伸直';
      } else {
        el.classList.remove('extended');
        el.querySelector('.chip-state').textContent = '彎曲';
      }
    }
  });
}

function updateKNNTableUI(topK) {
  if (!topK || topK.length === 0) {
    knnTableBody.innerHTML = '<tr><td colspan="4" class="empty-hint">無有效候選資料</td></tr>';
    return;
  }

  let totalW = topK.reduce((acc, item) => acc + (item.weight || 0), 0);
  if (totalW === 0) totalW = 1;

  let html = '';
  topK.forEach((item, idx) => {
    const pct = Math.round(((item.weight || 0) / totalW) * 100);
    html += `
      <tr>
        <td>#${idx + 1}</td>
        <td><strong>${item.label}</strong></td>
        <td>${item.dist.toFixed(2)}</td>
        <td class="weight-bar-cell">
          <div style="display:flex; justify-content:space-between; font-size:0.75rem;">
            <span>${pct}%</span>
          </div>
          <div class="weight-bar-track">
            <div class="weight-bar-fill" style="width: ${pct}%;"></div>
          </div>
        </td>
      </tr>
    `;
  });
  knnTableBody.innerHTML = html;
}

function renderVocabChips(vocabList) {
  vocabContainer.innerHTML = '';
  vocabList.forEach(word => {
    const chip = document.createElement('span');
    chip.className = 'vocab-chip';
    chip.textContent = word;
    chip.onclick = () => {
      heroSignWord.textContent = word;
      heroConfidence.textContent = '詞庫標註';
      confBarFill.style.width = '100%';
    };
    vocabContainer.appendChild(chip);
  });
  vocabMatchCount.textContent = `共支援 ${vocabList.length} 個台灣手語詞彙`;
}

// 9. 詞彙字典搜尋過濾
vocabSearchInput.addEventListener('input', (e) => {
  const query = e.target.value.trim().toLowerCase();
  const matched = uniqueVocab.filter(w => w.toLowerCase().includes(query));
  renderVocabChips(matched);
  vocabMatchCount.textContent = `搜尋符合: ${matched.length} / ${uniqueVocab.length} 個詞彙`;
});

// 10. 句子複製與清除按鈕
copySentenceBtn.addEventListener('click', () => {
  if (sentenceList.length === 0) return;
  const sentenceText = sentenceList.map(item => item.word).join(' ');
  navigator.clipboard.writeText(sentenceText).then(() => {
    const orig = copySentenceBtn.textContent;
    copySentenceBtn.textContent = '✅ 已複製！';
    setTimeout(() => copySentenceBtn.textContent = orig, 1500);
  });
});

clearSentenceBtn.addEventListener('click', () => {
  sentenceList = [];
  lastAddedSign = null;
  renderSentenceUI();
});

// 11. 攝影機開關與鏡像控制
flipMirrorBtn.addEventListener('click', () => {
  isMirror = !isMirror;
  canvasElement.classList.toggle('mirror', isMirror);
});

toggleCameraBtn.addEventListener('click', async () => {
  if (isCameraRunning) {
    if (camera) {
      await camera.stop();
      camera = null;
    }
    videoElement.srcObject = null;
    isCameraRunning = false;
    toggleCameraText.textContent = '啟動攝影機';
    toggleCameraBtn.classList.remove('btn-danger');
    toggleCameraBtn.classList.add('btn-primary');
    loadingOverlay.classList.remove('hidden');
    loadingMsg.textContent = '攝影機已停止。點擊「啟動攝影機」以開始辨識。';
  } else {
    startWebcam();
  }
});

async function startWebcam() {
  loadingOverlay.classList.remove('hidden');
  loadingMsg.textContent = '正在請求攝影機權限並初始化...';

  try {
    if (!handsDetector) {
      initMediaPipeHands();
    }

    camera = new Camera(videoElement, {
      onFrame: async () => {
        if (handsDetector && isCameraRunning) {
          await handsDetector.send({ image: videoElement });
        }
      },
      width: 640,
      height: 480
    });

    await camera.start();
    isCameraRunning = true;
    toggleCameraText.textContent = '停止攝影機';
    toggleCameraBtn.classList.remove('btn-primary');
    toggleCameraBtn.classList.add('btn-danger');
    loadingOverlay.classList.add('hidden');
  } catch (err) {
    console.error('開啟攝影機失敗:', err);
    loadingMsg.textContent = `無法開啟攝影機：${err.message || err}`;
  }
}

// 12. 本地影片上傳支援
videoFileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;

  if (isCameraRunning && camera) {
    camera.stop();
    isCameraRunning = false;
    toggleCameraText.textContent = '啟動攝影機';
    toggleCameraBtn.classList.remove('btn-danger');
    toggleCameraBtn.classList.add('btn-primary');
  }

  const url = URL.createObjectURL(file);
  videoElement.src = url;
  videoElement.loop = true;
  videoElement.play();

  loadingOverlay.classList.add('hidden');

  async function processVideoFrame() {
    if (!videoElement.paused && !videoElement.ended && handsDetector) {
      await handsDetector.send({ image: videoElement });
      requestAnimationFrame(processVideoFrame);
    }
  }

  videoElement.onplay = () => {
    requestAnimationFrame(processVideoFrame);
  };
});

// 13. 初始化 MediaPipe Hands
function initMediaPipeHands() {
  handsDetector = new Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
  });

  handsDetector.setOptions({
    maxNumHands: 2,
    modelComplexity: 1,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
  });

  handsDetector.onResults(onHandsResults);
}

// FPS 計時器 (每秒更新)
setInterval(() => {
  fps = frameCount;
  frameCount = 0;
  fpsVal.textContent = fps;
  latencyVal.textContent = latency;
}, 1000);

// 頁面載入自啟動
window.addEventListener('DOMContentLoaded', async () => {
  canvasElement.classList.add('mirror');
  initMediaPipeHands();
  const loaded = await loadKNNModel();
  if (loaded) {
    loadingOverlay.classList.remove('hidden');
    loadingMsg.textContent = '模型載入完成！請點擊「啟動攝影機」或選擇影片開始。';
  }
});
