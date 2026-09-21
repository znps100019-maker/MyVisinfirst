/**
 * AI 台灣手語專屬互動學習與即時測驗系統 - 核心腳本
 * 包含：即時辨識實驗室、手語教室、AI 測驗闖關、519 大字典、學習歷程
 */

// ===================== 全域狀態 =====================
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

// 手語辨識核心狀態
const HISTORY_SIZE = 8;
const STABLE_MIN_COUNT = 3;
let predictionHistory = [];
let sentenceList = [];
let lastAddedSign = null;
let currentCandidate = 'No hand';
let currentConfidence = 0;
let isStable = false;

// 手語教室狀態
let currentPracticeTarget = '1';
let practiceMatchStreak = 0;

// AI 測驗引擎狀態
let isQuizActive = false;
let quizQuestions = [];
let currentQuizIdx = 0;
let quizScore = 0;
let quizCombo = 0;
let quizMaxCombo = 0;
let quizCorrectCount = 0;
let quizTimerId = null;
let quizTimeLeft = 10.0;
let quizMatchStreak = 0;

// Web Audio 音效合成器
let audioCtx = null;

function getAudioContext() {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
  if (audioCtx.state === 'suspended') {
    audioCtx.resume();
  }
  return audioCtx;
}

function playTone(freq, type = 'sine', duration = 0.15, gainVal = 0.15) {
  try {
    const ctx = getAudioContext();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = type;
    osc.frequency.setValueAtTime(freq, ctx.currentTime);
    gain.gain.setValueAtTime(gainVal, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + duration);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + duration);
  } catch (e) {
    // 忽略未經使用者互動時的音效錯誤
  }
}

function playSuccessChime() {
  playTone(523.25, 'triangle', 0.12, 0.2); // C5
  setTimeout(() => playTone(659.25, 'triangle', 0.12, 0.2), 80); // E5
  setTimeout(() => playTone(783.99, 'triangle', 0.15, 0.2), 160); // G5
  setTimeout(() => playTone(1046.50, 'triangle', 0.3, 0.25), 240); // C6
}

function playErrorBuzz() {
  playTone(200, 'sawtooth', 0.25, 0.15);
}

function playTickSound() {
  playTone(880, 'sine', 0.04, 0.05);
}

// ===================== DOM 元素快取 =====================
const videoElement = document.getElementById('webcamVideo');
const canvasElement = document.getElementById('outputCanvas');
const canvasCtx = canvasElement.getContext('2d');
const loadingOverlay = document.getElementById('loadingOverlay');
const loadingMsg = document.getElementById('loadingMsg');
const modelStatusBadge = document.getElementById('modelStatusBadge');
const toggleCameraBtn = document.getElementById('toggleCameraBtn');
const toggleCameraText = document.getElementById('toggleCameraText');
const flipMirrorBtn = document.getElementById('flipMirrorBtn');
const videoFileInput = document.getElementById('videoFileInput');

// 頂部 Dock 元素
const dockSignWord = document.getElementById('dockSignWord');
const dockConfidence = document.getElementById('dockConfidence');
const dockConfBarFill = document.getElementById('dockConfBarFill');
const dockStabilityTag = document.getElementById('dockStabilityTag');
const fpsVal = document.getElementById('fpsVal');
const latencyVal = document.getElementById('latencyVal');
const handCountHud = document.getElementById('handCountHud');
const landmarkConfHud = document.getElementById('landmarkConfHud');

// 分頁 1 元素
const knnTableBody = document.getElementById('knnTableBody');
const heroSignWord = document.getElementById('heroSignWord');
const heroConfidence = document.getElementById('heroConfidence');
const confBarFill = document.getElementById('confBarFill');
const stabilityTag = document.getElementById('stabilityTag');
const decisionStatus = document.getElementById('decisionStatus');
const sentenceContainer = document.getElementById('sentenceContainer');
const copySentenceBtn = document.getElementById('copySentenceBtn');
const clearSentenceBtn = document.getElementById('clearSentenceBtn');

// 手指 DOM
const FINGERS = ['thumb', 'index', 'middle', 'ring', 'pinky'];
const FINGER_ELEMENTS = {
  thumb: document.getElementById('fingerThumb'),
  index: document.getElementById('fingerIndex'),
  middle: document.getElementById('fingerMiddle'),
  ring: document.getElementById('fingerRing'),
  pinky: document.getElementById('fingerPinky')
};

// 分頁 2 手語教室元素
const lessonCardsContainer = document.getElementById('lessonCardsContainer');
const practiceTargetWord = document.getElementById('practiceTargetWord');
const practiceTargetHint = document.getElementById('practiceTargetHint');
const practiceFeedbackBox = document.getElementById('practiceFeedbackBox');
const feedbackIcon = document.getElementById('feedbackIcon');
const feedbackTitle = document.getElementById('feedbackTitle');
const feedbackDesc = document.getElementById('feedbackDesc');
const practiceMatchPercent = document.getElementById('practiceMatchPercent');
const practiceMatchBar = document.getElementById('practiceMatchBar');
const nextLessonBtn = document.getElementById('nextLessonBtn');

// 分頁 3 AI 測驗元素
const quizLobby = document.getElementById('quizLobby');
const quizArena = document.getElementById('quizArena');
const quizSummaryModal = document.getElementById('quizSummaryModal');
const startQuizBtn = document.getElementById('startQuizBtn');
const abandonQuizBtn = document.getElementById('abandonQuizBtn');
const retryQuizBtn = document.getElementById('retryQuizBtn');
const viewStatsBtn = document.getElementById('viewStatsBtn');
const quizCountSelect = document.getElementById('quizCountSelect');
const quizCurrentQ = document.getElementById('quizCurrentQ');
const quizTotalQ = document.getElementById('quizTotalQ');
const quizCurrentScore = document.getElementById('quizCurrentScore');
const quizComboBadge = document.getElementById('quizComboBadge');
const timerBarFill = document.getElementById('timerBarFill');
const timerText = document.getElementById('timerText');
const quizTargetWord = document.getElementById('quizTargetWord');
const quizTargetHint = document.getElementById('quizTargetHint');
const quizCurrentSeen = document.getElementById('quizCurrentSeen');
const quizCurrentConf = document.getElementById('quizCurrentConf');
const quizMatchingBar = document.getElementById('quizMatchingBar');
const summaryFinalScore = document.getElementById('summaryFinalScore');
const summaryCorrectCount = document.getElementById('summaryCorrectCount');
const summaryAccuracy = document.getElementById('summaryAccuracy');
const summaryMaxCombo = document.getElementById('summaryMaxCombo');
const summaryGrade = document.getElementById('summaryGrade');

// 分頁 4 大字典元素
const dictSearchInput = document.getElementById('dictSearchInput');
const dictionaryGrid = document.getElementById('dictionaryGrid');
const dictResultCount = document.getElementById('dictResultCount');

// 分頁 5 學習歷程元素
const statsTotalQuizzes = document.getElementById('statsTotalQuizzes');
const statsHighScore = document.getElementById('statsHighScore');
const statsAvgAccuracy = document.getElementById('statsAvgAccuracy');
const statsMasteredWords = document.getElementById('statsMasteredWords');
const historyTableBody = document.getElementById('historyTableBody');
const clearStatsBtn = document.getElementById('clearStatsBtn');

// ===================== 手語教室內建精選單元 =====================
const CLASSROOM_LESSONS = [
  { word: '1', name: '數字 1', category: 'numbers', hint: '請伸出食指，其餘手指握拳。' },
  { word: '2', name: '數字 2', category: 'numbers', hint: '伸出食指與中指（勝利 V 手勢）。' },
  { word: '3', name: '數字 3', category: 'numbers', hint: '伸出食指、中指、無名指三指。' },
  { word: '4', name: '數字 4', category: 'numbers', hint: '伸出四指，拇指內扣。' },
  { word: '5', name: '數字 5', category: 'numbers', hint: '五指張開（打招呼手形）。' },
  { word: '6', name: '數字 6', category: 'numbers', hint: '拇指與小指伸直（打電話手勢）。' },
  { word: '7', name: '數字 7', category: 'numbers', hint: '拇指與食指伸直（槍形）。' },
  { word: '8', name: '數字 8', category: 'numbers', hint: '拇指、食指、中指三指伸直。' },
  { word: '9', name: '數字 9', category: 'numbers', hint: '食指彎曲成鉤狀，其餘手指收攏。' },
  { word: '10', name: '數字 10', category: 'numbers', hint: '雙手食指交叉成「十」字形。' },
  { word: '你好', name: '你好', category: 'greetings', hint: '單手拇指伸直（比讚），朝前微微點頭致意。' },
  { word: '謝謝', name: '謝謝', category: 'greetings', hint: '一手平放，另一手在掌心上方輕拍或致意。' },
  { word: '對不起', name: '對不起', category: 'greetings', hint: '手微握拳置於胸前，輕輕上下點動表示歉意。' },
  { word: '不要', name: '不要', category: 'daily', hint: '一手食指在胸前左右搖晃擺動。' },
  { word: '一模一樣', name: '一模一樣', category: 'daily', hint: '雙手食指水平相對平行移動。' },
  { word: '喜歡', name: '喜歡', category: 'emotions', hint: '右手張開置於胸前，輕撫心臟處。' },
  { word: '享受', name: '享受', category: 'emotions', hint: '雙手輕撫兩頰或胸口，神情陶醉。' },
  { word: '互相幫忙', name: '互相幫忙', category: 'daily', hint: '雙手手心相對，向對方微微靠攏推進。' }
];

// ===================== 1. 載入模型與初始化 =====================
async function loadKNNModel() {
  loadingMsg.textContent = '正在載入 519 詞彙手語特徵模型 (model.json)...';
  
  const candidatePaths = ['model.json', 'web/model.json', '../sign_language_app/model.json'];
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
  modelStatusBadge.innerHTML = `<span class="status-dot"></span><span>519 詞彙模型就緒 (${N} 筆特徵)</span>`;
  
  // 初始化渲染各大分頁內容
  initClassroomCards();
  initDictionaryGrid(uniqueVocab);
  renderStatsDashboard();
  return true;
}

// ===================== 2. 126 維度特徵正規化 =====================
function normalizeLandmarks(multiHandLandmarks) {
  const vec = new Float32Array(126); // 預設補零
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

  // 第二隻手 (以第一隻手手腕為基準相對正規化，後 63 維)
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

// ===================== 3. KNN 距離檢索與加權投票 =====================
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

  distList.sort((a, b) => a.dist - b.dist);
  const topK = distList.slice(0, Math.min(k, N));

  const minDist = topK[0].dist;
  const maxDist = numHands === 1 ? 16.0 : 45.0;

  if (minDist > maxDist) {
    return { label: 'Unknown', confidence: 0, topK };
  }

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

// ===================== 4. 手指伸展狀態偵測 =====================
function detectFingerStates(handLandmarks) {
  if (!handLandmarks || handLandmarks.length < 21) {
    return { thumb: false, index: false, middle: false, ring: false, pinky: false };
  }

  const wrist = handLandmarks[0];
  const dist = (p1, p2) => Math.hypot(p1.x - p2.x, p1.y - p2.y, (p1.z || 0) - (p2.z || 0));

  const indexExt = dist(handLandmarks[8], wrist) > dist(handLandmarks[6], wrist) * 1.05;
  const middleExt = dist(handLandmarks[12], wrist) > dist(handLandmarks[10], wrist) * 1.05;
  const ringExt = dist(handLandmarks[16], wrist) > dist(handLandmarks[14], wrist) * 1.05;
  const pinkyExt = dist(handLandmarks[20], wrist) > dist(handLandmarks[18], wrist) * 1.05;

  const pinkyMcp = handLandmarks[17];
  const thumbExt = dist(handLandmarks[4], pinkyMcp) > dist(handLandmarks[3], pinkyMcp) * 1.1;

  return { thumb: thumbExt, index: indexExt, middle: middleExt, ring: ringExt, pinky: pinkyExt };
}

// ===================== 5. 核心每幀回呼 (Core Frame Loop) =====================
function onHandsResults(results) {
  const now = performance.now();
  latency = Math.round(now - lastFrameTime);
  lastFrameTime = now;
  frameCount++;

  if (canvasElement.width !== videoElement.videoWidth && videoElement.videoWidth > 0) {
    canvasElement.width = videoElement.videoWidth;
    canvasElement.height = videoElement.videoHeight;
  }

  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

  const hands = results.multiHandLandmarks || [];
  const numHands = hands.length;

  if (numHands > 0) {
    handCountHud.textContent = numHands === 1 ? '單手追蹤' : `雙手追蹤 (${numHands})`;
    landmarkConfHud.textContent = '關節點鎖定';

    drawHandSkeletons(hands);

    const queryVec = normalizeLandmarks(hands);
    const { label, confidence, topK } = classifyKNN(queryVec, numHands);

    currentCandidate = label;
    currentConfidence = confidence;

    const fingerStates = detectFingerStates(hands[0]);
    updateFingerUI(fingerStates);
    updateKNNTableUI(topK);
    updateTemporalStability(label);

    // 處理手語教室模式的練習反饋
    handleClassroomFeedback(label, confidence);

    // 處理 AI 測驗模式的即時核對
    handleQuizFrameMatching(label, confidence);
  } else {
    handCountHud.textContent = '未偵測到手部';
    landmarkConfHud.textContent = '等待手部進入';
    currentCandidate = 'No hand';
    currentConfidence = 0;
    isStable = false;
    updateFingerUI({ thumb: false, index: false, middle: false, ring: false, pinky: false });
    knnTableBody.innerHTML = '<tr><td colspan="4" class="empty-hint">等待手部進入畫面...</td></tr>';
    updateTemporalStability('No hand');
    handleClassroomFeedback('No hand', 0);
    handleQuizFrameMatching('No hand', 0);
  }

  canvasCtx.restore();
  updateDockAndRecognitionUI();
}

// ===================== 6. 骨架渲染 =====================
function drawHandSkeletons(hands) {
  const CONNECTIONS = [
    [0,1],[1,2],[2,3],[3,4],
    [0,5],[5,6],[6,7],[7,8],
    [5,9],[9,10],[10,11],[11,12],
    [9,13],[13,14],[14,15],[15,16],
    [13,17],[17,18],[18,19],[19,20],[0,17]
  ];

  hands.forEach((landmarks, handIdx) => {
    const isPrimary = handIdx === 0;
    const strokeColor = isPrimary ? '#00f0ff' : '#a855f7';
    const jointColor = isPrimary ? '#ffffff' : '#ffd700';

    canvasCtx.lineWidth = 3.5;
    canvasCtx.strokeStyle = strokeColor;
    canvasCtx.shadowColor = strokeColor;
    canvasCtx.shadowBlur = 8;

    CONNECTIONS.forEach(([i, j]) => {
      const p1 = landmarks[i];
      const p2 = landmarks[j];
      canvasCtx.beginPath();
      canvasCtx.moveTo(p1.x * canvasElement.width, p1.y * canvasElement.height);
      canvasCtx.lineTo(p2.x * canvasElement.width, p2.y * canvasElement.height);
      canvasCtx.stroke();
    });

    landmarks.forEach((p, idx) => {
      const x = p.x * canvasElement.width;
      const y = p.y * canvasElement.height;
      canvasCtx.beginPath();
      const radius = idx === 0 ? 6.5 : (idx % 4 === 0 ? 5.5 : 3.8);
      canvasCtx.arc(x, y, radius, 0, 2 * Math.PI);
      canvasCtx.fillStyle = idx === 0 ? '#10b981' : (idx % 4 === 0 ? '#ff007f' : jointColor);
      canvasCtx.fill();
      canvasCtx.lineWidth = 1.5;
      canvasCtx.strokeStyle = '#000000';
      canvasCtx.stroke();
    });
  });
}

// ===================== 7. 時間一致性與語句串接 =====================
function updateTemporalStability(candidate) {
  predictionHistory.push(candidate);
  if (predictionHistory.length > HISTORY_SIZE) {
    predictionHistory.shift();
  }

  if (candidate === 'No hand' || candidate === 'Unknown') {
    isStable = false;
    return;
  }

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

// ===================== 8. UI 更新 =====================
function updateDockAndRecognitionUI() {
  const confPct = Math.round(currentConfidence * 100);

  if (currentCandidate === 'No hand') {
    dockSignWord.textContent = '請比出手語...';
    dockConfidence.textContent = '0%';
    dockConfBarFill.style.width = '0%';
    dockStabilityTag.className = 'stability-tag unconfirmed';
    dockStabilityTag.textContent = '等待手勢';

    heroSignWord.textContent = '請比出手語...';
    heroConfidence.textContent = '0%';
    confBarFill.style.width = '0%';
    stabilityTag.className = 'stability-tag unconfirmed';
    stabilityTag.textContent = '等待手勢';
    decisionStatus.innerHTML = '<span class="status-icon">⏳</span> 畫面無手部';
  } else if (currentCandidate === 'Unknown') {
    dockSignWord.textContent = '無法判讀';
    dockConfidence.textContent = `${confPct}%`;
    dockConfBarFill.style.width = `${confPct}%`;
    dockStabilityTag.className = 'stability-tag unconfirmed';
    dockStabilityTag.textContent = '未匹配';

    heroSignWord.textContent = '無法判讀';
    heroConfidence.textContent = `${confPct}%`;
    confBarFill.style.width = `${confPct}%`;
    stabilityTag.className = 'stability-tag unconfirmed';
    stabilityTag.textContent = '特徵距離過遠';
    decisionStatus.innerHTML = '<span class="status-icon">❓</span> 未在 519 詞庫中匹配';
  } else {
    dockSignWord.textContent = currentCandidate;
    dockConfidence.textContent = `${confPct}%`;
    dockConfBarFill.style.width = `${confPct}%`;
    dockStabilityTag.className = isStable ? 'stability-tag confirmed' : 'stability-tag unconfirmed';
    dockStabilityTag.textContent = isStable ? '已確認' : '確認中';

    heroSignWord.textContent = currentCandidate;
    heroConfidence.textContent = `${confPct}%`;
    confBarFill.style.width = `${confPct}%`;
    stabilityTag.className = isStable ? 'stability-tag confirmed' : 'stability-tag unconfirmed';
    stabilityTag.textContent = isStable ? '已確認 (穩定手勢)' : '確認中...';
    decisionStatus.innerHTML = isStable 
      ? '<span class="status-icon">✅</span> 成功鎖定手語詞彙' 
      : '<span class="status-icon">🔄</span> 手勢確認中';
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

// ===================== 9. 手語教室互動邏輯 =====================
function initClassroomCards() {
  lessonCardsContainer.innerHTML = '';
  CLASSROOM_LESSONS.forEach(lesson => {
    const card = document.createElement('div');
    card.className = `lesson-card ${lesson.word === currentPracticeTarget ? 'active' : ''}`;
    card.dataset.word = lesson.word;
    card.dataset.category = lesson.category;
    card.innerHTML = `
      <div class="lesson-word">${lesson.name}</div>
      <div class="lesson-cat">${getCategoryName(lesson.category)}</div>
    `;
    card.onclick = () => selectClassroomLesson(lesson);
    lessonCardsContainer.appendChild(card);
  });

  // 預設選中第一個單元
  selectClassroomLesson(CLASSROOM_LESSONS[0]);
}

function getCategoryName(cat) {
  const map = { numbers: '數字篇', greetings: '常用問候', daily: '生活用語', emotions: '情緒與感受' };
  return map[cat] || '一般詞彙';
}

function selectClassroomLesson(lesson) {
  currentPracticeTarget = lesson.word;
  practiceTargetWord.textContent = lesson.name;
  practiceTargetHint.textContent = lesson.hint;
  practiceMatchStreak = 0;

  document.querySelectorAll('.lesson-card').forEach(c => {
    c.classList.toggle('active', c.dataset.word === lesson.word);
  });

  practiceFeedbackBox.className = 'practice-feedback-box';
  feedbackIcon.textContent = '🎯';
  feedbackTitle.textContent = `請在鏡頭前比出【${lesson.name}】`;
  feedbackDesc.textContent = '系統正透過 126 維度空間座標比對您的手部姿態...';
  practiceMatchPercent.textContent = '0%';
  practiceMatchBar.style.width = '0%';
}

function handleClassroomFeedback(label, confidence) {
  if (!practiceTargetWord) return;

  const isMatched = label === currentPracticeTarget;
  if (isMatched) {
    practiceMatchStreak++;
    const pct = Math.min(100, Math.round(confidence * 100));
    practiceMatchPercent.textContent = `${pct}%`;
    practiceMatchBar.style.width = `${pct}%`;

    if (practiceMatchStreak >= 3) {
      practiceFeedbackBox.className = 'practice-feedback-box matched';
      feedbackIcon.textContent = '🎉';
      feedbackTitle.textContent = `太棒了！成功比出【${practiceTargetWord.textContent}】！`;
      feedbackDesc.textContent = `模型信心度達 ${pct}%，手勢姿勢標準！`;
      if (practiceMatchStreak === 3) {
        playSuccessChime();
      }
    }
  } else {
    practiceMatchStreak = 0;
    practiceMatchPercent.textContent = '0%';
    practiceMatchBar.style.width = '0%';
    practiceFeedbackBox.className = 'practice-feedback-box';
    feedbackIcon.textContent = '🎯';
    feedbackTitle.textContent = `請比出【${practiceTargetWord.textContent}】`;
    feedbackDesc.textContent = label !== 'No hand' && label !== 'Unknown' 
      ? `目前偵測到：${label}，請參照提示調整手勢。` 
      : '鏡頭偵測中，請比出手勢...';
  }
}

nextLessonBtn.addEventListener('click', () => {
  const currentIdx = CLASSROOM_LESSONS.findIndex(l => l.word === currentPracticeTarget);
  const nextIdx = (currentIdx + 1) % CLASSROOM_LESSONS.length;
  selectClassroomLesson(CLASSROOM_LESSONS[nextIdx]);
});

// 分類篩選
document.querySelectorAll('.cat-pill').forEach(pill => {
  pill.addEventListener('click', (e) => {
    document.querySelectorAll('.cat-pill').forEach(p => p.classList.remove('active'));
    e.target.classList.add('active');
    const cat = e.target.dataset.cat;
    document.querySelectorAll('.lesson-card').forEach(card => {
      if (cat === 'all' || card.dataset.category === cat) {
        card.style.display = 'flex';
      } else {
        card.style.display = 'none';
      }
    });
  });
});

// ===================== 10. AI 測驗闖關引擎 =====================
startQuizBtn.addEventListener('click', () => {
  const count = parseInt(quizCountSelect.value, 10) || 10;
  startQuizGame(count);
});

abandonQuizBtn.addEventListener('click', () => {
  if (confirm('確定要放棄本次測驗嗎？')) {
    stopQuizGame();
  }
});

retryQuizBtn.addEventListener('click', () => {
  quizSummaryModal.style.display = 'none';
  const count = parseInt(quizCountSelect.value, 10) || 10;
  startQuizGame(count);
});

viewStatsBtn.addEventListener('click', () => {
  quizSummaryModal.style.display = 'none';
  quizLobby.style.display = 'flex';
  document.querySelector('.nav-tab[data-tab="tab-stats"]').click();
});

function startQuizGame(count) {
  // 自動啟動攝影機以供測驗
  if (!isCameraRunning) {
    startWebcam();
  }

  isQuizActive = true;
  quizScore = 0;
  quizCombo = 0;
  quizMaxCombo = 0;
  quizCorrectCount = 0;
  currentQuizIdx = 0;
  quizMatchStreak = 0;

  // 隨機抽選題目（優先從有明確提示的精選庫抽，若不足從 519 庫補足）
  const pool = [...CLASSROOM_LESSONS];
  if (uniqueVocab.length > pool.length) {
    const remaining = uniqueVocab.filter(w => !pool.some(p => p.word === w));
    remaining.sort(() => Math.random() - 0.5);
    remaining.slice(0, 30).forEach(w => {
      pool.push({ word: w, name: w, category: 'vocab', hint: `請比出台灣手語【${w}】手勢。` });
    });
  }

  pool.sort(() => Math.random() - 0.5);
  quizQuestions = pool.slice(0, count);

  quizLobby.style.display = 'none';
  quizSummaryModal.style.display = 'none';
  quizArena.style.display = 'flex';

  quizTotalQ.textContent = quizQuestions.length;
  updateQuizHeader();
  nextQuizQuestion();
}

function stopQuizGame() {
  isQuizActive = false;
  clearInterval(quizTimerId);
  quizArena.style.display = 'none';
  quizLobby.style.display = 'flex';
}

function updateQuizHeader() {
  quizCurrentQ.textContent = currentQuizIdx + 1;
  quizCurrentScore.textContent = quizScore;
  quizComboBadge.textContent = `🔥 x${quizCombo}`;
  quizComboBadge.style.color = quizCombo > 2 ? '#00f0ff' : '#ff5555';
}

function nextQuizQuestion() {
  if (currentQuizIdx >= quizQuestions.length) {
    finishQuizGame();
    return;
  }

  const q = quizQuestions[currentQuizIdx];
  quizTargetWord.textContent = q.name;
  quizTargetHint.textContent = q.hint;
  quizMatchStreak = 0;
  quizTimeLeft = 10.0;
  updateQuizHeader();

  clearInterval(quizTimerId);
  timerBarFill.style.width = '100%';
  timerText.textContent = `${quizTimeLeft.toFixed(1)}s`;

  quizTimerId = setInterval(() => {
    quizTimeLeft -= 0.1;
    if (quizTimeLeft <= 0) {
      quizTimeLeft = 0;
      clearInterval(quizTimerId);
      onQuizTimeout();
    }
    const pct = Math.max(0, (quizTimeLeft / 10.0) * 100);
    timerBarFill.style.width = `${pct}%`;
    timerText.textContent = `${quizTimeLeft.toFixed(1)}s`;

    if (quizTimeLeft <= 3.0 && quizTimeLeft > 0) {
      playTickSound();
    }
  }, 100);
}

function handleQuizFrameMatching(label, confidence) {
  if (!isQuizActive || currentQuizIdx >= quizQuestions.length) return;

  const target = quizQuestions[currentQuizIdx].word;
  quizCurrentSeen.textContent = label === 'No hand' ? '無手部' : label;

  if (label === target && confidence >= 0.25) {
    quizMatchStreak++;
    const pct = Math.round(confidence * 100);
    quizCurrentConf.textContent = `${pct}%`;
    quizMatchingBar.style.width = `${pct}%`;

    if (quizMatchStreak >= 2) {
      clearInterval(quizTimerId);
      onQuizQuestionCorrect();
    }
  } else {
    quizMatchStreak = 0;
    quizCurrentConf.textContent = '0%';
    quizMatchingBar.style.width = '0%';
  }
}

function onQuizQuestionCorrect() {
  playSuccessChime();
  quizCorrectCount++;
  quizCombo++;
  if (quizCombo > quizMaxCombo) quizMaxCombo = quizCombo;

  // 分數計算：基礎 100 分 + 剩餘時間獎勵 + 連擊加成
  const timeBonus = Math.round(quizTimeLeft * 15);
  const comboBonus = quizCombo * 25;
  const gained = 100 + timeBonus + comboBonus;
  quizScore += gained;

  quizTargetWord.style.color = '#10b981';
  quizTargetWord.textContent = `答對！+${gained} 分 🎉`;

  setTimeout(() => {
    quizTargetWord.style.color = '#ffffff';
    currentQuizIdx++;
    nextQuizQuestion();
  }, 1000);
}

function onQuizTimeout() {
  playErrorBuzz();
  quizCombo = 0;
  quizTargetWord.style.color = '#ef4444';
  quizTargetWord.textContent = '時間到！❌';

  setTimeout(() => {
    quizTargetWord.style.color = '#ffffff';
    currentQuizIdx++;
    nextQuizQuestion();
  }, 1000);
}

function finishQuizGame() {
  isQuizActive = false;
  clearInterval(quizTimerId);
  quizArena.style.display = 'none';
  quizSummaryModal.style.display = 'flex';

  const accuracy = Math.round((quizCorrectCount / quizQuestions.length) * 100);
  summaryFinalScore.textContent = quizScore;
  summaryCorrectCount.textContent = `${quizCorrectCount} / ${quizQuestions.length}`;
  summaryAccuracy.textContent = `${accuracy}%`;
  summaryMaxCombo.textContent = `🔥 ${quizMaxCombo}`;

  let grade = 'S 級 (手語神手)';
  if (accuracy < 60) grade = 'C 級 (仍需努力)';
  else if (accuracy < 80) grade = 'B 級 (表現良好)';
  else if (accuracy < 95) grade = 'A 級 (手語達人)';
  summaryGrade.textContent = `成績結算：${grade}`;

  // 儲存至本地歷史歷程
  saveQuizResult({
    date: new Date().toLocaleString([], { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' }),
    totalQ: quizQuestions.length,
    score: quizScore,
    accuracy: accuracy,
    grade: grade
  });
}

// ===================== 11. 519 詞彙大字典 =====================
function initDictionaryGrid(vocabList) {
  dictionaryGrid.innerHTML = '';
  vocabList.forEach(word => {
    const card = document.createElement('div');
    card.className = 'dict-card';
    card.innerHTML = `<div class="dict-word">${word}</div>`;
    card.onclick = () => {
      // 點擊後跳轉到即時辨識並以該字作為目標
      document.querySelector('.nav-tab[data-tab="tab-recognition"]').click();
      heroSignWord.textContent = word;
      heroConfidence.textContent = '字典檢視';
      confBarFill.style.width = '100%';
    };
    dictionaryGrid.appendChild(card);
  });
  dictResultCount.textContent = `共收錄 ${vocabList.length} 個台灣手語詞彙`;
}

dictSearchInput.addEventListener('input', (e) => {
  const query = e.target.value.trim().toLowerCase();
  const matched = uniqueVocab.filter(w => w.toLowerCase().includes(query));
  initDictionaryGrid(matched);
  dictResultCount.textContent = `符合搜尋：${matched.length} / ${uniqueVocab.length} 個詞彙`;
});

// ===================== 12. 學習歷程與本地資料儲存 =====================
function getStoredHistory() {
  try {
    return JSON.parse(localStorage.getItem('sign_language_quiz_history')) || [];
  } catch (e) {
    return [];
  }
}

function saveQuizResult(result) {
  const history = getStoredHistory();
  history.unshift(result);
  if (history.length > 50) history.pop();
  try {
    localStorage.setItem('sign_language_quiz_history', JSON.stringify(history));
  } catch (e) {}
  renderStatsDashboard();
}

function renderStatsDashboard() {
  const history = getStoredHistory();
  statsTotalQuizzes.textContent = history.length;

  if (history.length === 0) {
    statsHighScore.textContent = '0';
    statsAvgAccuracy.textContent = '0%';
    statsMasteredWords.textContent = '0';
    historyTableBody.innerHTML = '<tr><td colspan="5" class="empty-hint">尚無測驗紀錄，前往「AI 測驗闖關」開始你的第一次挑戰！</td></tr>';
    return;
  }

  const highScore = Math.max(...history.map(h => h.score));
  const avgAcc = Math.round(history.reduce((acc, h) => acc + h.accuracy, 0) / history.length);
  const masteredEstimate = Math.min(uniqueVocab.length, Math.round(history.length * 4.5));

  statsHighScore.textContent = highScore;
  statsAvgAccuracy.textContent = `${avgAcc}%`;
  statsMasteredWords.textContent = masteredEstimate;

  let html = '';
  history.forEach(h => {
    html += `
      <tr>
        <td>${h.date}</td>
        <td>${h.totalQ} 題</td>
        <td><strong style="color:#00f0ff;">${h.score}</strong></td>
        <td>${h.accuracy}%</td>
        <td>${h.grade.split(' ')[0]}</td>
      </tr>
    `;
  });
  historyTableBody.innerHTML = html;
}

clearStatsBtn.addEventListener('click', () => {
  if (confirm('確定要清空所有的學習歷程與測驗紀錄嗎？')) {
    localStorage.removeItem('sign_language_quiz_history');
    renderStatsDashboard();
  }
});

// ===================== 13. 分頁切換器 =====================
document.querySelectorAll('.nav-tab').forEach(tabBtn => {
  tabBtn.addEventListener('click', (e) => {
    const targetId = tabBtn.dataset.tab;
    document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

    tabBtn.classList.add('active');
    const targetEl = document.getElementById(targetId);
    if (targetEl) targetEl.classList.add('active');

    // 切換時停止進行中的測驗
    if (targetId !== 'tab-quiz' && isQuizActive) {
      stopQuizGame();
    }
  });
});

// ===================== 14. 攝影機控制 =====================
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

// 影片檔案上傳支援
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

// 語句複製與清空
copySentenceBtn.addEventListener('click', () => {
  if (sentenceList.length === 0) return;
  const text = sentenceList.map(item => item.word).join(' ');
  navigator.clipboard.writeText(text).then(() => {
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

// FPS 定時器
setInterval(() => {
  fps = frameCount;
  frameCount = 0;
  fpsVal.textContent = fps;
  latencyVal.textContent = latency;
}, 1000);

// 自啟動
window.addEventListener('DOMContentLoaded', async () => {
  canvasElement.classList.add('mirror');
  initMediaPipeHands();
  const loaded = await loadKNNModel();
  if (loaded) {
    loadingOverlay.classList.remove('hidden');
    loadingMsg.textContent = '模型載入完成！請點擊「啟動攝影機」或選擇上方分頁開始體驗。';
  }
});
