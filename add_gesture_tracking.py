"""添加手勢追蹤和連貫性判斷"""
with open('hand_detector.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 在 __init__ 中添加手勢追蹤
old_init = '''    def __init__(self, max_num_hands=2, history_size=8, stable_min_count=3):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.4,
        )
        self.history = deque(maxlen=history_size)
        self.stable_min_count = stable_min_count
        self.sentence = []
        self.last_added_sign = None'''

new_init = '''    def __init__(self, max_num_hands=2, history_size=8, stable_min_count=3):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.4,
        )
        self.history = deque(maxlen=history_size)
        self.stable_min_count = stable_min_count
        self.sentence = []
        self.last_added_sign = None
        self.gesture_history = []  # 完整的手勢歷史
        self.gesture_start_time = None  # 目前手勢開始時間'''

if old_init in content:
    content = content.replace(old_init, new_init)
    print("✓ 已添加手勢追蹤初始化")
else:
    print("✗ 未找到初始化程式碼")

# 在 process 方法中添加手勢歷史記錄
old_process = '''    def process(self, frame):
        """Return detected hands with landmarks, handedness, and sign labels."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)'''

new_process = '''    def process(self, frame):
        """Return detected hands with landmarks, handedness, and sign labels."""
        import time
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        current_time = time.time()'''

if old_process in content:
    content = content.replace(old_process, new_process)
    print("✓ 已添加時間追蹤")
else:
    print("✗ 未找到 process 方法")

with open('hand_detector.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("\n手勢追蹤添加完成！")
