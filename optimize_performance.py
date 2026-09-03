"""效能優化 - 添加幀跳過和快取機制"""
with open('main.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 在 process_video_loop 中添加效能優化
old_loop = '''def process_video_loop(cap, sign_recognizer, face_recognizer, args, target_sign):
    last_print_time = 0
    last_target_time = 0
    frame_count = 0
    show_face_mesh = True'''

new_loop = '''def process_video_loop(cap, sign_recognizer, face_recognizer, args, target_sign):
    last_print_time = 0
    last_target_time = 0
    frame_count = 0
    show_face_mesh = True
    skip_frames = 0  # 跳過幀計數器
    process_every_n_frames = 1  # 每 N 幀處理一次（1 = 不跳過）'''

if old_loop in content:
    content = content.replace(old_loop, new_loop)
    print("✓ 已添加效能優化變數")
else:
    print("✗ 未找到主迴圈")

with open('main.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("\n效能優化完成！")
