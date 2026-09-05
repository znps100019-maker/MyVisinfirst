import os
import sys
import cv2
import math

# Reuse the non-ascii path bypass from main.py
def handle_non_ascii_path():
    if sys.platform != "win32":
        return
    project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    if any(ord(c) > 127 for c in project_root):
        import string
        drive = None
        for letter in string.ascii_uppercase[::-1]:
            candidate = f"{letter}:"
            if not os.path.exists(candidate + "\\"):
                drive = candidate
                break
        if not drive:
            sys.exit(1)
        import subprocess
        subprocess.run(["subst", drive, project_root], shell=True, stdout=subprocess.DEVNULL)
        relative_script = os.path.relpath(os.path.abspath(__file__), project_root)
        virtual_script = os.path.join(drive, relative_script)
        virtual_python = os.path.join(drive, ".venv", "Scripts", "python.exe")
        if not os.path.exists(virtual_python):
            virtual_python = sys.executable.replace(project_root, drive)
        args = [virtual_python, virtual_script] + sys.argv[1:]
        try:
            result = subprocess.run(args)
            returncode = result.returncode
        finally:
            subprocess.run(["subst", drive, "/d"], shell=True, stdout=subprocess.DEVNULL)
        sys.exit(returncode)

handle_non_ascii_path()

import mediapipe as mp
mp_hands = mp.solutions.hands

def distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.6)
    
    print("Ratio Diagnostic Tool Started. Press q to quit.")
    print("Formulas:")
    print("  Ratio = Distance(Wrist, Tip) / Distance(Wrist, MCP)")
    print("  StraightRatio = Distance(MCP, Tip) / (MCP-PIP + PIP-DIP + DIP-TIP)")
    print("----------------------------------------------------------------")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                pts = landmarks.landmark
                wrist = pts[0]
                
                finger_names = ["index", "middle", "ring", "pinky"]
                mcps = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}
                tips = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}
                
                print_str = ""
                for f in finger_names:
                    mcp = pts[mcps[f]]
                    pip = pts[mcps[f]+1]
                    dip = pts[mcps[f]+2]
                    tip = pts[tips[f]]
                    
                    # Wrist-based ratio
                    d_wrist_tip = distance(wrist, tip)
                    d_wrist_mcp = distance(wrist, mcp)
                    ratio_wrist = d_wrist_tip / max(d_wrist_mcp, 0.001)
                    
                    # Straight ratio (existing method)
                    d_straight = distance(mcp, tip)
                    d_segments = distance(mcp, pip) + distance(pip, dip) + distance(dip, tip)
                    ratio_straight = d_straight / max(d_segments, 0.001)
                    
                    print_str += f"{f[:3]}: W_T={ratio_wrist:.2f} S={ratio_straight:.2f} | "
                
                # Also calculate thumb spread ratio
                thumb_tip = pts[4]
                index_mcp = pts[5]
                middle_mcp = pts[9]
                d_thumb_spread = distance(thumb_tip, index_mcp) / max(distance(wrist, middle_mcp), 0.001)
                
                print_str += f"ThumbSpread: {d_thumb_spread:.2f}"
                print(print_str, end="\r")
                
        cv2.imshow("Diagnostics", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
