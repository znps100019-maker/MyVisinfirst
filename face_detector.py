import math
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

class FaceExpressionRecognizer:
    """Detect facial landmarks and classify static facial expressions."""
    
    def __init__(self, max_num_faces=1):
        # refine_landmarks=True to get refined points around eyes and lips
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def process(self, frame):
        """Processes the image and returns a dict with raw landmarks and detected expression."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.face_mesh.process(rgb)
        
        if not results.multi_face_landmarks:
            return None
            
        landmarks = results.multi_face_landmarks[0]
        pts = landmarks.landmark
        
        # 1. Eye Aspect Ratio (EAR) for left and right eyes
        # Left eye landmarks: 362 (inner), 385 (top-right), 386 (top-left), 263 (outer), 374 (bottom-left), 380 (bottom-right)
        left_ear = self._eye_aspect_ratio(pts, 362, 385, 386, 263, 374, 380)
        # Right eye landmarks: 33 (outer), 159 (top-left), 158 (top-right), 133 (inner), 145 (bottom-right), 153 (bottom-left)
        right_ear = self._eye_aspect_ratio(pts, 33, 159, 158, 133, 145, 153)
        
        # A threshold of 0.20 determines eye closure
        is_left_closed = left_ear < 0.20
        is_right_closed = right_ear < 0.20
        
        # 2. Smile detection
        # Mouth width (61 to 291)
        d_mouth = self._distance(pts[61], pts[291])
        # Distance between outer corners of eyes (33 to 263) as scale reference
        d_eyes = self._distance(pts[33], pts[263])
        smile_ratio = d_mouth / max(d_eyes, 0.001)
        is_smiling = smile_ratio > 0.56
        
        # 3. Mouth open detection (Surprise)
        # Inner mouth height (13 to 14)
        d_inner_mouth = self._distance(pts[13], pts[14])
        # Face height (Forehead 10 to Chin 152)
        d_face_height = self._distance(pts[10], pts[152])
        mouth_ratio = d_inner_mouth / max(d_face_height, 0.001)
        is_mouth_open = mouth_ratio > 0.08
        
        # 4. Expressions Decision Tree
        expression = "Neutral"
        if is_left_closed and is_right_closed:
            expression = "Blink"
        elif is_left_closed:
            expression = "Wink Left"
        elif is_right_closed:
            expression = "Wink Right"
        elif is_mouth_open:
            expression = "Mouth Open"
        elif is_smiling:
            expression = "Smiling"
            
        return {
            "landmarks": landmarks,
            "expression": expression,
            "left_ear": round(left_ear, 3),
            "right_ear": round(right_ear, 3),
            "smile_ratio": round(smile_ratio, 3),
            "mouth_ratio": round(mouth_ratio, 3)
        }
        
    def draw(self, frame, face_data):
        """Draw face mesh landmarks on the frame."""
        if not face_data:
            return
            
        # Tesselation (fine mesh structure) - light cyan/teal in BGR
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_data["landmarks"],
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(240, 240, 100), thickness=1, circle_radius=1)
        )
        
        # Contours (eyes, lips, face shape outline) - vibrant orange in BGR
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_data["landmarks"],
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 120, 255), thickness=1, circle_radius=1)
        )
        
    def close(self):
        self.face_mesh.close()
        
    def _distance(self, p1, p2):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
        
    def _eye_aspect_ratio(self, points, p1, p2, p3, p4, p5, p6):
        d_vert1 = self._distance(points[p2], points[p6])
        d_vert2 = self._distance(points[p3], points[p5])
        d_horiz = self._distance(points[p1], points[p4])
        return (d_vert1 + d_vert2) / max(2.0 * d_horiz, 0.001)
