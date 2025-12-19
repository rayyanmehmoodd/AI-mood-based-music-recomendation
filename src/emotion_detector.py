"""
Emotion Detection Module using Computer Vision
Real-time facial emotion recognition using DeepFace or mini_XCEPTION.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, List
import os

# Try to import TensorFlow
TF_AVAILABLE = False
try:
    import tensorflow as tf
    # Suppress TF warnings
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    TF_AVAILABLE = True
except ImportError:
    pass

# Try to import DeepFace
DEEPFACE_AVAILABLE = False
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    pass


class EmotionDetector:
    """
    Real-time facial emotion detection using CNN.
    Uses DeepFace with proper face detection for accurate results.
    """
    
    # Standard emotion classes (FER-2013)
    EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    # Emotion to music mood mapping
    EMOTION_MOOD_MAP = {
        'Happy': 'happy_energetic',
        'Sad': 'sad_melancholic', 
        'Angry': 'intense_powerful',
        'Fear': 'calm_soothing',
        'Surprise': 'upbeat_exciting',
        'Disgust': 'neutral_ambient',
        'Neutral': 'balanced_mixed'
    }
    
    def __init__(self, use_deepface: bool = True, model_path: Optional[str] = None, 
                 cascade_path: Optional[str] = None, smoothing_window: int = 3):
        """Initialize the EmotionDetector."""
        self.dnn_net = None
        self.model = None
        self.use_deepface = use_deepface and DEEPFACE_AVAILABLE and TF_AVAILABLE
        
        # Temporal smoothing for stable predictions (reduced window for responsiveness)
        self.smoothing_window = smoothing_window
        self.emotion_history = []  # Store recent emotion scores
        
        # Initialize Haar Cascade face detector
        cascade_file = cascade_path or cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_file)
        
        # Initialize DNN face detector (more accurate)
        self._init_dnn_face_detector()
        
        # Try loading mini_XCEPTION if not using DeepFace
        if not self.use_deepface and TF_AVAILABLE:
            self._load_mini_xception()
        
        status = []
        if self.use_deepface:
            status.append("DeepFace")
        elif self.model is not None:
            status.append("mini_XCEPTION")
        if self.dnn_net is not None:
            status.append("DNN-FaceDetect")
        
        print(f"✓ EmotionDetector initialized ({', '.join(status) or 'Fallback mode'})")
    
    def _init_dnn_face_detector(self):
        """Initialize OpenCV DNN face detector."""
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'emotion')
        prototxt = os.path.join(model_dir, 'deploy.prototxt')
        caffemodel = os.path.join(model_dir, 'res10_300x300_ssd_iter_140000.caffemodel')
        
        if os.path.exists(prototxt) and os.path.exists(caffemodel):
            try:
                self.dnn_net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
            except Exception:
                pass
    
    def _load_mini_xception(self):
        """Load mini_XCEPTION model."""
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'emotion')
        model_path = os.path.join(model_dir, 'fer2013_mini_XCEPTION.hdf5')
        
        if os.path.exists(model_path):
            try:
                from tensorflow.keras.models import load_model
                self.model = load_model(model_path, compile=False)
            except Exception as e:
                print(f"✗ Could not load mini_XCEPTION: {e}")
    
    def detect_faces_dnn(self, frame: np.ndarray, confidence_threshold: float = 0.6) -> List[Tuple[int, int, int, int]]:
        """Detect faces using DNN model (more accurate)."""
        if self.dnn_net is None:
            return []
        
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.dnn_net.setInput(blob)
        detections = self.dnn_net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                # Ensure valid coordinates
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 > x1 and y2 > y1:
                    faces.append((x1, y1, x2-x1, y2-y1))
        
        return faces
    
    def detect_faces_haar(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using Haar Cascade."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(48, 48))
        return list(faces) if len(faces) > 0 else []
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using best available method."""
        faces = self.detect_faces_dnn(frame)
        if not faces:
            faces = self.detect_faces_haar(frame)
        return faces
    
    def detect_emotion(self, frame: np.ndarray, use_smoothing: bool = False) -> Dict:
        """
        Main emotion detection method.
        
        Args:
            frame: BGR image from OpenCV
            use_smoothing: Whether to apply temporal smoothing (disabled by default for consistency)
            
        Returns:
            Dictionary with emotion predictions
        """
        # Get raw prediction - smoothing disabled by default for consistency
        if self.use_deepface:
            result = self._detect_deepface(frame)
        elif self.model is not None:
            result = self._detect_mini_xception(frame)
        else:
            result = self._detect_heuristic(frame)
        
        # Apply temporal smoothing only if explicitly enabled
        if use_smoothing and result.get('face_detected', False):
            result = self._apply_smoothing(result)
        
        return result
    
    def _apply_smoothing(self, result: Dict) -> Dict:
        """Apply temporal smoothing to stabilize emotion predictions."""
        emotion_scores = result.get('emotion_scores', {})
        current_dominant = result.get('dominant_emotion', 'Neutral')
        
        # Add current scores to history
        self.emotion_history.append(emotion_scores.copy())
        
        # Keep only last 3 frames (smaller window = more responsive)
        if len(self.emotion_history) > 3:
            self.emotion_history = self.emotion_history[-3:]
        
        # Need at least 2 frames to smooth
        if len(self.emotion_history) < 2:
            return result
        
        # Weighted average: recent frames matter more
        # Weights: [1, 2, 3] for 3 frames (most recent = 3)
        averaged_scores = {e: 0.0 for e in self.EMOTIONS}
        weights = list(range(1, len(self.emotion_history) + 1))
        total_weight = sum(weights)
        
        for weight, scores in zip(weights, self.emotion_history):
            for emotion in self.EMOTIONS:
                averaged_scores[emotion] += scores.get(emotion, 0) * weight
        
        # Normalize
        for emotion in averaged_scores:
            averaged_scores[emotion] /= total_weight
        
        # Find dominant
        dominant_emotion = max(averaged_scores.items(), key=lambda x: x[1])
        
        result['emotion_scores'] = averaged_scores
        result['dominant_emotion'] = dominant_emotion[0]
        result['confidence'] = dominant_emotion[1]
        result['smoothed'] = True
        
        return result
    
    def reset_smoothing(self):
        """Reset the emotion history for fresh predictions."""
        self.emotion_history = []
    
    def _detect_deepface(self, frame: np.ndarray) -> Dict:
        """
        Detect emotion using DeepFace - based on manish-9245's implementation.
        Reference: https://github.com/manish-9245/Facial-Emotion-Recognition-using-OpenCV-and-Deepface
        """
        # Convert frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Convert grayscale frame to RGB format (as per the repo's approach)
        rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
        
        # Detect faces using Haar Cascade
        faces = self.face_cascade.detectMultiScale(
            gray_frame, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return {
                'dominant_emotion': 'Neutral',
                'emotion_scores': {e: 1/7 for e in self.EMOTIONS},
                'confidence': 0.0,
                'face_detected': False
            }
        
        # Get largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        
        # Extract the face ROI (Region of Interest) - exactly as in the repo
        face_roi = rgb_frame[y:y + h, x:x + w]
        
        if face_roi.size == 0:
            return {
                'dominant_emotion': 'Neutral',
                'emotion_scores': {e: 1/7 for e in self.EMOTIONS},
                'confidence': 0.0,
                'face_detected': False
            }
        
        try:
            # Perform emotion analysis on the face ROI - exactly as in the repo
            result = DeepFace.analyze(
                face_roi, 
                actions=['emotion'], 
                enforce_detection=False
            )
            
            # Handle list result (DeepFace returns list)
            if isinstance(result, list):
                result = result[0] if result else {}
            
            # Determine the dominant emotion - exactly as in the repo
            dominant = result.get('dominant_emotion', 'neutral')
            emotions = result.get('emotion', {})
            
            # Normalize emotion scores (DeepFace returns 0-100, we use 0-1)
            emotion_scores = {}
            for e in self.EMOTIONS:
                key = e.lower()
                emotion_scores[e] = emotions.get(key, 0) / 100.0
            
            confidence = emotion_scores.get(dominant.capitalize(), 0)
            
            return {
                'dominant_emotion': dominant.capitalize(),
                'emotion_scores': emotion_scores,
                'confidence': confidence,
                'face_detected': True,
                'face_region': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}
            }
            
        except Exception as e:
            return {
                'dominant_emotion': 'Neutral',
                'emotion_scores': {em: 1/7 for em in self.EMOTIONS},
                'confidence': 0.0,
                'face_detected': True,
                'error': str(e)
            }
    
    def _detect_mini_xception(self, frame: np.ndarray) -> Dict:
        """Detect emotion using mini_XCEPTION model."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detect_faces(frame)
        
        if not faces:
            return {
                'dominant_emotion': 'Neutral',
                'emotion_scores': {e: 1/7 for e in self.EMOTIONS},
                'confidence': 0.0,
                'face_detected': False
            }
        
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        
        # Extract face ROI
        face_roi = gray[y:y+h, x:x+w]
        if face_roi.size == 0:
            return {
                'dominant_emotion': 'Neutral',
                'emotion_scores': {e: 1/7 for e in self.EMOTIONS},
                'confidence': 0.0,
                'face_detected': False
            }
        
        # Preprocess: resize to 64x64, normalize
        face_resized = cv2.resize(face_roi, (64, 64))
        face_input = face_resized.astype('float32') / 255.0
        face_input = np.expand_dims(np.expand_dims(face_input, 0), -1)
        
        # Predict
        predictions = self.model.predict(face_input, verbose=0)[0]
        
        emotion_scores = {self.EMOTIONS[i]: float(predictions[i]) for i in range(7)}
        dominant_idx = np.argmax(predictions)
        
        return {
            'dominant_emotion': self.EMOTIONS[dominant_idx],
            'emotion_scores': emotion_scores,
            'confidence': float(predictions[dominant_idx]),
            'face_detected': True,
            'face_region': {'x': x, 'y': y, 'w': w, 'h': h}
        }
    
    def _detect_heuristic(self, frame: np.ndarray) -> Dict:
        """Fallback heuristic-based detection using facial features."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detect_faces(frame)
        
        if not faces:
            return {
                'dominant_emotion': 'Neutral',
                'emotion_scores': {e: 1/7 for e in self.EMOTIONS},
                'confidence': 0.3,
                'face_detected': False
            }
        
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_roi = gray[y:y+h, x:x+w]
        
        if face_roi.size == 0:
            return {
                'dominant_emotion': 'Neutral',
                'emotion_scores': {e: 1/7 for e in self.EMOTIONS},
                'confidence': 0.3,
                'face_detected': False
            }
        
        # Simple feature analysis
        face_resized = cv2.resize(face_roi, (48, 48))
        
        # Analyze different regions
        mouth = face_resized[36:, :]
        eyes = face_resized[12:24, :]
        
        mouth_edges = cv2.Canny(mouth, 50, 150)
        eye_edges = cv2.Canny(eyes, 50, 150)
        
        mouth_activity = np.sum(mouth_edges > 0) / mouth_edges.size
        eye_activity = np.sum(eye_edges > 0) / eye_edges.size
        overall_std = np.std(face_resized)
        
        # Score emotions based on features
        scores = np.array([0.1, 0.05, 0.1, 0.15, 0.15, 0.1, 0.35])  # Base: Neutral
        
        if mouth_activity > 0.2:  # Smile likely
            scores[3] += 0.3  # Happy
            scores[6] -= 0.1
        if eye_activity > 0.25:  # Wide eyes
            scores[5] += 0.2  # Surprise
        if overall_std > 50:  # High contrast
            scores[0] += 0.15  # Angry
        if overall_std < 35:  # Low contrast
            scores[4] += 0.2  # Sad
        
        scores = scores / scores.sum()
        dominant_idx = np.argmax(scores)
        
        return {
            'dominant_emotion': self.EMOTIONS[dominant_idx],
            'emotion_scores': dict(zip(self.EMOTIONS, scores.tolist())),
            'confidence': float(scores[dominant_idx]),
            'face_detected': True,
            'face_region': {'x': x, 'y': y, 'w': w, 'h': h}
        }
    
    def get_music_mood(self, emotion: str) -> str:
        """Map detected emotion to music mood category."""
        return self.EMOTION_MOOD_MAP.get(emotion, 'balanced_mixed')
    
    def process_frame(self, frame: np.ndarray, draw_annotations: bool = True) -> Tuple[np.ndarray, Dict]:
        """Process a video frame: detect face and emotion."""
        result = self.detect_emotion(frame)
        
        if draw_annotations:
            frame = self._draw_annotations(frame, result)
            
        return frame, result
    
    def _draw_annotations(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """Draw face bounding box and emotion label on frame."""
        annotated = frame.copy()
        
        region = result.get('face_region', {})
        if region:
            x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 0), region.get('h', 0)
            cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            emotion = result.get('dominant_emotion', 'Unknown')
            conf = result.get('confidence', 0)
            label = f"{emotion} ({conf*100:.0f}%)"
            cv2.putText(annotated, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                       
        return annotated
    
    def capture_single_emotion(self, num_samples: int = 3) -> Dict:
        """
        Capture multiple frames from webcam and detect emotion using majority voting.
        This provides more stable results than single-frame detection.
        
        Args:
            num_samples: Number of frames to sample and average (default=3)
        
        Returns:
            Emotion detection result with averaged scores
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            return {'error': 'Could not open webcam'}
        
        # Warm up camera
        for _ in range(10):
            cap.read()
        
        # Collect multiple samples
        all_scores = {e: [] for e in self.EMOTIONS}
        valid_samples = 0
        
        for i in range(num_samples):
            ret, frame = cap.read()
            if ret:
                result = self.detect_emotion(frame, use_smoothing=False)
                if result.get('face_detected'):
                    valid_samples += 1
                    scores = result.get('emotion_scores', {})
                    for emotion in self.EMOTIONS:
                        all_scores[emotion].append(scores.get(emotion, 0))
            
            # Small delay between captures
            import time
            time.sleep(0.2)
        
        cap.release()
        
        if valid_samples == 0:
            return {'error': 'No face detected in any sample'}
        
        # Average the scores across all valid samples
        averaged_scores = {}
        for emotion in self.EMOTIONS:
            if all_scores[emotion]:
                averaged_scores[emotion] = sum(all_scores[emotion]) / len(all_scores[emotion])
            else:
                averaged_scores[emotion] = 0.0
        
        # Find dominant emotion
        dominant = max(averaged_scores.items(), key=lambda x: x[1])
        
        return {
            'dominant_emotion': dominant[0],
            'emotion_scores': averaged_scores,
            'confidence': dominant[1],
            'face_detected': True,
            'samples_used': valid_samples
        }
    
    def start_webcam(self, callback=None, window_name: str = "Emotion Detection"):
        """Start real-time webcam emotion detection."""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("✗ Could not open webcam")
            return
            
        print("✓ Webcam started. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            annotated_frame, result = self.process_frame(frame)
            
            if callback and result.get('face_detected'):
                mood = self.get_music_mood(result['dominant_emotion'])
                callback(result['dominant_emotion'], mood)
                
            cv2.imshow(window_name, annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
