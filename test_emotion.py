import sys
sys.path.insert(0, '.')
from src.emotion_detector import EmotionDetector
import cv2
import time

print("Testing improved smoothing - try different expressions!")
detector = EmotionDetector()

cap = cv2.VideoCapture(0)
for _ in range(15): 
    cap.read()

for i in range(8):
    ret, frame = cap.read()
    if ret:
        r = detector.detect_emotion(frame, use_smoothing=True)
        sm = "(smoothed)" if r.get("smoothed") else "(raw)"
        print(f"{i+1}. {r['dominant_emotion']:10} {r['confidence']*100:5.1f}% {sm}")
    time.sleep(0.8)

cap.release()
print("Done!")
