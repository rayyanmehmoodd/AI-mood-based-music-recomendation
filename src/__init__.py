# AI Emotion-Based Music Recommendation System
# Core source modules

from .data_processor import DataProcessor
from .clustering import MusicClusterer
from .emotion_detector import EmotionDetector
from .popularity_predictor import PopularityPredictor
from .emotion_music_mapper import EmotionMusicMapper

__all__ = [
    'DataProcessor',
    'MusicClusterer', 
    'EmotionDetector',
    'PopularityPredictor',
    'EmotionMusicMapper'
]

__version__ = '1.0.0'
