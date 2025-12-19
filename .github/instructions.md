# Copilot Instructions for AI Emotion-Based Music Recommendation System

## Project Overview
This is a multimodal AI system combining Computer Vision (emotion detection), Unsupervised Learning (music clustering), and Supervised Learning (popularity prediction) to recommend music based on facial expressions.

## Architecture & Data Flow
```
Webcam → EmotionDetector (CNN) → detected_emotion
                                        ↓
Spotify Data → DataProcessor → MusicClusterer → cluster_profiles
                                        ↓
                              EmotionMusicMapper → recommended_songs
                                        ↓
                              PopularityPredictor → ranked_playlist
                                        ↓
                              Streamlit App → User Interface
```

## Key Modules & Their Roles

### `src/data_processor.py`
- Handles Spotify CSV data loading and preprocessing
- Key features: `danceability, energy, loudness, acousticness, valence, tempo`
- Use `StandardScaler` for normalization before clustering/prediction
- `CLUSTERING_FEATURES` vs `AUDIO_FEATURES` - clustering uses subset for better separation

### `src/clustering.py` - MusicClusterer
- Primary: K-Means (best for this use case)
- Also supports: Hierarchical, DBSCAN
- Always run `find_optimal_k()` before final clustering
- PCA to 2D for visualization, but cluster on full features
- Cluster profiles map to mood quadrants (energy × valence)

### `src/emotion_detector.py` - EmotionDetector
- Prefers DeepFace (pre-trained) over custom CNN
- Falls back to Haar Cascades for face detection
- 7 emotion classes: Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral
- `capture_single_emotion()` for one-shot, `start_webcam()` for continuous

### `src/popularity_predictor.py` - PopularityPredictor
- Random Forest typically performs best for this task
- ANN requires validation split for early stopping
- Popularity prediction is inherently noisy (R² ~0.15-0.25 is normal)
- Feature importance: energy, loudness, danceability usually top

### `src/emotion_music_mapper.py` - EmotionMusicMapper
- Maps emotions to audio feature profiles (e.g., Happy → high valence + high energy)
- `EMOTION_AUDIO_PROFILE` defines target ranges for each emotion
- `recommend_songs()` filters by cluster then sorts by popularity

## Development Patterns

### Adding New Clustering Algorithm
```python
# In clustering.py, follow pattern:
def fit_new_algo(self, X, **params):
    model = NewAlgorithm(**params)
    self.cluster_labels = model.fit_predict(X)
    self.models['new_algo'] = model
    return self.cluster_labels
```

### Adding New Prediction Model
```python
# In popularity_predictor.py:
def train_new_model(self, X_train, y_train):
    model = NewModel()
    model.fit(X_train, y_train)
    self.models['new_model'] = model
```

### Emotion-Cluster Mapping
```python
# In emotion_music_mapper.py, EMOTION_AUDIO_PROFILE format:
'Emotion': {
    'feature': (min_value, max_value),  # Target range
}
```

## Testing & Debugging

### Generate Test Data
```python
from src.data_processor import DataProcessor
processor = DataProcessor()
df = processor.create_sample_data(n_samples=1000)  # Synthetic data
```

### Check Webcam
```python
import cv2
cap = cv2.VideoCapture(0)
print("Webcam available:", cap.isOpened())
cap.release()
```

### Validate Clustering
```python
from sklearn.metrics import silhouette_score
score = silhouette_score(X, labels)  # Should be > 0.2
```

## Running the Project

### Notebooks (for analysis)
```bash
cd notebooks/
jupyter notebook
# Run: 01_eda_analysis.ipynb → 02_clustering_analysis.ipynb → 03_popularity_prediction.ipynb
```

### Streamlit App
```bash
streamlit run app/streamlit_app.py
# Access at http://localhost:8501
```

## Common Issues & Solutions

1. **DeepFace import error**: Install with `pip install deepface tf-keras`
2. **Webcam not opening**: Check permissions, try different camera index (0, 1, 2)
3. **Low clustering scores**: Normalize features, try different k values
4. **Slow ANN training**: Reduce epochs, use GPU if available
5. **Streamlit webcam issues**: Use `cv2.VideoCapture(0)` not higher indices

## Data Paths Convention
- Raw data: `data/raw/`
- Processed data: `data/processed/`
- Saved models: `models/{clustering,emotion,popularity}/`
- Model naming: `{algorithm}_model.pkl` or `.keras` for TF models

## Code Style
- Type hints for function parameters
- Docstrings with Args/Returns sections
- Print status with ✓ or ✗ prefixes
- Use `Optional[]` for nullable parameters
