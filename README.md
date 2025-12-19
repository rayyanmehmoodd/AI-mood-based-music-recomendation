# 🎵 AI Emotion-Based Music Recommendation System

An intelligent multimodal AI system that combines **Computer Vision**, **Unsupervised Learning**, and **Supervised Learning** to deliver personalized music recommendations based on real-time facial emotion detection.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🌟 Features

- **Real-Time Emotion Detection**: CNN-based facial emotion recognition using webcam
- **Smart Music Clustering**: K-Means, Hierarchical, and DBSCAN algorithms for song grouping
- **Popularity Prediction**: Multiple ML models (Linear Regression, Random Forest, ANN)
- **Emotion-Music Mapping**: Intelligent bridging of detected emotions to music clusters
- **Interactive Dashboard**: Streamlit-based web interface for seamless user experience

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERFACE (Streamlit)               │
└─────────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            │                 │                 │
            ▼                 ▼                 ▼
┌───────────────────┐ ┌───────────────┐ ┌───────────────────┐
│ Emotion Detection │ │   Clustering  │ │    Popularity     │
│       (CNN)       │ │   (K-Means)   │ │    Prediction     │
└───────────────────┘ └───────────────┘ └───────────────────┘
            │                 │                 │
            └─────────────────┼─────────────────┘
                              │
                              ▼
            ┌─────────────────────────────────────┐
            │       Emotion-Music Mapper          │
            └─────────────────────────────────────┘
                              │
                              ▼
            ┌─────────────────────────────────────┐
            │    Personalized Playlist Output     │
            └─────────────────────────────────────┘
```

## 📁 Project Structure

```
AI proj/
├── app/
│   └── streamlit_app.py      # Main web application
├── data/
│   ├── raw/                  # Original datasets (any CSV!)
│   └── processed/            # Cleaned & processed data
├── models/
│   ├── clustering/           # Saved clustering models
│   ├── emotion/              # CNN emotion models
│   └── popularity/           # Prediction models
├── notebooks/
│   ├── 01_eda_analysis.ipynb         # Exploratory Data Analysis
│   ├── 02_clustering_analysis.ipynb   # Clustering experiments
│   └── 03_popularity_prediction.ipynb # Model comparison
├── src/
│   ├── __init__.py
│   ├── data_processor.py     # Flexible data loading with auto-detection
│   ├── clustering.py         # Clustering algorithms
│   ├── emotion_detector.py   # CNN emotion detection
│   ├── popularity_predictor.py # Regression models
│   ├── emotion_music_mapper.py # Emotion-to-cluster mapping
│   └── api_integrations.py   # Spotify & Last.fm API clients
├── config.py                 # API keys configuration
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
cd "AI proj"

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

**Option A: Spotify Dataset (Recommended)**
Download the Spotify dataset from Kaggle:
- [Spotify Tracks Dataset](https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db)

Place the CSV file in `data/raw/spotify_data.csv`

**Option B: Any Music CSV Dataset**
The system supports **any CSV music dataset** with automatic column detection! Just ensure your CSV has:
- Track/song names
- Numeric audio features (any combination works)

Supported column name variations:
| Feature | Accepted Names |
|---------|----------------|
| Track Name | `track_name`, `name`, `song`, `title`, `track` |
| Artist | `artist`, `artist_name`, `artists`, `performer` |
| Energy | `energy`, `Energy`, `energy_level` |
| Valence | `valence`, `positivity`, `mood`, `happiness` |
| Tempo | `tempo`, `bpm`, `beats_per_minute`, `Tempo` |
| Popularity | `popularity`, `streams`, `play_count`, `listens` |

### 3. API Setup (Optional)

For enhanced features like fetching track metadata, configure free APIs:

**Create `config.py` in project root:**
```python
# Spotify API (get from https://developer.spotify.com/dashboard)
SPOTIFY_CLIENT_ID = "your_client_id_here"
SPOTIFY_CLIENT_SECRET = "your_client_secret_here"

# Last.fm API (get from https://www.last.fm/api/account/create)
LASTFM_API_KEY = "your_api_key_here"
```

**Getting Free API Keys:**

1. **Spotify API** (Free):
   - Go to https://developer.spotify.com/dashboard
   - Log in or create account
   - Click "Create App"
   - Fill in any name/description
   - Copy Client ID and Client Secret

2. **Last.fm API** (Free):
   - Go to https://www.last.fm/api/account/create
   - Create an account if needed
   - Fill out API application form
   - Copy your API key

### 4. Run Analysis Notebooks

```bash
# Start Jupyter
jupyter notebook notebooks/
```

Run notebooks in order:
1. `01_eda_analysis.ipynb` - Explore the data
2. `02_clustering_analysis.ipynb` - Train clustering models
3. `03_popularity_prediction.ipynb` - Train prediction models

### 4. Run Analysis Notebooks

```bash
# Start Jupyter
jupyter notebook notebooks/
```

Run notebooks in order:
1. `01_eda_analysis.ipynb` - Explore the data
2. `02_clustering_analysis.ipynb` - Train clustering models
3. `03_popularity_prediction.ipynb` - Train prediction models

### 5. Launch Web Application

```bash
streamlit run app/streamlit_app.py
```

Open http://localhost:8501 in your browser.

**Using the Web App:**
- Upload any music CSV file directly in the app
- The system auto-detects columns and maps them
- Use webcam for emotion detection or manually select emotions
- Get personalized music recommendations!

## 📊 Datasets

### Flexible CSV Support
The system is designed to work with **any music dataset**! It automatically:
- Detects column names using fuzzy matching
- Maps non-standard columns to standard features
- Generates synthetic popularity if not present
- Works with partial feature sets

### Recommended Dataset (Spotify)
- **Source**: Kaggle Spotify High Popularity Dataset
- **Size**: 30,000+ tracks
- **Features**: danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo
- **Target**: Popularity score (0-100)

### Visual Dataset (FER-2013)
- **Source**: FER-2013 / DeepFace pre-trained weights
- **Classes**: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

## 🔬 Methodology

### Phase 1: Music Clustering
- **Algorithms**: K-Means, Hierarchical Clustering, DBSCAN
- **Feature Engineering**: PCA for dimensionality reduction
- **Outcome**: 4-6 distinct music mood clusters

### Phase 2: Emotion Detection (CNN)
- **Input**: Real-time webcam feed
- **Model**: DeepFace or custom CNN on FER-2013
- **Process**: Face Detection → Emotion Classification → Mood Mapping

### Phase 3: Popularity Prediction
- **Algorithms**: Linear Regression, Random Forest, ANN
- **Goal**: Predict song popularity from audio features

### Phase 4: Integration
- **Interface**: Streamlit web dashboard
- **Workflow**: Detect emotion → Query cluster → Return playlist

## 🎯 Usage Examples

### Using the Modules

```python
from src.data_processor import DataProcessor
from src.clustering import MusicClusterer
from src.emotion_detector import EmotionDetector
from src.emotion_music_mapper import EmotionMusicMapper

# Load ANY CSV dataset - columns auto-detected!
processor = DataProcessor('data/raw/your_music_data.csv')
processor.load_data()
processor.clean_data()
X, features = processor.get_features(feature_set='clustering')

# See what columns were detected
print(f"Detected features: {processor.detected_features}")
print(f"Column mapping: {processor.column_mapping}")

# Cluster songs
clusterer = MusicClusterer(n_clusters=5)
labels = clusterer.fit_kmeans(X)
profiles = clusterer.get_cluster_profiles(X, labels)

# Detect emotion
detector = EmotionDetector(use_deepface=True)
result = detector.capture_single_emotion()
emotion = result['dominant_emotion']

# Get recommendations
mapper = EmotionMusicMapper()
mapper.set_cluster_profiles(profiles)
recommendations = mapper.recommend_songs(emotion, df, labels, n_songs=10)
```

### Using APIs (Optional)

```python
from src.api_integrations import SpotifyAPI, LastFMAPI

# Spotify API - enrich tracks with audio features
spotify = SpotifyAPI()
if spotify.authenticate():
    features = spotify.get_audio_features(track_ids)
    
# Last.fm API - get similar tracks
lastfm = LastFMAPI()
similar = lastfm.get_similar_tracks("Artist Name", "Track Name")
```

## 📈 Results

### Clustering Performance
| Algorithm | Silhouette Score | Clusters |
|-----------|-----------------|----------|
| K-Means   | ~0.25-0.35      | 5        |
| Hierarchical | ~0.20-0.30   | 5        |
| DBSCAN    | Variable        | Auto     |

### Prediction Performance
| Model | R² Score | RMSE |
|-------|----------|------|
| Random Forest | ~0.15-0.25 | ~20-25 |
| Gradient Boosting | ~0.15-0.25 | ~20-25 |
| ANN | ~0.10-0.20 | ~22-28 |

## 🛠️ Technologies

- **Language**: Python 3.10+
- **Computer Vision**: OpenCV, DeepFace
- **ML/DL**: scikit-learn, TensorFlow/Keras
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Web Interface**: Streamlit

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Spotify for audio feature data
- FER-2013 dataset creators
- DeepFace library maintainers
- Streamlit team for the amazing framework
