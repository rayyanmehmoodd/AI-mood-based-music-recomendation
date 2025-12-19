"""
Streamlit Web Application for Emotion-Based Music Recommendation
Interactive dashboard with real-time webcam emotion detection.
Supports any CSV music dataset.
"""

import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import time
import os
import sys
import io

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processor import DataProcessor
from src.clustering import MusicClusterer
from src.emotion_detector import EmotionDetector
from src.popularity_predictor import PopularityPredictor
from src.emotion_music_mapper import EmotionMusicMapper

# Page configuration
st.set_page_config(
    page_title="🎵 Mood Music",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS - Professional Light Blue Theme
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(180deg, #f0f8ff 0%, #e6f3ff 100%);
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Typography */
    .main-title {
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #4A90E2 0%, #5BA3F5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin: 2rem 0;
        letter-spacing: -2px;
    }
    
    .section-title {
        font-size: 2rem;
        font-weight: 700;
        color: #2C5F8D;
        margin: 2rem 0 1rem 0;
    }
    
    .subtitle {
        font-size: 1rem;
        color: #5A7A99;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Cards */
    .mood-card {
        background: linear-gradient(135deg, #4A90E2 0%, #5BA3F5 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(74, 144, 226, 0.3);
        transition: transform 0.2s;
    }
    
    .mood-card:hover {
        transform: translateY(-5px);
    }
    
    .song-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.8rem 0;
        border-left: 4px solid #4A90E2;
        transition: all 0.2s;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }
    
    .song-card:hover {
        background: #f8fcff;
        transform: translateX(5px);
        box-shadow: 0 4px 12px rgba(74, 144, 226, 0.15);
    }
    
    .cluster-pill {
        display: inline-block;
        background: #4A90E2;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.5rem;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    /* Camera and Radio Inputs */
    [data-testid="stCameraInput"] {
        border-radius: 16px;
        overflow: hidden;
        border: 2px solid #d0e7f9;
    }
    
    .stRadio > div {
        background: #ffffff;
        border-radius: 12px;
        padding: 0.75rem;
        border: 1px solid #d0e7f9;
    }
    
    /* Selectbox & Slider */
    .stSelectbox > div > div {
        background: #ffffff;
        border-radius: 8px;
        border: 1px solid #d0e7f9;
    }
    
    .stSelectbox [data-baseweb="select"] {
        color: #2C3E50 !important;
    }
    
    .stSelectbox [data-baseweb="select"] > div {
        color: #2C3E50 !important;
    }
    
    .stSlider > div > div {
        background: #ffffff;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #4A90E2 0%, #5BA3F5 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 0.8rem 2rem;
        font-weight: 700;
        font-size: 1rem;
        transition: all 0.3s;
        box-shadow: 0 4px 15px rgba(74, 144, 226, 0.4);
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(74, 144, 226, 0.6);
    }
    
    /* Upload section */
    [data-testid="stFileUploader"] {
        background: #ffffff;
        border: 2px dashed #4A90E2 !important;
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    [data-testid="stFileUploader"] section {
        background: #f8fcff !important;
        border: 2px solid #d0e7f9 !important;
        border-radius: 12px !important;
        margin-top: 1rem;
    }
    
    [data-testid="stFileUploader"] section [data-testid="stMarkdownContainer"] p {
        color: #2C3E50 !important;
    }
    
    [data-testid="stFileUploader"] button {
        background: #4A90E2 !important;
        color: white !important;
        border: none !important;
        padding: 0.5rem 1.5rem !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stFileUploader"] button:hover {
        background: #5BA3F5 !important;
    }
    
    /* Metrics */
    .metric-box {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 0.5rem 0;
        border: 1px solid #d0e7f9;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 900;
        color: #4A90E2;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #5A7A99;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #5A7A99;
        border: none;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 1rem 2rem;
    }
    
    .stTabs [aria-selected="true"] {
        color: #4A90E2;
        border-bottom: 3px solid #4A90E2;
    }
    
    /* Camera input */
    [data-testid="stCameraInput"] {
        border-radius: 20px;
        overflow: hidden;
    }
    
    /* Dataframes */
    .stDataFrame {
        background: #ffffff;
        border-radius: 15px;
        border: 1px solid #d0e7f9;
    }
    
    /* Sliders */
    .stSlider {
        padding: 1rem 0;
    }
    
    .stSlider > div > div > div {
        background: #4A90E2 !important;
    }
    
    .stSlider > div > div > div > div {
        background: #4A90E2 !important;
    }
    
    .stSlider [data-baseweb="slider"] {
        background: linear-gradient(90deg, #d0e7f9 0%, #4A90E2 100%) !important;
    }
    
    /* Text color overrides */
    .stMarkdown, p, span, label {
        color: #2C3E50 !important;
    }
    
    .stCaption {
        color: #5A7A99 !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load pre-trained models."""
    models = {}
    
    # Initialize components
    models['data_processor'] = DataProcessor()
    models['clusterer'] = MusicClusterer()
    models['emotion_detector'] = EmotionDetector(use_deepface=True)
    models['predictor'] = PopularityPredictor()
    models['mapper'] = EmotionMusicMapper()
    
    return models


def process_uploaded_csv(uploaded_file) -> pd.DataFrame:
    """Process an uploaded CSV file."""
    try:
        # Read CSV
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return None


def process_music_data(df: pd.DataFrame, models: dict) -> pd.DataFrame:
    """Process music data through the pipeline."""
    processor = models['data_processor']
    clusterer = models['clusterer']
    
    # Set raw data and process
    processor.raw_data = df
    processor._detect_columns()
    processor.clean_data()
    
    # Get features for clustering
    X, features = processor.get_features(feature_set='clustering', normalize=True)
    
    if len(features) > 0:
        # Cluster the data
        from sklearn.preprocessing import StandardScaler
        labels = clusterer.fit_kmeans(X.values, n_clusters=min(5, len(df)//10 + 2))
        processor.processed_data['cluster'] = labels
        
        # Get cluster profiles
        profiles = clusterer.get_cluster_profiles(X, labels)
        models['mapper'].set_cluster_profiles(profiles)
        
        # Store clustering features for visualization
        st.session_state['clustering_features'] = features
        
        st.success(f"✓ Processed {len(df)} tracks into {len(set(labels))} clusters")
    else:
        st.warning("⚠ Could not detect audio features - using random clusters")
        processor.processed_data['cluster'] = np.random.randint(0, 5, len(df))
    
    # Ensure critical columns are preserved
    critical_cols = ['artist', 'track_artist', 'track name', 'track_name', 'name', 'song']
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in ['artist', 'track artist', 'track_artist', 'artist name', 'artist_name']:
            processor.processed_data['artist'] = df[col]
        if col_lower in ['track name', 'track_name', 'name', 'song', 'title']:
            processor.processed_data['track_name'] = df[col]
    
    return processor.processed_data


# Sample data functionality removed - app now requires user to upload CSV


def detect_emotion_from_webcam():
    """Capture and analyze emotion from webcam."""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        return None, "Could not access webcam. Try using 'Select Your Mood' option instead."
        
    # Capture frame
    for _ in range(10):  # Skip initial frames
        cap.read()
        
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None, "Could not capture frame. Try using 'Select Your Mood' option instead."
        
    return frame, None


def main():
    """Main application with Spotify-like interface."""
    
    # Header
    st.markdown('<h1 class="main-title">🎵 Mood Music</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-Powered Music Recommendations Based on Your Emotions</p>', unsafe_allow_html=True)
    
    # Load models
    with st.spinner("Loading AI models..."):
        models = load_models()
    
    # File upload - prominent and centered
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        upload_container = st.container()
        with upload_container:
            st.markdown("### 📁 Upload Your Music Library")
            uploaded_file = st.file_uploader(
                "Drop your CSV file here",
                type=['csv'],
                help="CSV with audio features: danceability, energy, valence, tempo, etc.",
                label_visibility="visible"
            )
    
    if uploaded_file is None:
        # Welcome screen
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-box">
                <div class="metric-value">📷</div>
                <div class="metric-label">Emotion Detection</div>
                <p style="color: #b3b3b3; margin-top: 1rem;">Analyze your mood through facial recognition</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-box">
                <div class="metric-value">🎵</div>
                <div class="metric-label">Smart Recommendations</div>
                <p style="color: #b3b3b3; margin-top: 1rem;">Get personalized playlists based on your emotions</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-box">
                <div class="metric-value">📊</div>
                <div class="metric-label">Popularity Prediction</div>
                <p style="color: #b3b3b3; margin-top: 1rem;">Predict song popularity using AI</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.info("👆 **Upload your music dataset to get started**")
        return
    
    # Process uploaded data (only if not already in session state or file changed)
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    
    if 'music_data' not in st.session_state or st.session_state.get('file_id') != file_id:
        with st.spinner("🎵 Processing your music library..."):
            raw_df = process_uploaded_csv(uploaded_file)
            if raw_df is None:
                st.error("❌ Could not process the file. Please check your CSV format.")
                return
            
            music_data = process_music_data(raw_df, models)
            if music_data is None or len(music_data) == 0:
                st.error("❌ No valid data found in the file.")
                return
            
            # Store in session state
            st.session_state['music_data'] = music_data
            st.session_state['file_id'] = file_id
    else:
        # Use cached data from session state
        music_data = st.session_state['music_data']
    
    # Success message with stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{len(music_data)}</div>
            <div class="metric-label">Total Songs</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        n_clusters = len(music_data['cluster'].unique()) if 'cluster' in music_data.columns else 0
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{n_clusters}</div>
            <div class="metric-label">Clusters</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        processor = models['data_processor']
        n_features = len(processor.detected_features)
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{n_features}</div>
            <div class="metric-label">Audio Features</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        has_popularity = any('popularity' in col.lower() for col in music_data.columns)
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{"✓" if has_popularity else "✗"}</div>
            <div class="metric-label">Popularity Data</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Main navigation - 3 sections
    tab1, tab2, tab3 = st.tabs(["🎵 Discover Music", "🔍 Clusters & Moods", "📊 Popularity Predictor"])
    
    with tab1:
        show_music_discovery(models, music_data)
    
    with tab2:
        show_clusters_and_moods(models, music_data)
    
    with tab3:
        show_analytics(models, music_data)


# Section Functions
def show_music_discovery(models, music_data):
    """Music discovery section - emotion detection and recommendations."""
    
    st.markdown('<h2 class="section-title">Discover Your Perfect Playlist</h2>', unsafe_allow_html=True)
    st.caption("Let AI detect your mood and recommend songs that match your vibe")
    
    # Two columns: Detection | Recommendations
    col_detect, col_recommend = st.columns([1, 1.5])
    
    with col_detect:
        # Mood selection method
        detection_method = st.radio(
            "How do you want to select your mood?",
            ["📷 Camera Detection", "😊 Manual Selection"],
            horizontal=True,
            label_visibility="collapsed"
        )
        
        if detection_method == "📷 Camera Detection":
            st.markdown("### 📷 Capture Your Mood")
            camera_photo = st.camera_input("Take a photo", key="camera_input", label_visibility="collapsed")
            
            if camera_photo is not None:
                image = Image.open(camera_photo)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                frame = np.array(image)
                frame = frame[:, :, ::-1]
                
                with st.spinner("🔍 Analyzing..."):
                    try:
                        result = models['emotion_detector'].detect_emotion(frame)
                        
                        if 'error' in result or not result.get('face_detected'):
                            st.error("❌ No face detected")
                        else:
                            emotion = result.get('dominant_emotion', 'Neutral')
                            confidence = result.get('confidence', 0.5)
                            scores = result.get('emotion_scores', {})
                            
                            st.session_state['current_emotion'] = emotion
                            st.session_state['emotion_confidence'] = confidence
                            st.session_state['emotion_scores'] = scores
                            
                            st.success(f"✅ Detected: **{emotion}** ({confidence:.0%})")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.info("📸 Take a photo to detect your mood")
        
        else:  # Manual Selection
            st.markdown("### 😊 Select Your Mood")
            emotions = ['Happy', 'Sad', 'Angry', 'Fear', 'Surprise', 'Neutral']
            emotion_emojis = {'Happy': '😊', 'Sad': '😢', 'Angry': '😠', 'Fear': '😨', 'Surprise': '😲', 'Neutral': '😐'}
            
            # Create 2x3 grid of emotion buttons
            for i in range(0, len(emotions), 3):
                cols = st.columns(3)
                for j, col in enumerate(cols):
                    if i + j < len(emotions):
                        emotion = emotions[i + j]
                        emoji = emotion_emojis[emotion]
                        if col.button(f"{emoji} {emotion}", width="stretch", key=f"emotion_{emotion}"):
                            st.session_state['current_emotion'] = emotion
                            st.session_state['emotion_confidence'] = 1.0
                            st.rerun()
        
        # Display selected emotion
        if 'current_emotion' in st.session_state:
            emotion = st.session_state['current_emotion']
            confidence = st.session_state.get('emotion_confidence', 1.0)
            
            st.markdown("---")
            st.markdown(f"""
            <div class="mood-card">
                <h2>{get_emotion_emoji(emotion)} {emotion}</h2>
                <p>Confidence: {confidence:.0%}</p>
                <p style="font-size: 0.9rem; opacity: 0.9;">{models['mapper'].get_mood_description(emotion)}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col_recommend:
        if 'current_emotion' in st.session_state:
            display_recommendations(models, music_data, st.session_state['current_emotion'])
        else:
            st.markdown("### 🎵 Your Playlist")
            st.info("👈 Select or detect your mood to see recommendations")


def show_clusters_and_moods(models, music_data):
    """Clusters and emotion mapping section."""
    
    st.markdown('<h2 class="section-title">Music Clusters & Mood Mapping</h2>', unsafe_allow_html=True)
    st.caption("Explore how songs are grouped and which clusters match different emotions")
    
    # Clustering controls
    st.markdown("### 🎯 Create Music Clusters")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown('<p style="color: #2C3E50; font-weight: 600; margin-bottom: 0.5rem;">Clustering Algorithm</p>', unsafe_allow_html=True)
        algorithm = st.selectbox(
            "Clustering Algorithm",
            ["K-Means", "Hierarchical"],
            label_visibility="collapsed"
        )
    with col2:
        st.markdown('<p style="color: #2C3E50; font-weight: 600; margin-bottom: 0.5rem;">Number of Clusters</p>', unsafe_allow_html=True)
        n_clusters = st.slider("Clusters", 2, 10, 5, label_visibility="collapsed")
    with col3:
        st.write("")  # Empty space for layout balance
        linkage = 'ward' if algorithm == "Hierarchical" else None
    
    if st.button(f"▶️ Run {algorithm} Clustering", type="primary", width="stretch"):
        show_clustering_results(models, music_data, algorithm, n_clusters, linkage)
    
    st.caption("💡 Adjust parameters and click the button to re-cluster your songs")
    
    # Show results if clustering exists
    if 'cluster' in music_data.columns:
        st.markdown("---")
        
        # Show all cluster details
        st.markdown("### 📦 Cluster Details")
        st.caption("Overview of all music clusters created by the algorithm")
        
        n_clusters = len(music_data['cluster'].unique())
        cluster_stats = []
        
        for cluster_id in sorted(music_data['cluster'].unique()):
            cluster_songs = music_data[music_data['cluster'] == cluster_id]
            n_songs = len(cluster_songs)
            
            # Get cluster characteristics
            characteristics = []
            if 'energy' in cluster_songs.columns:
                avg_energy = cluster_songs['energy'].mean()
                if avg_energy > 0.7:
                    characteristics.append("High Energy")
                elif avg_energy < 0.4:
                    characteristics.append("Low Energy")
            
            if 'valence' in cluster_songs.columns:
                avg_valence = cluster_songs['valence'].mean()
                if avg_valence > 0.6:
                    characteristics.append("Positive")
                elif avg_valence < 0.4:
                    characteristics.append("Melancholic")
            
            if 'danceability' in cluster_songs.columns:
                avg_dance = cluster_songs['danceability'].mean()
                if avg_dance > 0.7:
                    characteristics.append("Danceable")
            
            if 'acousticness' in cluster_songs.columns:
                avg_acoustic = cluster_songs['acousticness'].mean()
                if avg_acoustic > 0.6:
                    characteristics.append("Acoustic")
            
            if 'tempo' in cluster_songs.columns:
                avg_tempo = cluster_songs['tempo'].mean()
                if avg_tempo > 140:
                    characteristics.append("Fast")
                elif avg_tempo < 100:
                    characteristics.append("Slow")
            
            char_text = ", ".join(characteristics) if characteristics else "Balanced"
            
            st.markdown(f"""
            <div class="song-card" style="margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h3 style="margin: 0; color: #4A90E2;">Cluster {cluster_id}</h3>
                        <p style="margin: 0.5rem 0 0 0; color: #2C3E50; font-weight: 500;">{char_text}</p>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 2rem; color: #4A90E2; font-weight: bold;">{n_songs}</div>
                        <div style="color: #5A7A99; font-size: 0.9rem;">songs</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Emotion to Cluster Mapping
        st.markdown("### 🎭 Emotion → Cluster Mapping")
        
        emotion_mapper = models['mapper']
        if emotion_mapper.cluster_profiles is not None:
            emotions = ['Happy', 'Sad', 'Angry', 'Fear', 'Surprise', 'Neutral']
            
            # Display as pills/badges
            for emotion in emotions:
                target_clusters = emotion_mapper.get_clusters_for_emotion(emotion)
                if target_clusters:
                    emoji = get_emotion_emoji(emotion)
                    clusters_html = ''.join([f'<span class="cluster-pill">Cluster {c}</span>' for c in target_clusters[:3]])
                    
                    if emotion == 'Happy':
                        desc = "High energy, positive, upbeat songs"
                    elif emotion == 'Sad':
                        desc = "Low energy, melancholic, slower songs"
                    elif emotion == 'Angry':
                        desc = "High energy, intense, fast-paced songs"
                    elif emotion == 'Fear':
                        desc = "Medium energy, calming, soothing songs"
                    elif emotion == 'Surprise':
                        desc = "High energy, dynamic, exciting songs"
                    else:
                        desc = "Balanced, diverse selection of songs"
                    
                    st.markdown(f"""
                    <div class="song-card">
                        <h3>{emoji} {emotion}</h3>
                        <p style="color: #b3b3b3;">{desc}</p>
                        <div style="margin-top: 1rem;">{clusters_html}</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Visualization
        st.markdown("---")
        st.markdown("### 📊 Cluster Visualization")
        show_cluster_visualization(models, music_data)


def show_clustering_results(models, music_data, algorithm, n_clusters, linkage):
    """Run clustering and update data."""
    numeric_cols = music_data.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns.tolist()
    exclude = ['cluster', 'id', 'track_id', 'popularity', 'predicted_popularity', 'index']
    available = [f for f in numeric_cols if f.lower() not in [e.lower() for e in exclude]]
    
    priority_features = ['danceability', 'energy', 'valence', 'loudness', 'tempo', 'acousticness']
    clustering_features = [f for f in priority_features if f in music_data.columns]
    
    if not clustering_features:
        clustering_features = available[:6] if available else []
    
    if not clustering_features:
        st.error("❌ No numeric features available for clustering")
        return
    
    with st.spinner(f"Running {algorithm} clustering with {n_clusters} clusters..."):
        try:
            X = music_data[clustering_features].fillna(0)
            X_scaled = models['clusterer'].scaler.fit_transform(X)
            
            if algorithm == "K-Means":
                labels = models['clusterer'].fit_kmeans(X_scaled, n_clusters=n_clusters)
            else:
                labels = models['clusterer'].fit_hierarchical(X_scaled, n_clusters=n_clusters, linkage=linkage)
            
            music_data['cluster'] = labels
            
            profiles = music_data[music_data['cluster'] != -1].groupby('cluster')[clustering_features].mean()
            models['mapper'].set_cluster_profiles(profiles)
            
            # Update session state with new clustering
            st.session_state['music_data'] = music_data
            st.session_state['clustering_features'] = clustering_features
            
            st.success(f"✅ {algorithm} clustering completed with {n_clusters} clusters!")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Clustering failed: {e}")


def show_cluster_visualization(models, music_data):
    """Display cluster visualization."""
    try:
        from sklearn.decomposition import PCA
        import plotly.express as px
        
        # Use the same features that were used for clustering
        if 'clustering_features' in st.session_state:
            clustering_features = st.session_state['clustering_features']
        else:
            # Fallback to default features
            numeric_cols = music_data.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns.tolist()
            exclude = ['cluster', 'id', 'track_id', 'popularity', 'predicted_popularity', 'index']
            available = [f for f in numeric_cols if f.lower() not in [e.lower() for e in exclude]]
            priority_features = ['danceability', 'energy', 'valence', 'loudness', 'tempo', 'acousticness']
            clustering_features = [f for f in priority_features if f in music_data.columns]
        
        X = music_data[clustering_features].fillna(0)
        
        if hasattr(models['clusterer'], 'scaler') and hasattr(models['clusterer'].scaler, 'mean_'):
            X_scaled = models['clusterer'].scaler.transform(X)
        else:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        
        pca = PCA(n_components=2, random_state=42)
        X_2d = pca.fit_transform(X_scaled)
        
        plot_df = pd.DataFrame({
            'Component 1': X_2d[:, 0],
            'Component 2': X_2d[:, 1],
            'Cluster': music_data['cluster'].astype(str)
        })
        
        if 'track_name' in music_data.columns:
            plot_df['Song'] = music_data['track_name'].values
        
        fig = px.scatter(
            plot_df,
            x='Component 1',
            y='Component 2',
            color='Cluster',
            hover_data=['Song'] if 'Song' in plot_df.columns else None,
            title='Songs Grouped by Similarity',
            height=500
        )
        
        fig.update_layout(
            plot_bgcolor='#282828',
            paper_bgcolor='#282828',
            font=dict(color='#ffffff')
        )
        
        st.plotly_chart(fig, width='stretch')
        st.caption("Songs closer together are more similar in style and mood")
        
    except Exception as e:
        st.warning(f"Could not create visualization: {e}")


def display_recommendations(models, music_data, emotion):
    """Display song recommendations for given emotion."""
    st.markdown("### 🎵 Your Personalized Playlist")
    
    with st.spinner("Finding perfect songs for your mood..."):
        try:
            # Get cluster labels if available
            cluster_labels = music_data['cluster'].values if 'cluster' in music_data.columns else np.zeros(len(music_data))
            
            recommendations = models['mapper'].recommend_songs(
                emotion=emotion,
                music_data=music_data,
                cluster_labels=cluster_labels,
                n_songs=10,
                prioritize_popular=True
            )
            
            if recommendations.empty:
                st.warning("No recommendations found for this mood. Try different data or emotions.")
                return
            
            st.success(f"✅ Found {len(recommendations)} songs matching your {emotion} mood!")
            
            # Display as cards
            for idx, song in recommendations.iterrows():
                # Try to get track name
                track = 'Unknown Track'
                for col in ['track_name', 'name', 'song', 'title']:
                    if col in song.index and pd.notna(song[col]):
                        track = song[col]
                        break
                
                # Try to get artist name
                artist = 'Unknown Artist'
                for col in ['artist', 'artists', 'track_artist', 'artist_name']:
                    if col in song.index and pd.notna(song[col]):
                        artist = song[col]
                        break
                
                popularity = song.get('popularity', song.get('track_popularity', song.get('predicted_popularity', 0)))
                score = song.get('emotion_match_score', 0)
                
                # Get key audio features
                energy = song.get('energy', 0)
                valence = song.get('valence', 0)
                tempo = song.get('tempo', 0)
                
                st.markdown(f"""
                <div class="song-card">
                    <h3 style="margin: 0; color: #ffffff;">🎵 {track}</h3>
                    <p style="margin: 0.5rem 0; color: #1DB954; font-weight: 600;">{artist}</p>
                    <p style="margin: 0.5rem 0; color: #b3b3b3; font-size: 0.9rem;">
                        Match Score: {score*100:.1f}% • Popularity: {popularity:.0f}
                    </p>
                    <p style="margin: 0.5rem 0 0 0; color: #b3b3b3; font-size: 0.85rem;">
                        Energy: {energy:.2f} • Valence: {valence:.2f} • Tempo: {tempo:.0f} BPM
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"❌ Error generating recommendations: {e}")


def train_popularity_model(models, music_data, model_type):
    """Train popularity prediction model."""
    with st.spinner(f"Training {model_type}..."):
        try:
            # Get features
            numeric_cols = music_data.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns.tolist()
            exclude = ['cluster', 'id', 'track_id', 'popularity', 'track_popularity', 'predicted_popularity', 'index']
            available = [f for f in numeric_cols if f.lower() not in [e.lower() for e in exclude]]
            
            priority_features = ['danceability', 'energy', 'valence', 'loudness', 'tempo', 
                               'acousticness', 'speechiness', 'instrumentalness', 'liveness', 'duration_ms']
            features = [f for f in priority_features if f in available]
            
            if not features:
                features = available[:10]
            
            if not features:
                st.error("❌ No numeric features found for prediction")
                return
            
            # Find popularity column - check processor's column mapping
            processor = models['data_processor']
            popularity_col = processor.column_mapping.get('popularity', 'popularity')
            
            # Also check for common variations if not found
            if popularity_col not in music_data.columns:
                for col in music_data.columns:
                    if 'popularity' in col.lower():
                        popularity_col = col
                        break
            
            # Check for target
            if popularity_col not in music_data.columns or music_data[popularity_col].isna().all():
                st.error(f"❌ No popularity data found. Cannot train model.")
                return
            
            # Prepare data
            X = music_data[features].fillna(0)
            y = music_data[popularity_col].fillna(music_data[popularity_col].mean())
            
            # Train based on model type
            if "Gradient Boosting" in model_type:
                from sklearn.ensemble import GradientBoostingRegressor
                model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            elif "Random Forest" in model_type:
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:  # Linear Regression
                from sklearn.linear_model import Ridge
                model = Ridge(alpha=1.0)
            
            model.fit(X, y)
            models['predictor'].best_model = model
            models['predictor'].model_name = model_type
            models['predictor'].feature_names = features
            
            # Make predictions immediately
            predictions = model.predict(X)
            music_data['predicted_popularity'] = predictions
            
            st.success(f"✅ {model_type} trained successfully!")
            # Don't rerun - just let the page update naturally
            
        except Exception as e:
            st.error(f"❌ Training failed: {e}")


def show_analytics(models, music_data):
    """Popularity prediction section."""
    
    st.markdown('<h2 class="section-title">Popularity Predictor</h2>', unsafe_allow_html=True)
    st.caption("Train AI models to predict which songs will become popular")
    
    # Model selection
    st.markdown("### 🤖 Select Model")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<p style="color: #2C3E50; font-weight: 600; margin-bottom: 0.5rem;">Choose Prediction Model</p>', unsafe_allow_html=True)
        model_type = st.selectbox(
            "Choose prediction model",
            ["Gradient Boosting", "Random Forest", "Linear Regression"],
            label_visibility="collapsed"
        )
    
    with col2:
        if st.button("▶️ Train Model", type="primary", width="stretch"):
            train_popularity_model(models, music_data, model_type)
    
    # Apply predictions if model exists but predictions column doesn't
    if hasattr(models['predictor'], 'best_model') and models['predictor'].best_model is not None:
        if 'predicted_popularity' not in music_data.columns and hasattr(models['predictor'], 'feature_names'):
            try:
                features = models['predictor'].feature_names
                X = music_data[features].fillna(0)
                predictions = models['predictor'].best_model.predict(X)
                music_data['predicted_popularity'] = predictions
            except Exception as e:
                st.warning(f"Could not apply predictions: {e}")
    
    # Show predictions if model exists
    if hasattr(models['predictor'], 'best_model') and models['predictor'].best_model is not None:
        st.markdown("---")
        st.markdown("### 📊 Prediction Results")
        
        if 'predicted_popularity' in music_data.columns:
            # Get actual popularity column name
            processor = models['data_processor']
            popularity_col = processor.column_mapping.get('popularity', 'popularity')
            if popularity_col not in music_data.columns:
                for col in music_data.columns:
                    if 'popularity' in col.lower():
                        popularity_col = col
                        break
            
            # Model performance
            if popularity_col in music_data.columns and music_data[popularity_col].notna().sum() > 0:
                st.markdown("---")
                st.markdown("#### 📈 Model Performance")
                
                from sklearn.metrics import mean_absolute_error, r2_score
                actual = music_data[popularity_col].dropna()
                predicted = music_data.loc[actual.index, 'predicted_popularity']
                
                mae = mean_absolute_error(actual, predicted)
                r2 = r2_score(actual, predicted)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f'<p style="color: #5A7A99; font-size: 0.9rem; margin: 0;">Mean Error</p><p style="color: #2C3E50; font-size: 2rem; font-weight: bold; margin: 0;">{mae:.1f}</p><p style="color: #5A7A99; font-size: 0.85rem;">points</p>', unsafe_allow_html=True)
                with col2:
                    st.markdown(f'<p style="color: #5A7A99; font-size: 0.9rem; margin: 0;">R² Score</p><p style="color: #2C3E50; font-size: 2rem; font-weight: bold; margin: 0;">{r2:.3f}</p>', unsafe_allow_html=True)
            
            # Feature Importance
            show_importance = False
            feature_names = models['predictor'].feature_names
            importances = None
            
            if hasattr(models['predictor'].best_model, 'feature_importances_'):
                # For tree-based models (Random Forest, Gradient Boosting)
                importances = models['predictor'].best_model.feature_importances_
                show_importance = True
            elif hasattr(models['predictor'].best_model, 'coef_'):
                # For linear models (Linear Regression)
                importances = np.abs(models['predictor'].best_model.coef_)
                show_importance = True
            
            if show_importance and importances is not None:
                st.markdown("---")
                st.markdown("#### 🔍 Feature Importance")
                st.caption("Top 10 Most Important Features for Popularity Prediction")
                
                import plotly.graph_objects as go
                
                # Sort by importance
                indices = np.argsort(importances)[::-1][:10]  # Top 10
                top_features = [feature_names[i] for i in indices]
                top_importances = [importances[i] for i in indices]
                
                # Create horizontal bar chart
                fig = go.Figure(go.Bar(
                    x=top_importances,
                    y=top_features,
                    orientation='h',
                    marker=dict(
                        color=top_importances,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="importance")
                    ),
                    text=[f"{imp:.3f}" for imp in top_importances],
                    textposition='outside'
                ))
                
                fig.update_layout(
                    title="Top 10 Most Important Features for Popularity Prediction",
                    xaxis_title="importance",
                    yaxis_title="feature",
                    height=500,
                    plot_bgcolor='#1a1a1a',
                    paper_bgcolor='#1a1a1a',
                    font=dict(color='#ffffff', size=12),
                    yaxis=dict(autorange="reversed")
                )
                
                st.plotly_chart(fig, width='stretch')
                st.caption("💡 Higher values mean the feature has more influence on popularity predictions")
        else:
            st.info("Train the model to see predictions")
        
        # New song prediction section
        st.markdown("---")
        st.markdown("### 🎤 Predict New Song Popularity")
        st.caption("Upload audio features of a new song to predict its potential popularity")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            new_song_file = st.file_uploader(
                "Upload new song CSV (with audio features)",
                type=['csv'],
                help="CSV should contain: danceability, energy, loudness, valence, tempo, etc.",
                key="new_song_upload"
            )
        
        with col2:
            # Download template button
            template_path = "data/processed/new_song_template.csv"
            if os.path.exists(template_path):
                with open(template_path, 'r') as f:
                    template_csv = f.read()
                st.download_button(
                    label="📄 Download Template",
                    data=template_csv,
                    file_name="new_song_template.csv",
                    mime="text/csv",
                    help="Download a template CSV to fill in your song's audio features"
                )
        
        if new_song_file is not None:
            try:
                new_song_df = pd.read_csv(new_song_file)
                
                # Get required features
                features = models['predictor'].feature_names
                missing_features = [f for f in features if f not in new_song_df.columns]
                
                if missing_features:
                    st.error(f"❌ Missing required features: {', '.join(missing_features)}")
                    st.info(f"Required features: {', '.join(features)}")
                else:
                    # Make prediction
                    X_new = new_song_df[features].fillna(0)
                    predictions = models['predictor'].best_model.predict(X_new)
                    new_song_df['predicted_popularity'] = predictions
                    
                    st.success(f"✅ Predicted popularity for {len(new_song_df)} song(s)!")
                    
                    # Display predictions
                    for idx, song in new_song_df.iterrows():
                        track = song.get('track_name', song.get('name', f'Song {idx+1}'))
                        artist = song.get('artists', song.get('artist', 'Unknown Artist'))
                        predicted = song['predicted_popularity']
                        
                        # Get key audio features for display
                        energy = song.get('energy', 0)
                        valence = song.get('valence', 0)
                        danceability = song.get('danceability', 0)
                        
                        st.markdown(f"""
                        <div class="song-card">
                            <h3 style="margin: 0; color: #ffffff;">🎵 {track}</h3>
                            <p style="margin: 0.5rem 0; color: #1DB954; font-weight: 600;">{artist}</p>
                            <p style="margin: 0.5rem 0; color: #ffffff; font-size: 1.2rem;">
                                Predicted Popularity: <strong>{predicted:.0f}</strong>
                            </p>
                            <p style="margin: 0.5rem 0 0 0; color: #b3b3b3; font-size: 0.85rem;">
                                Energy: {energy:.2f} • Valence: {valence:.2f} • Danceability: {danceability:.2f}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Download predictions
                    csv = new_song_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions",
                        data=csv,
                        file_name="predicted_popularity.csv",
                        mime="text/csv"
                    )
                    
            except Exception as e:
                st.error(f"❌ Error processing file: {e}")
        else:
            st.info("👆 Upload a CSV file with audio features to predict popularity")


def get_emotion_emoji(emotion: str) -> str:
    """Get emoji for emotion."""
    emoji_map = {
        'Happy': '😊',
        'Sad': '😢',
        'Angry': '😠',
        'Fear': '😨',
        'Surprise': '😲',
        'Disgust': '🤢',
        'Neutral': '😐'
    }
    return emoji_map.get(emotion, '🎵')


if __name__ == "__main__":
    main()



