"""
Configuration file for API keys and settings.

SETUP INSTRUCTIONS:
1. Copy this file to 'config_local.py' (which is gitignored)
2. Replace placeholder values with your actual API keys

FREE API OPTIONS:
- DeepFace: No API key needed (uses local models)
- Spotify API: Free tier available at https://developer.spotify.com/
- Last.fm API: Free at https://www.last.fm/api/account/create

"""

# =============================================================================
# SPOTIFY API (Optional - for fetching additional track data)
# Get your keys at: https://developer.spotify.com/dashboard/
# =============================================================================
SPOTIFY_CLIENT_ID = "your_spotify_client_id_here"
SPOTIFY_CLIENT_SECRET = "your_spotify_client_secret_here"

# =============================================================================
# LAST.FM API (Optional - for additional metadata)
# Get your key at: https://www.last.fm/api/account/create
# =============================================================================
LASTFM_API_KEY = "your_lastfm_api_key_here"

# =============================================================================
# EMOTION DETECTION SETTINGS
# DeepFace models (no API key needed - runs locally)
# =============================================================================
EMOTION_MODEL = "deepface"  # Options: "deepface", "custom_cnn"
DEEPFACE_BACKEND = "opencv"  # Options: "opencv", "ssd", "dlib", "mtcnn", "retinaface"

# =============================================================================
# APPLICATION SETTINGS
# =============================================================================
DEFAULT_CLUSTERS = 5
MAX_RECOMMENDATIONS = 20
ENABLE_WEBCAM = True
DEBUG_MODE = False

# =============================================================================
# FILE PATHS
# =============================================================================
DATA_RAW_PATH = "data/raw/"
DATA_PROCESSED_PATH = "data/processed/"
MODELS_PATH = "models/"


def load_config():
    """Load configuration, preferring local config if available."""
    config = {
        'spotify_client_id': SPOTIFY_CLIENT_ID,
        'spotify_client_secret': SPOTIFY_CLIENT_SECRET,
        'lastfm_api_key': LASTFM_API_KEY,
        'emotion_model': EMOTION_MODEL,
        'deepface_backend': DEEPFACE_BACKEND,
        'default_clusters': DEFAULT_CLUSTERS,
        'max_recommendations': MAX_RECOMMENDATIONS,
        'enable_webcam': ENABLE_WEBCAM,
        'debug_mode': DEBUG_MODE,
    }
    
    # Try to load local config (with real API keys)
    try:
        import config_local
        config.update({
            'spotify_client_id': getattr(config_local, 'SPOTIFY_CLIENT_ID', SPOTIFY_CLIENT_ID),
            'spotify_client_secret': getattr(config_local, 'SPOTIFY_CLIENT_SECRET', SPOTIFY_CLIENT_SECRET),
            'lastfm_api_key': getattr(config_local, 'LASTFM_API_KEY', LASTFM_API_KEY),
        })
        print("✓ Loaded local configuration")
    except ImportError:
        print("⚠ Using default configuration (no API keys)")
    
    return config
