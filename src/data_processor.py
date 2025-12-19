"""
Data Processing Module for Any Music Dataset
Handles loading, cleaning, and preprocessing of audio features.
Automatically detects and adapts to CSV column structures.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import os
from typing import Tuple, Optional, List, Dict


class DataProcessor:
    """
    Flexible data processing class for any music dataset CSV.
    
    Handles:
    - Automatic column detection and mapping
    - Data loading from any CSV structure
    - Feature selection and engineering
    - Normalization and scaling
    - Train/test splitting
    
    Supports datasets from: Spotify, Apple Music, Last.fm, custom CSVs
    """
    
    # Standard audio features (will auto-detect similar column names)
    AUDIO_FEATURES = [
        'danceability', 'energy', 'loudness', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness', 
        'valence', 'tempo', 'duration_ms', 'duration'
    ]
    
    # Features for clustering (subset for better results)
    CLUSTERING_FEATURES = [
        'danceability', 'energy', 'loudness', 'acousticness',
        'valence', 'tempo'
    ]
    
    # Common column name mappings (maps various names to standard names)
    COLUMN_ALIASES = {
        # Track/Song name
        'track_name': ['track_name', 'name', 'song', 'song_name', 'title', 'track', 'song_title', 'track name'],
        # Artist
        'artist': ['track_artist', 'artist', 'artist_name', 'artists', 'performer', 'singer', 'artist name', 'track artist'],
        # Album
        'album': ['track_album_name', 'album', 'album_name', 'release', 'album name'],
        # Popularity
        'popularity': ['popularity', 'pop', 'popularity_score', 'score', 'rating', 'streams', 'track_popularity'],
        # Duration
        'duration_ms': ['duration_ms', 'duration', 'length', 'time', 'track_duration', 'length_ms', 'duration ms'],
        # Audio features
        'danceability': ['danceability', 'dance', 'danceable'],
        'energy': ['energy', 'energetic', 'energy_level'],
        'loudness': ['loudness', 'loud', 'volume', 'db'],
        'speechiness': ['speechiness', 'speech', 'vocal'],
        'acousticness': ['acousticness', 'acoustic'],
        'instrumentalness': ['instrumentalness', 'instrumental'],
        'liveness': ['liveness', 'live'],
        'valence': ['valence', 'positivity', 'mood', 'happiness'],
        'tempo': ['tempo', 'bpm', 'beats_per_minute', 'speed'],
        # Genre
        'genre': ['genre', 'genres', 'playlist_genre', 'category', 'style', 'playlist genre'],
    }
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the DataProcessor.
        
        Args:
            data_path: Path to the CSV data file
        """
        self.data_path = data_path
        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        self.column_mapping: Dict[str, str] = {}  # Maps standard names to actual column names
        self.detected_features: List[str] = []
        self.has_popularity: bool = False
        
    def load_data(self, path: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from CSV file and auto-detect column structure.
        
        Args:
            path: Optional path override
            
        Returns:
            Loaded DataFrame
        """
        load_path = path or self.data_path
        if load_path is None:
            raise ValueError("No data path provided")
        
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        for encoding in encodings:
            try:
                self.raw_data = pd.read_csv(load_path, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError(f"Could not read CSV with any encoding: {encodings}")
        
        print(f"✓ Loaded {len(self.raw_data)} tracks from {load_path}")
        print(f"✓ Found {len(self.raw_data.columns)} columns")
        
        # Auto-detect column mappings
        self._detect_columns()
        
        return self.raw_data
    
    def _detect_columns(self):
        """Auto-detect and map column names to standard names."""
        if self.raw_data is None:
            return
            
        actual_columns = [col.lower().strip() for col in self.raw_data.columns]
        original_columns = list(self.raw_data.columns)
        
        # Create mapping from standard names to actual column names
        for standard_name, aliases in self.COLUMN_ALIASES.items():
            for alias in aliases:
                for i, col in enumerate(actual_columns):
                    if alias.lower() == col or alias.lower() in col:
                        self.column_mapping[standard_name] = original_columns[i]
                        break
                if standard_name in self.column_mapping:
                    break
        
        # Detect available audio features
        self.detected_features = []
        for feature in self.AUDIO_FEATURES:
            if feature in self.column_mapping:
                self.detected_features.append(self.column_mapping[feature])
            elif feature in self.raw_data.columns:
                self.detected_features.append(feature)
                self.column_mapping[feature] = feature
        
        # Check for popularity column
        self.has_popularity = 'popularity' in self.column_mapping
        
        print(f"✓ Detected {len(self.detected_features)} audio features: {self.detected_features}")
        if self.has_popularity:
            print(f"✓ Found popularity column: {self.column_mapping['popularity']}")
        else:
            print("⚠ No popularity column found - will generate synthetic scores")
            
    def get_column(self, standard_name: str) -> Optional[str]:
        """Get actual column name from standard name."""
        return self.column_mapping.get(standard_name)
    
    def get_info(self) -> dict:
        """Get dataset information summary."""
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        return {
            'total_tracks': len(self.raw_data),
            'features': list(self.raw_data.columns),
            'detected_audio_features': self.detected_features,
            'column_mapping': self.column_mapping,
            'has_popularity': self.has_popularity,
            'missing_values': self.raw_data.isnull().sum().to_dict(),
            'dtypes': self.raw_data.dtypes.astype(str).to_dict()
        }
    
    def clean_data(self) -> pd.DataFrame:
        """
        Clean the raw dataset.
        
        Operations:
        - Remove duplicates
        - Handle missing values
        - Normalize column values
        - Generate popularity if missing
        
        Returns:
            Cleaned DataFrame
        """
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        df = self.raw_data.copy()
        initial_count = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values in detected features
        for feature in self.detected_features:
            if feature in df.columns:
                # Fill numeric columns with median
                if df[feature].dtype in ['float64', 'int64']:
                    df[feature] = df[feature].fillna(df[feature].median())
        
        # Normalize features that should be 0-1
        normalized_features = ['danceability', 'energy', 'speechiness', 'acousticness', 
                               'instrumentalness', 'liveness', 'valence']
        for std_name in normalized_features:
            col = self.column_mapping.get(std_name)
            if col and col in df.columns:
                # Check if values are percentages (0-100) and convert to 0-1
                if df[col].max() > 1:
                    df[col] = df[col] / 100.0
        
        # Normalize loudness (typically -60 to 0 dB)
        loudness_col = self.column_mapping.get('loudness')
        if loudness_col and loudness_col in df.columns:
            if df[loudness_col].min() >= 0:  # Might be absolute values
                df[loudness_col] = -df[loudness_col]
        
        # Generate popularity if not present
        if not self.has_popularity:
            df['popularity'] = self._generate_synthetic_popularity(df)
            self.column_mapping['popularity'] = 'popularity'
            self.has_popularity = True
            print("✓ Generated synthetic popularity scores")
        
        # Remove rows with all NaN in audio features
        df = df.dropna(subset=self.detected_features, how='all')
            
        self.processed_data = df
        print(f"✓ Cleaned data: {initial_count} → {len(df)} tracks")
        return self.processed_data
    
    def _generate_synthetic_popularity(self, df: pd.DataFrame) -> np.ndarray:
        """Generate synthetic popularity based on available features."""
        scores = np.zeros(len(df))
        weights_applied = 0
        
        # Weight features that typically correlate with popularity
        feature_weights = {
            'energy': 0.3,
            'danceability': 0.25,
            'valence': 0.2,
            'loudness': 0.15,
            'tempo': 0.1
        }
        
        for std_name, weight in feature_weights.items():
            col = self.column_mapping.get(std_name)
            if col and col in df.columns:
                values = df[col].values.copy()
                # Normalize to 0-1
                if std_name == 'loudness':
                    values = (values + 60) / 60  # Normalize loudness
                elif std_name == 'tempo':
                    values = (values - 60) / 140  # Normalize tempo
                values = np.clip(values, 0, 1)
                scores += values * weight
                weights_applied += weight
        
        if weights_applied > 0:
            scores = scores / weights_applied
        else:
            scores = np.random.uniform(0.3, 0.7, len(df))
        
        # Scale to 0-100 and add some noise
        scores = scores * 70 + 15 + np.random.normal(0, 10, len(df))
        scores = np.clip(scores, 0, 100)
        
        return scores.astype(int)
    
    def get_features(self, 
                     feature_set: str = 'clustering',
                     normalize: bool = True) -> Tuple[pd.DataFrame, List[str]]:
        """
        Extract and optionally normalize features.
        
        Args:
            feature_set: 'clustering', 'all', or 'prediction'
            normalize: Whether to standardize features
            
        Returns:
            Tuple of (feature DataFrame, feature names)
        """
        if self.processed_data is None:
            self.clean_data()
            
        df = self.processed_data
        
        # Get actual column names for features
        if feature_set == 'clustering':
            target_features = self.CLUSTERING_FEATURES
        else:
            target_features = self.AUDIO_FEATURES
            
        # Map to actual column names
        features = []
        for std_name in target_features:
            col = self.column_mapping.get(std_name)
            if col and col in df.columns:
                features.append(col)
            elif std_name in df.columns:
                features.append(std_name)
        
        if not features:
            # Fallback: use any numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude ID-like columns
            features = [c for c in numeric_cols if 'id' not in c.lower() and 'index' not in c.lower()][:6]
            print(f"⚠ Using fallback numeric columns: {features}")
            
        X = df[features].copy()
        
        # Fill any remaining NaN
        X = X.fillna(X.median())
        
        if normalize:
            X_scaled = self.scaler.fit_transform(X)
            X = pd.DataFrame(X_scaled, columns=features, index=df.index)
            
        return X, features
    
    def prepare_prediction_data(self,
                                target: str = 'popularity',
                                test_size: float = 0.2,
                                random_state: int = 42) -> Tuple:
        """
        Prepare data for supervised learning (popularity prediction).
        
        Args:
            target: Target column name (standard or actual)
            test_size: Proportion for test set
            random_state: Random seed
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        if self.processed_data is None:
            self.clean_data()
            
        df = self.processed_data
        
        # Get actual target column name
        target_col = self.column_mapping.get(target, target)
        
        if target_col not in df.columns:
            raise ValueError(f"Target '{target}' not found in data. Available: {list(df.columns)}")
            
        # Get features
        X, features = self.get_features(feature_set='all', normalize=False)
        y = df[target_col]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=features)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state
        )
        
        print(f"✓ Split data: Train={len(X_train)}, Test={len(X_test)}")
        return X_train, X_test, y_train, y_test
    
    def get_track_metadata(self) -> pd.DataFrame:
        """Get track metadata (name, artist, etc.) if available."""
        if self.processed_data is None:
            self.clean_data()
        
        # Get actual column names for metadata
        metadata_std = ['track_name', 'artist', 'album', 'genre']
        metadata_cols = []
        
        for std_name in metadata_std:
            col = self.column_mapping.get(std_name)
            if col and col in self.processed_data.columns:
                metadata_cols.append(col)
        
        if not metadata_cols:
            # Try to find any text columns
            text_cols = self.processed_data.select_dtypes(include=['object']).columns.tolist()
            metadata_cols = text_cols[:4]  # Take first 4 text columns
            
        return self.processed_data[metadata_cols] if metadata_cols else pd.DataFrame()
    
    def save_processed_data(self, output_path: str):
        """Save processed data to CSV."""
        if self.processed_data is None:
            raise ValueError("No processed data to save")
            
        self.processed_data.to_csv(output_path, index=False)
        print(f"✓ Saved processed data to {output_path}")
        
    def create_sample_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Create synthetic sample data for testing.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with synthetic music data
        """
        np.random.seed(42)
        
        data = {
            'track_name': [f'Track_{i}' for i in range(n_samples)],
            'track_artist': [f'Artist_{np.random.randint(1, 100)}' for _ in range(n_samples)],
            'danceability': np.random.uniform(0, 1, n_samples),
            'energy': np.random.uniform(0, 1, n_samples),
            'loudness': np.random.uniform(-60, 0, n_samples),
            'speechiness': np.random.uniform(0, 1, n_samples),
            'acousticness': np.random.uniform(0, 1, n_samples),
            'instrumentalness': np.random.uniform(0, 1, n_samples),
            'liveness': np.random.uniform(0, 1, n_samples),
            'valence': np.random.uniform(0, 1, n_samples),
            'tempo': np.random.uniform(60, 200, n_samples),
            'duration_ms': np.random.randint(60000, 600000, n_samples),
            'popularity': np.random.randint(0, 100, n_samples)
        }
        
        self.raw_data = pd.DataFrame(data)
        print(f"✓ Created {n_samples} synthetic tracks for testing")
        return self.raw_data


# Utility functions
def download_kaggle_dataset(dataset_name: str, output_dir: str):
    """
    Download dataset from Kaggle (requires kaggle API setup).
    
    Args:
        dataset_name: Kaggle dataset identifier (e.g., 'zaheenhamidani/ultimate-spotify-tracks-db')
        output_dir: Directory to save the dataset
    """
    try:
        import kaggle
        kaggle.api.dataset_download_files(dataset_name, path=output_dir, unzip=True)
        print(f"✓ Downloaded {dataset_name} to {output_dir}")
    except Exception as e:
        print(f"✗ Kaggle download failed: {e}")
        print("Manual download: https://www.kaggle.com/datasets/")
