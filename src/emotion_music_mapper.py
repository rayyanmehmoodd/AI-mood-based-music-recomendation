"""
Emotion to Music Mapper
Bridges the gap between detected emotions and music clusters.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import random


class EmotionMusicMapper:
    """
    Maps detected emotions to appropriate music clusters and songs.
    
    This is the integration layer that connects:
    - Computer Vision (Emotion Detection)
    - Unsupervised Learning (Music Clustering)
    - Supervised Learning (Popularity Prediction)
    """
    
    # Default emotion to audio feature mapping
    # Maps emotions to preferred audio characteristic ranges
    EMOTION_AUDIO_PROFILE = {
        'Happy': {
            'valence': (0.6, 1.0),       # High positivity
            'energy': (0.5, 1.0),         # Medium-high energy
            'danceability': (0.5, 1.0),   # Danceable
            'tempo': (100, 180),          # Upbeat tempo
            'loudness': (-15, 0),         # Not too quiet
            'acousticness': (0.0, 0.5)    # Can be electronic
        },
        'Sad': {
            'valence': (0.0, 0.4),        # Low positivity
            'energy': (0.0, 0.5),         # Low energy
            'danceability': (0.0, 0.5),   # Less danceable
            'tempo': (60, 100),           # Slower tempo
            'loudness': (-30, -10),       # Quieter
            'acousticness': (0.4, 1.0)    # More acoustic
        },
        'Angry': {
            'valence': (0.0, 0.5),        # Lower positivity
            'energy': (0.7, 1.0),         # High energy
            'danceability': (0.3, 0.8),   # Variable
            'tempo': (120, 200),          # Fast tempo
            'loudness': (-10, 0),         # Loud
            'acousticness': (0.0, 0.3)    # Less acoustic
        },
        'Fear': {
            'valence': (0.2, 0.5),        # Medium-low
            'energy': (0.3, 0.6),         # Medium energy
            'danceability': (0.2, 0.6),   # Variable
            'tempo': (80, 140),           # Medium tempo
            'loudness': (-25, -10),       # Moderate volume
            'acousticness': (0.3, 0.8)    # More acoustic
        },
        'Surprise': {
            'valence': (0.4, 0.9),        # Generally positive
            'energy': (0.5, 1.0),         # Higher energy
            'danceability': (0.4, 0.9),   # Danceable
            'tempo': (100, 160),          # Upbeat
            'loudness': (-15, -5),        # Moderate-loud
            'acousticness': (0.0, 0.6)    # Variable
        },
        'Disgust': {
            'valence': (0.3, 0.6),        # Neutral
            'energy': (0.3, 0.6),         # Medium
            'danceability': (0.3, 0.6),   # Medium
            'tempo': (80, 120),           # Moderate
            'loudness': (-20, -10),       # Medium
            'acousticness': (0.3, 0.7)    # Variable
        },
        'Neutral': {
            'valence': (0.3, 0.7),        # Balanced
            'energy': (0.3, 0.7),         # Balanced
            'danceability': (0.3, 0.7),   # Balanced
            'tempo': (80, 140),           # Moderate
            'loudness': (-20, -5),        # Moderate
            'acousticness': (0.2, 0.8)    # Variable
        }
    }
    
    # Mood descriptions for UI
    MOOD_DESCRIPTIONS = {
        'Happy': "You're feeling happy! 🎉 Here are some upbeat, positive tracks to match your mood.",
        'Sad': "Feeling a bit down? 🌧️ Here are some soothing, reflective tracks.",
        'Angry': "Intense mood detected! 🔥 Here's some powerful music to channel that energy.",
        'Fear': "Need some comfort? 🌙 Here are calming tracks to help you relax.",
        'Surprise': "Feeling excited! ⚡ Here are some dynamic tracks for you.",
        'Disgust': "Looking for a change? 🎵 Here's a balanced mix for you.",
        'Neutral': "Feeling balanced 😊 Here's a diverse playlist for your neutral mood."
    }
    
    def __init__(self, 
                 cluster_profiles: Optional[pd.DataFrame] = None,
                 cluster_labels: Optional[Dict[int, str]] = None):
        """
        Initialize the EmotionMusicMapper.
        
        Args:
            cluster_profiles: DataFrame with cluster feature means
            cluster_labels: Dictionary mapping cluster IDs to descriptions
        """
        self.cluster_profiles = cluster_profiles
        self.cluster_labels = cluster_labels or {}
        self.emotion_cluster_map: Dict[str, List[int]] = {}
        
    def set_cluster_profiles(self, profiles: pd.DataFrame, 
                              labels: Optional[Dict[int, str]] = None):
        """
        Set cluster profiles and optionally their labels.
        
        Args:
            profiles: DataFrame with cluster feature means (index = cluster ID)
            labels: Optional cluster descriptions
        """
        self.cluster_profiles = profiles
        if labels:
            self.cluster_labels = labels
            
        # Automatically map emotions to clusters
        self._build_emotion_cluster_map()
        
    def _build_emotion_cluster_map(self):
        """Build mapping from emotions to suitable clusters."""
        if self.cluster_profiles is None:
            return
            
        for emotion, profile in self.EMOTION_AUDIO_PROFILE.items():
            matching_clusters = []
            
            for cluster_id in self.cluster_profiles.index:
                if cluster_id == -1:  # Skip noise cluster
                    continue
                    
                cluster = self.cluster_profiles.loc[cluster_id]
                score = self._compute_match_score(cluster, profile)
                matching_clusters.append((cluster_id, score))
                
            # Sort by score and keep top matches
            matching_clusters.sort(key=lambda x: x[1], reverse=True)
            self.emotion_cluster_map[emotion] = [c[0] for c in matching_clusters[:3]]
            
    def _compute_match_score(self, cluster: pd.Series, profile: Dict) -> float:
        """
        Compute how well a cluster matches an emotion profile.
        
        Args:
            cluster: Cluster feature values
            profile: Target feature ranges
            
        Returns:
            Match score (higher is better)
        """
        score = 0.0
        count = 0
        
        for feature, (min_val, max_val) in profile.items():
            if feature in cluster.index:
                value = cluster[feature]
                
                # Normalize feature if needed
                if feature == 'tempo':
                    value = (value - 60) / 140  # Normalize to ~0-1
                elif feature == 'loudness':
                    value = (value + 60) / 60  # Normalize to ~0-1
                    min_val = (min_val + 60) / 60
                    max_val = (max_val + 60) / 60
                    
                # Score based on how close to target range
                if min_val <= value <= max_val:
                    # Perfect match - in range
                    score += 1.0
                else:
                    # Partial match - distance from range
                    if value < min_val:
                        score += max(0, 1 - (min_val - value))
                    else:
                        score += max(0, 1 - (value - max_val))
                        
                count += 1
                
        return score / count if count > 0 else 0.0
    
    def get_clusters_for_emotion(self, emotion: str) -> List[int]:
        """
        Get recommended cluster IDs for an emotion.
        
        Args:
            emotion: Detected emotion string
            
        Returns:
            List of cluster IDs (best matches first)
        """
        emotion = emotion.capitalize()
        
        if emotion in self.emotion_cluster_map:
            return self.emotion_cluster_map[emotion]
        elif emotion in self.EMOTION_AUDIO_PROFILE:
            # Build mapping on-the-fly if not already done
            self._build_emotion_cluster_map()
            return self.emotion_cluster_map.get(emotion, [])
        else:
            # Return all clusters for unknown emotions
            if self.cluster_profiles is not None:
                return [c for c in self.cluster_profiles.index if c != -1]
            return []
    
    def recommend_songs(self, 
                        emotion: str,
                        music_data: pd.DataFrame,
                        cluster_labels: np.ndarray,
                        popularity_scores: Optional[np.ndarray] = None,
                        n_songs: int = 10,
                        prioritize_popular: bool = True) -> pd.DataFrame:
        """
        Recommend songs based on detected emotion.
        
        Args:
            emotion: Detected emotion
            music_data: DataFrame with song data
            cluster_labels: Cluster assignments for each song
            popularity_scores: Optional predicted popularity scores
            n_songs: Number of songs to recommend
            prioritize_popular: Whether to prioritize by popularity
            
        Returns:
            DataFrame with recommended songs
        """
        emotion = emotion.capitalize()
        
        # STEP 1: Filter by emotion-matched clusters (if cluster profiles exist)
        if self.cluster_profiles is not None:
            target_clusters = self.get_clusters_for_emotion(emotion)
            if target_clusters and 'cluster' in music_data.columns:
                # Filter to only songs in matching clusters
                filtered_data = music_data[music_data['cluster'].isin(target_clusters)].copy()
                if len(filtered_data) == 0:
                    # Fallback if no songs in target clusters
                    filtered_data = music_data.copy()
            else:
                filtered_data = music_data.copy()
        else:
            filtered_data = music_data.copy()
        
        # STEP 2: Get emotion audio profile for fine-tuning
        profile = self.get_audio_profile(emotion)
        
        # STEP 3: Score each song based on how well it matches the emotion profile
        filtered_data['emotion_match_score'] = 0.0
        
        for idx, song in filtered_data.iterrows():
            score = 0.0
            count = 0
            
            for feature, (min_val, max_val) in profile.items():
                if feature in song.index and pd.notna(song[feature]):
                    value = song[feature]
                    
                    # Normalize if needed
                    if feature == 'tempo':
                        norm_value = (value - 60) / 140
                        norm_min = (min_val - 60) / 140
                        norm_max = (max_val - 60) / 140
                    elif feature == 'loudness':
                        norm_value = (value + 60) / 60
                        norm_min = (min_val + 60) / 60
                        norm_max = (max_val + 60) / 60
                    else:
                        norm_value = value
                        norm_min = min_val
                        norm_max = max_val
                    
                    # Score based on how close to target range
                    if norm_min <= norm_value <= norm_max:
                        score += 1.0
                    else:
                        if norm_value < norm_min:
                            score += max(0, 1 - (norm_min - norm_value))
                        else:
                            score += max(0, 1 - (norm_value - norm_max))
                    
                    count += 1
            
            filtered_data.at[idx, 'emotion_match_score'] = score / count if count > 0 else 0.0
        
        # Filter to top matching songs (top 40% by match score)
        threshold = filtered_data['emotion_match_score'].quantile(0.6)
        filtered_data = filtered_data[filtered_data['emotion_match_score'] >= threshold]
        
        if len(filtered_data) == 0:
            # Fallback to all songs if no matches
            filtered_data = music_data.copy()
        
        # Remove duplicate songs (based on track name if available)
        if 'track_name' in filtered_data.columns:
            filtered_data = filtered_data.drop_duplicates(subset=['track_name'], keep='first')
        
        # Add popularity scores if provided
        if popularity_scores is not None:
            filtered_data['predicted_popularity'] = popularity_scores[filtered_data.index]
        elif 'popularity' in filtered_data.columns:
            filtered_data['predicted_popularity'] = filtered_data['popularity']
        elif 'track_popularity' in filtered_data.columns:
            filtered_data['predicted_popularity'] = filtered_data['track_popularity']
        else:
            filtered_data['predicted_popularity'] = 50  # Default
            
        # Sort by combination of emotion match and popularity
        if prioritize_popular:
            # Combine emotion match (70%) and popularity (30%)
            filtered_data['final_score'] = (
                0.7 * filtered_data['emotion_match_score'] + 
                0.3 * (filtered_data['predicted_popularity'] / 100.0)
            )
            filtered_data = filtered_data.sort_values('final_score', ascending=False)
        else:
            filtered_data = filtered_data.sort_values('emotion_match_score', ascending=False)
            
        # Return top N songs
        result = filtered_data.head(n_songs)
        
        # Add cluster info
        result = result.copy()
        if len(result) <= len(cluster_labels):
            result['cluster'] = cluster_labels[result.index]
        result['matched_emotion'] = emotion
        
        return result
    
    def get_mood_description(self, emotion: str) -> str:
        """Get user-friendly mood description."""
        emotion = emotion.capitalize()
        return self.MOOD_DESCRIPTIONS.get(emotion, f"Mood: {emotion}")
    
    def get_audio_profile(self, emotion: str) -> Dict:
        """Get target audio profile for an emotion."""
        emotion = emotion.capitalize()
        return self.EMOTION_AUDIO_PROFILE.get(emotion, self.EMOTION_AUDIO_PROFILE['Neutral'])
    
    def create_playlist(self,
                        emotion: str,
                        music_data: pd.DataFrame,
                        cluster_labels: np.ndarray,
                        popularity_scores: Optional[np.ndarray] = None,
                        playlist_size: int = 20,
                        diversity: float = 0.3) -> Dict:
        """
        Create a complete playlist with metadata.
        
        Args:
            emotion: Detected emotion
            music_data: DataFrame with song data
            cluster_labels: Cluster assignments
            popularity_scores: Optional predicted popularity
            playlist_size: Number of songs
            diversity: Fraction of songs from secondary clusters (0-1)
            
        Returns:
            Dictionary with playlist data and metadata
        """
        emotion = emotion.capitalize()
        
        # Primary recommendations
        primary_size = int(playlist_size * (1 - diversity))
        primary_songs = self.recommend_songs(
            emotion, music_data, cluster_labels, popularity_scores,
            n_songs=primary_size, prioritize_popular=True
        )
        
        # Add some diversity from other clusters
        diversity_size = playlist_size - len(primary_songs)
        if diversity_size > 0:
            # Get songs from other clusters
            target_clusters = self.get_clusters_for_emotion(emotion)
            other_mask = ~np.isin(cluster_labels, target_clusters)
            other_songs = music_data[other_mask].sample(
                n=min(diversity_size, other_mask.sum())
            )
        else:
            other_songs = pd.DataFrame()
            
        # Combine
        playlist = pd.concat([primary_songs, other_songs], ignore_index=True)
        
        return {
            'emotion': emotion,
            'mood_description': self.get_mood_description(emotion),
            'audio_profile': self.get_audio_profile(emotion),
            'playlist': playlist,
            'total_songs': len(playlist),
            'primary_cluster_songs': len(primary_songs),
            'target_clusters': self.get_clusters_for_emotion(emotion)
        }
    
    def explain_recommendation(self, emotion: str, song: pd.Series) -> str:
        """
        Generate explanation for why a song was recommended.
        
        Args:
            emotion: Detected emotion
            song: Song data Series
            
        Returns:
            Human-readable explanation
        """
        emotion = emotion.capitalize()
        profile = self.get_audio_profile(emotion)
        
        reasons = []
        
        if 'valence' in song.index:
            val_range = profile.get('valence', (0, 1))
            if val_range[0] <= song['valence'] <= val_range[1]:
                if song['valence'] > 0.6:
                    reasons.append("positive mood")
                elif song['valence'] < 0.4:
                    reasons.append("reflective tone")
                    
        if 'energy' in song.index:
            if song['energy'] > 0.7:
                reasons.append("high energy")
            elif song['energy'] < 0.3:
                reasons.append("calm atmosphere")
                
        if 'tempo' in song.index:
            if song['tempo'] > 140:
                reasons.append("upbeat tempo")
            elif song['tempo'] < 90:
                reasons.append("relaxed pace")
                
        if 'acousticness' in song.index and song['acousticness'] > 0.7:
            reasons.append("acoustic sound")
            
        if reasons:
            return f"Recommended for its {', '.join(reasons)}"
        else:
            return "Matches your current mood"
    
    def get_cluster_mood_distribution(self) -> pd.DataFrame:
        """
        Get the mapping of emotions to clusters as a DataFrame.
        
        Returns:
            DataFrame showing emotion-cluster relationships
        """
        if not self.emotion_cluster_map:
            self._build_emotion_cluster_map()
            
        rows = []
        for emotion, clusters in self.emotion_cluster_map.items():
            for i, cluster in enumerate(clusters):
                rows.append({
                    'emotion': emotion,
                    'cluster': cluster,
                    'rank': i + 1,
                    'cluster_label': self.cluster_labels.get(cluster, f'Cluster {cluster}')
                })
                
        return pd.DataFrame(rows)
