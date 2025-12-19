"""
Spotify API Integration Module
Fetch additional track data and audio features using Spotify Web API.

SETUP:
1. Create app at https://developer.spotify.com/dashboard/
2. Get Client ID and Client Secret
3. Add to config.py or config_local.py

FREE TIER: 
- No payment required
- Rate limit: ~180 requests/minute
"""

import requests
import base64
import pandas as pd
from typing import Optional, List, Dict
import time


class SpotifyAPI:
    """
    Spotify Web API client for fetching track data.
    
    Free tier features:
    - Search tracks
    - Get audio features
    - Get track info
    """
    
    BASE_URL = "https://api.spotify.com/v1"
    AUTH_URL = "https://accounts.spotify.com/api/token"
    
    def __init__(self, client_id: str, client_secret: str):
        """
        Initialize Spotify API client.
        
        Args:
            client_id: Spotify app client ID
            client_secret: Spotify app client secret
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token: Optional[str] = None
        self.token_expires: float = 0
        
    def _get_auth_header(self) -> str:
        """Get base64 encoded auth header."""
        credentials = f"{self.client_id}:{self.client_secret}"
        return base64.b64encode(credentials.encode()).decode()
    
    def authenticate(self) -> bool:
        """
        Authenticate with Spotify API using Client Credentials flow.
        
        Returns:
            True if authentication successful
        """
        if self.client_id == "your_spotify_client_id_here":
            print("⚠ Spotify API not configured. Add your credentials to config.py")
            return False
            
        headers = {
            "Authorization": f"Basic {self._get_auth_header()}",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        data = {"grant_type": "client_credentials"}
        
        try:
            response = requests.post(self.AUTH_URL, headers=headers, data=data)
            response.raise_for_status()
            
            result = response.json()
            self.access_token = result["access_token"]
            self.token_expires = time.time() + result["expires_in"] - 60
            
            print("✓ Spotify API authenticated")
            return True
            
        except Exception as e:
            print(f"✗ Spotify authentication failed: {e}")
            return False
    
    def _ensure_auth(self):
        """Ensure we have a valid token."""
        if not self.access_token or time.time() >= self.token_expires:
            self.authenticate()
    
    def _make_request(self, endpoint: str, params: dict = None) -> Optional[dict]:
        """Make authenticated request to Spotify API."""
        self._ensure_auth()
        
        if not self.access_token:
            return None
            
        headers = {"Authorization": f"Bearer {self.access_token}"}
        
        try:
            response = requests.get(
                f"{self.BASE_URL}/{endpoint}",
                headers=headers,
                params=params
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"✗ API request failed: {e}")
            return None
    
    def search_track(self, query: str, limit: int = 1) -> Optional[Dict]:
        """
        Search for a track.
        
        Args:
            query: Search query (track name, artist, etc.)
            limit: Number of results
            
        Returns:
            Search results
        """
        result = self._make_request("search", {
            "q": query,
            "type": "track",
            "limit": limit
        })
        
        if result and "tracks" in result:
            return result["tracks"]["items"]
        return None
    
    def get_audio_features(self, track_id: str) -> Optional[Dict]:
        """
        Get audio features for a track.
        
        Args:
            track_id: Spotify track ID
            
        Returns:
            Audio features dict
        """
        return self._make_request(f"audio-features/{track_id}")
    
    def get_audio_features_batch(self, track_ids: List[str]) -> Optional[List[Dict]]:
        """
        Get audio features for multiple tracks (max 100).
        
        Args:
            track_ids: List of Spotify track IDs
            
        Returns:
            List of audio features
        """
        # Spotify API limit is 100 tracks per request
        track_ids = track_ids[:100]
        
        result = self._make_request("audio-features", {
            "ids": ",".join(track_ids)
        })
        
        if result and "audio_features" in result:
            return result["audio_features"]
        return None
    
    def get_track(self, track_id: str) -> Optional[Dict]:
        """
        Get track information.
        
        Args:
            track_id: Spotify track ID
            
        Returns:
            Track information dict
        """
        return self._make_request(f"tracks/{track_id}")
    
    def enrich_dataframe(self, df: pd.DataFrame, 
                         track_col: str = 'track_name',
                         artist_col: str = 'artist') -> pd.DataFrame:
        """
        Enrich a DataFrame with Spotify audio features.
        
        Args:
            df: DataFrame with track names
            track_col: Column name for track names
            artist_col: Column name for artist names
            
        Returns:
            DataFrame with added audio features
        """
        if not self.authenticate():
            print("⚠ Cannot enrich data - Spotify API not available")
            return df
            
        print(f"Enriching {len(df)} tracks with Spotify data...")
        
        # Features to add
        spotify_features = ['danceability', 'energy', 'loudness', 'speechiness',
                           'acousticness', 'instrumentalness', 'liveness', 
                           'valence', 'tempo', 'popularity']
        
        # Initialize new columns
        for feature in spotify_features:
            if feature not in df.columns:
                df[feature] = None
        
        # Search and get features for each track
        found = 0
        for idx, row in df.iterrows():
            query = f"{row[track_col]}"
            if artist_col in df.columns:
                query += f" {row[artist_col]}"
                
            tracks = self.search_track(query, limit=1)
            
            if tracks:
                track = tracks[0]
                track_id = track['id']
                
                # Get audio features
                features = self.get_audio_features(track_id)
                
                if features:
                    for feature in spotify_features:
                        if feature in features:
                            df.at[idx, feature] = features[feature]
                        elif feature == 'popularity':
                            df.at[idx, feature] = track.get('popularity', 50)
                    found += 1
            
            # Rate limiting
            time.sleep(0.1)
            
            # Progress update
            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{len(df)} tracks...")
        
        print(f"✓ Enriched {found}/{len(df)} tracks with Spotify data")
        return df


class LastFMAPI:
    """
    Last.fm API client for additional metadata.
    
    Free tier - Get API key at: https://www.last.fm/api/account/create
    """
    
    BASE_URL = "http://ws.audioscrobbler.com/2.0/"
    
    def __init__(self, api_key: str):
        """
        Initialize Last.fm API client.
        
        Args:
            api_key: Last.fm API key
        """
        self.api_key = api_key
    
    def _make_request(self, method: str, params: dict = None) -> Optional[dict]:
        """Make request to Last.fm API."""
        if self.api_key == "your_lastfm_api_key_here":
            return None
            
        request_params = {
            "method": method,
            "api_key": self.api_key,
            "format": "json",
            **(params or {})
        }
        
        try:
            response = requests.get(self.BASE_URL, params=request_params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"✗ Last.fm request failed: {e}")
            return None
    
    def get_track_info(self, track: str, artist: str) -> Optional[Dict]:
        """
        Get track information including tags.
        
        Args:
            track: Track name
            artist: Artist name
            
        Returns:
            Track info dict
        """
        return self._make_request("track.getInfo", {
            "track": track,
            "artist": artist
        })
    
    def get_similar_tracks(self, track: str, artist: str, limit: int = 10) -> Optional[List]:
        """
        Get similar tracks.
        
        Args:
            track: Track name
            artist: Artist name
            limit: Number of results
            
        Returns:
            List of similar tracks
        """
        result = self._make_request("track.getSimilar", {
            "track": track,
            "artist": artist,
            "limit": limit
        })
        
        if result and "similartracks" in result:
            return result["similartracks"].get("track", [])
        return None
    
    def get_top_tags(self, track: str, artist: str) -> Optional[List]:
        """
        Get top tags for a track.
        
        Args:
            track: Track name  
            artist: Artist name
            
        Returns:
            List of tags
        """
        result = self._make_request("track.getTopTags", {
            "track": track,
            "artist": artist
        })
        
        if result and "toptags" in result:
            return result["toptags"].get("tag", [])
        return None
