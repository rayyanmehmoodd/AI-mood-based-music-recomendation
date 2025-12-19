"""
Music Clustering Module
Implements K-Means, Hierarchical Clustering, and DBSCAN for grouping songs.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Optional, List
import joblib
import os


class MusicClusterer:
    """
    Music clustering system using multiple algorithms.
    
    Supports:
    - K-Means Clustering
    - Hierarchical (Agglomerative) Clustering  
    - DBSCAN
    - PCA for dimensionality reduction
    """
    
    # Cluster interpretation mapping
    CLUSTER_MOODS = {
        'high_energy_happy': {'energy': 'high', 'valence': 'high', 'danceability': 'high'},
        'calm_peaceful': {'energy': 'low', 'acousticness': 'high', 'valence': 'medium'},
        'sad_melancholic': {'energy': 'low', 'valence': 'low', 'acousticness': 'high'},
        'intense_angry': {'energy': 'high', 'valence': 'low', 'loudness': 'high'},
        'party_upbeat': {'danceability': 'high', 'energy': 'high', 'tempo': 'high'},
        'focus_instrumental': {'instrumentalness': 'high', 'speechiness': 'low'}
    }
    
    def __init__(self, n_clusters: int = 5, random_state: int = 42):
        """
        Initialize the MusicClusterer.
        
        Args:
            n_clusters: Default number of clusters
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.models: Dict = {}
        self.pca: Optional[PCA] = None
        self.scaler = StandardScaler()
        self.cluster_labels: Optional[np.ndarray] = None
        self.cluster_centers: Optional[np.ndarray] = None
        self.feature_names: List[str] = []
        
    def fit_pca(self, X: np.ndarray, n_components: int = 2) -> np.ndarray:
        """
        Apply PCA for dimensionality reduction.
        
        Args:
            X: Feature matrix
            n_components: Number of components to keep
            
        Returns:
            Transformed data
        """
        self.pca = PCA(n_components=n_components, random_state=self.random_state)
        X_pca = self.pca.fit_transform(X)
        
        explained_var = sum(self.pca.explained_variance_ratio_) * 100
        print(f"✓ PCA: {n_components} components explain {explained_var:.1f}% variance")
        
        return X_pca
    
    def fit_kmeans(self, X: np.ndarray, n_clusters: Optional[int] = None) -> np.ndarray:
        """
        Fit K-Means clustering.
        
        Args:
            X: Feature matrix
            n_clusters: Number of clusters (uses default if None)
            
        Returns:
            Cluster labels
        """
        k = n_clusters or self.n_clusters
        
        kmeans = KMeans(
            n_clusters=k,
            random_state=self.random_state,
            n_init=10,
            max_iter=300
        )
        
        self.cluster_labels = kmeans.fit_predict(X)
        self.cluster_centers = kmeans.cluster_centers_
        self.models['kmeans'] = kmeans
        
        inertia = kmeans.inertia_
        print(f"✓ K-Means: {k} clusters, Inertia={inertia:.2f}")
        
        return self.cluster_labels
    
    def fit_hierarchical(self, X: np.ndarray, n_clusters: Optional[int] = None,
                         linkage: str = 'ward') -> np.ndarray:
        """
        Fit Hierarchical (Agglomerative) clustering.
        
        Args:
            X: Feature matrix
            n_clusters: Number of clusters
            linkage: Linkage criterion ('ward', 'complete', 'average', 'single')
            
        Returns:
            Cluster labels
        """
        k = n_clusters or self.n_clusters
        
        hierarchical = AgglomerativeClustering(
            n_clusters=k,
            linkage=linkage
        )
        
        self.cluster_labels = hierarchical.fit_predict(X)
        self.models['hierarchical'] = hierarchical
        
        print(f"✓ Hierarchical: {k} clusters, Linkage={linkage}")
        
        return self.cluster_labels
    
    def fit_dbscan(self, X: np.ndarray, eps: float = 0.5, 
                   min_samples: int = 5) -> np.ndarray:
        """
        Fit DBSCAN clustering.
        
        Args:
            X: Feature matrix
            eps: Maximum distance between samples
            min_samples: Minimum samples in neighborhood
            
        Returns:
            Cluster labels (-1 indicates noise)
        """
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.cluster_labels = dbscan.fit_predict(X)
        self.models['dbscan'] = dbscan
        
        n_clusters = len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0)
        n_noise = list(self.cluster_labels).count(-1)
        
        print(f"✓ DBSCAN: {n_clusters} clusters, {n_noise} noise points")
        
        return self.cluster_labels
    
    def find_optimal_k(self, X: np.ndarray, k_range: Tuple[int, int] = (2, 10)) -> Dict:
        """
        Find optimal number of clusters using multiple metrics.
        
        Args:
            X: Feature matrix
            k_range: Range of k values to test (min, max)
            
        Returns:
            Dictionary with evaluation metrics for each k
        """
        results = {
            'k': [],
            'inertia': [],
            'silhouette': [],
            'calinski_harabasz': [],
            'davies_bouldin': []
        }
        
        for k in range(k_range[0], k_range[1] + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X)
            
            results['k'].append(k)
            results['inertia'].append(kmeans.inertia_)
            results['silhouette'].append(silhouette_score(X, labels))
            results['calinski_harabasz'].append(calinski_harabasz_score(X, labels))
            results['davies_bouldin'].append(davies_bouldin_score(X, labels))
            
        # Find optimal k (highest silhouette)
        optimal_k = results['k'][np.argmax(results['silhouette'])]
        print(f"✓ Optimal k={optimal_k} (highest silhouette score)")
        
        return results
    
    def evaluate_clustering(self, X: np.ndarray, labels: Optional[np.ndarray] = None) -> Dict:
        """
        Evaluate clustering quality.
        
        Args:
            X: Feature matrix
            labels: Cluster labels (uses stored labels if None)
            
        Returns:
            Dictionary of evaluation metrics
        """
        labels = labels if labels is not None else self.cluster_labels
        
        if labels is None:
            raise ValueError("No cluster labels available")
            
        # Filter out noise points for metrics
        mask = labels != -1
        X_filtered = X[mask]
        labels_filtered = labels[mask]
        
        if len(set(labels_filtered)) < 2:
            return {'error': 'Need at least 2 clusters for evaluation'}
            
        metrics = {
            'silhouette_score': silhouette_score(X_filtered, labels_filtered),
            'calinski_harabasz_score': calinski_harabasz_score(X_filtered, labels_filtered),
            'davies_bouldin_score': davies_bouldin_score(X_filtered, labels_filtered),
            'n_clusters': len(set(labels_filtered)),
            'cluster_sizes': dict(zip(*np.unique(labels, return_counts=True)))
        }
        
        print(f"✓ Silhouette: {metrics['silhouette_score']:.3f}")
        return metrics
    
    def get_cluster_profiles(self, X: pd.DataFrame, labels: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Generate interpretable profiles for each cluster.
        
        Args:
            X: Feature DataFrame with column names
            labels: Cluster labels
            
        Returns:
            DataFrame with mean feature values per cluster
        """
        labels = labels if labels is not None else self.cluster_labels
        
        if labels is None:
            raise ValueError("No cluster labels available")
            
        X_copy = X.copy()
        X_copy['cluster'] = labels
        
        profiles = X_copy.groupby('cluster').mean()
        
        # Add cluster sizes
        sizes = X_copy.groupby('cluster').size()
        profiles['size'] = sizes
        
        return profiles
    
    def interpret_clusters(self, profiles: pd.DataFrame) -> Dict[int, str]:
        """
        Automatically interpret cluster characteristics.
        
        Args:
            profiles: Cluster profiles DataFrame
            
        Returns:
            Dictionary mapping cluster ID to mood description
        """
        interpretations = {}
        
        for cluster_id in profiles.index:
            if cluster_id == -1:
                interpretations[-1] = "Noise/Outliers"
                continue
                
            row = profiles.loc[cluster_id]
            
            # Determine characteristics
            traits = []
            
            if 'energy' in row:
                if row['energy'] > 0.7:
                    traits.append("High Energy")
                elif row['energy'] < 0.3:
                    traits.append("Low Energy")
                    
            if 'valence' in row:
                if row['valence'] > 0.6:
                    traits.append("Happy/Positive")
                elif row['valence'] < 0.4:
                    traits.append("Sad/Melancholic")
                    
            if 'danceability' in row:
                if row['danceability'] > 0.7:
                    traits.append("Danceable")
                    
            if 'acousticness' in row:
                if row['acousticness'] > 0.7:
                    traits.append("Acoustic")
                    
            if 'tempo' in row:
                if row['tempo'] > 140:
                    traits.append("Fast Tempo")
                elif row['tempo'] < 90:
                    traits.append("Slow Tempo")
                    
            interpretations[cluster_id] = " / ".join(traits) if traits else f"Cluster {cluster_id}"
            
        return interpretations
    
    def predict_cluster(self, X: np.ndarray, algorithm: str = 'kmeans') -> np.ndarray:
        """
        Predict cluster for new data points.
        
        Args:
            X: Feature matrix for new data
            algorithm: Which model to use
            
        Returns:
            Predicted cluster labels
        """
        if algorithm not in self.models:
            raise ValueError(f"Model '{algorithm}' not trained")
            
        if algorithm == 'kmeans':
            return self.models['kmeans'].predict(X)
        else:
            # For hierarchical and DBSCAN, use nearest centroid approach
            # (simplified - in practice might need different approach)
            raise NotImplementedError(f"Prediction not implemented for {algorithm}")
    
    def plot_clusters(self, X: np.ndarray, labels: Optional[np.ndarray] = None,
                      title: str = "Music Clusters") -> plt.Figure:
        """
        Visualize clusters in 2D space.
        
        Args:
            X: Feature matrix (2D or will be reduced via PCA)
            labels: Cluster labels
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        labels = labels if labels is not None else self.cluster_labels
        
        # Reduce to 2D if needed
        if X.shape[1] > 2:
            pca = PCA(n_components=2, random_state=self.random_state)
            X_2d = pca.fit_transform(X)
        else:
            X_2d = X
            
        fig, ax = plt.subplots(figsize=(10, 8))
        
        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, 
                            cmap='viridis', alpha=0.6, s=50)
        
        plt.colorbar(scatter, ax=ax, label='Cluster')
        ax.set_xlabel('PC1' if X.shape[1] > 2 else 'Feature 1')
        ax.set_ylabel('PC2' if X.shape[1] > 2 else 'Feature 2')
        ax.set_title(title)
        
        return fig
    
    def plot_elbow(self, results: Dict) -> plt.Figure:
        """
        Plot elbow curve for optimal k selection.
        
        Args:
            results: Output from find_optimal_k()
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Inertia (Elbow)
        axes[0, 0].plot(results['k'], results['inertia'], 'bo-')
        axes[0, 0].set_xlabel('Number of Clusters (k)')
        axes[0, 0].set_ylabel('Inertia')
        axes[0, 0].set_title('Elbow Method')
        
        # Silhouette Score
        axes[0, 1].plot(results['k'], results['silhouette'], 'go-')
        axes[0, 1].set_xlabel('Number of Clusters (k)')
        axes[0, 1].set_ylabel('Silhouette Score')
        axes[0, 1].set_title('Silhouette Analysis')
        
        # Calinski-Harabasz
        axes[1, 0].plot(results['k'], results['calinski_harabasz'], 'ro-')
        axes[1, 0].set_xlabel('Number of Clusters (k)')
        axes[1, 0].set_ylabel('Calinski-Harabasz Score')
        axes[1, 0].set_title('Calinski-Harabasz Index')
        
        # Davies-Bouldin
        axes[1, 1].plot(results['k'], results['davies_bouldin'], 'mo-')
        axes[1, 1].set_xlabel('Number of Clusters (k)')
        axes[1, 1].set_ylabel('Davies-Bouldin Score')
        axes[1, 1].set_title('Davies-Bouldin Index (lower is better)')
        
        plt.tight_layout()
        return fig
    
    def save_model(self, filepath: str, algorithm: str = 'kmeans'):
        """Save trained model to disk."""
        if algorithm not in self.models:
            raise ValueError(f"Model '{algorithm}' not trained")
            
        model_data = {
            'model': self.models[algorithm],
            'pca': self.pca,
            'scaler': self.scaler,
            'n_clusters': self.n_clusters,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, filepath)
        print(f"✓ Saved {algorithm} model to {filepath}")
        
    def load_model(self, filepath: str) -> str:
        """Load trained model from disk."""
        model_data = joblib.load(filepath)
        
        algorithm = 'kmeans'  # Default assumption
        if hasattr(model_data['model'], 'linkage'):
            algorithm = 'hierarchical'
        elif hasattr(model_data['model'], 'eps'):
            algorithm = 'dbscan'
            
        self.models[algorithm] = model_data['model']
        self.pca = model_data.get('pca')
        self.scaler = model_data.get('scaler', StandardScaler())
        self.n_clusters = model_data.get('n_clusters', 5)
        self.feature_names = model_data.get('feature_names', [])
        
        print(f"✓ Loaded {algorithm} model from {filepath}")
        return algorithm
