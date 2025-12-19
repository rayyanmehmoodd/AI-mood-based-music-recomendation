# Unsupervised Learning for Music Clustering

## Overview
This project uses **unsupervised learning** to automatically group similar songs based on audio features like energy, valence, danceability, etc. **No labels required** - the algorithms find patterns in the data themselves!

## ❓ What is Unsupervised Learning?
Unlike supervised learning (where you train on labeled data), unsupervised learning discovers hidden patterns and structures in unlabeled data. Perfect for:
- Grouping similar items (clustering)
- Finding anomalies/outliers
- Dimensionality reduction

## 🎯 Three Clustering Algorithms Implemented

### 1. **K-Means Clustering** (Most Popular)
**How it works:**
- Partitions data into K spherical clusters
- Each song belongs to the cluster with the nearest centroid (center point)
- Iteratively updates centroids until convergence

**Pros:**
- ✅ Fast and efficient
- ✅ Works well with spherical clusters
- ✅ Easy to interpret

**Cons:**
- ❌ Must specify K (number of clusters) upfront
- ❌ Sensitive to outliers
- ❌ Assumes clusters are roughly equal size

**Best for:** Large datasets with well-separated clusters

**Example in this project:**
```python
from src.clustering import MusicClusterer

clusterer = MusicClusterer(n_clusters=5)
labels = clusterer.fit_kmeans(X_scaled, k=5)
```

**Finding Optimal K:**
- **Elbow Method**: Plot inertia (within-cluster sum of squares) vs K, look for "elbow"
- **Silhouette Score**: Measures how similar songs are to their own cluster vs others (range: -1 to 1, higher is better)
- **Calinski-Harabasz Score**: Ratio of between-cluster to within-cluster variance (higher is better)

---

### 2. **Hierarchical Clustering** (Tree-Based)
**How it works:**
- Builds a tree (dendrogram) of nested clusters
- Can use different linkage methods:
  - **Ward**: Minimizes within-cluster variance (default)
  - **Complete**: Maximum distance between clusters
  - **Average**: Average distance between all pairs
  - **Single**: Minimum distance between clusters

**Pros:**
- ✅ Don't need to specify K upfront
- ✅ Creates a hierarchy of clusters
- ✅ Deterministic (same result every time)

**Cons:**
- ❌ Slower for large datasets
- ❌ Cannot undo merges (greedy approach)

**Best for:** Exploring cluster hierarchies, smaller datasets

**Example:**
```python
labels = clusterer.fit_hierarchical(X_scaled, n_clusters=5, linkage='ward')
```

---

### 3. **DBSCAN** (Density-Based)
**How it works:**
- Groups songs in dense regions
- Songs in low-density regions are labeled as outliers/noise (-1)
- Two parameters:
  - **eps**: Maximum distance between two samples to be neighbors
  - **min_samples**: Minimum number of samples in a neighborhood

**Pros:**
- ✅ Finds arbitrary-shaped clusters
- ✅ Detects outliers automatically
- ✅ No need to specify K

**Cons:**
- ❌ Sensitive to parameter tuning
- ❌ Struggles with varying densities
- ❌ Not ideal for high-dimensional data

**Best for:** Finding outliers, non-spherical clusters

**Example:**
```python
labels = clusterer.fit_dbscan(X_scaled, eps=0.5, min_samples=5)
# -1 labels indicate noise/outliers
```

---

## 📊 Evaluation Metrics

### Silhouette Score
- **Range**: -1 to 1
- **Interpretation**: 
  - Close to 1: Well-separated clusters
  - Close to 0: Overlapping clusters
  - Negative: Songs might be in wrong cluster
- **Typical values**: 0.2-0.5 for real-world data

### Calinski-Harabasz Score
- **Range**: 0 to ∞
- **Interpretation**: Higher is better (ratio of between-cluster to within-cluster variance)
- **No fixed threshold**: Compare relative values

### Davies-Bouldin Score
- **Range**: 0 to ∞
- **Interpretation**: Lower is better (average similarity of clusters)
- **Good values**: < 1.0

---

## 🎵 Audio Features Used for Clustering

The system uses these Spotify audio features:

1. **danceability** (0-1): How suitable for dancing
2. **energy** (0-1): Intensity and activity
3. **valence** (0-1): Musical positiveness (happy vs sad)
4. **loudness** (dB): Overall loudness
5. **tempo** (BPM): Speed/pace
6. **acousticness** (0-1): Likelihood of being acoustic

---

## 🔧 Usage in Streamlit App

1. Navigate to **"🔍 Explore Clusters"** mode
2. Choose your algorithm (K-Means, Hierarchical, or DBSCAN)
3. Adjust parameters:
   - **K-Means/Hierarchical**: Set number of clusters (2-10)
   - **DBSCAN**: Set eps and min_samples
4. Click **"▶️ Run Clustering"**
5. View:
   - Evaluation metrics
   - Cluster profiles (average feature values)
   - 2D PCA visualization
   - Cluster characteristics (auto-interpreted)

---

## 💡 Why NOT K-Nearest Neighbors (KNN)?

**KNN is SUPERVISED learning** - it requires labeled training data to classify new instances. It's used for:
- Classification (predict labels)
- Regression (predict values)

**Clustering is UNSUPERVISED** - it discovers groups without labels.

If you wanted to use KNN in this project, you'd need:
1. Pre-labeled songs (e.g., "happy", "sad", "energetic")
2. Train KNN on labeled data
3. Predict labels for new songs

But we're using clustering to **discover** groups automatically!

---

## 📈 When to Use Each Algorithm

| Scenario | Best Algorithm |
|----------|---------------|
| Large dataset, need speed | K-Means |
| Want to explore hierarchy | Hierarchical |
| Need to detect outliers | DBSCAN |
| Know ideal number of clusters | K-Means or Hierarchical |
| Don't know K | DBSCAN or Hierarchical |
| Non-spherical clusters | DBSCAN |
| Interpretability important | K-Means |

---

## 🧪 Example Workflow

```python
from src.data_processor import DataProcessor
from src.clustering import MusicClusterer

# Load and process data
processor = DataProcessor()
df = processor.load_data('data/raw/sample_music.csv')
X_scaled = processor.preprocess_for_clustering(df)

# Initialize clusterer
clusterer = MusicClusterer(n_clusters=5)

# Method 1: Find optimal K first
results = clusterer.find_optimal_k(X_scaled, k_range=(2, 10))
optimal_k = results['k'][np.argmax(results['silhouette'])]

# Method 2: Run clustering
labels = clusterer.fit_kmeans(X_scaled, k=optimal_k)

# Evaluate
metrics = clusterer.evaluate_clustering(X_scaled, labels)
print(f"Silhouette: {metrics['silhouette_score']:.3f}")

# Get profiles
profiles = clusterer.get_cluster_profiles(df[['danceability', 'energy', 'valence']], labels)
print(profiles)

# Interpret
interpretations = clusterer.interpret_clusters(profiles)
for cluster_id, description in interpretations.items():
    print(f"Cluster {cluster_id}: {description}")
```

---

## 🎓 Key Takeaways

1. **Unsupervised learning** = no labels needed, algorithm finds patterns
2. **K-Means** = fast, good for spherical clusters, need to specify K
3. **Hierarchical** = creates tree structure, no K needed upfront
4. **DBSCAN** = finds outliers, arbitrary shapes, parameter-sensitive
5. **KNN is supervised** - not for clustering!
6. Use **silhouette score** to evaluate clustering quality
7. **PCA** reduces dimensions for visualization (2D plot)

---

## 📚 Further Reading

- [scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html)
- [K-Means Deep Dive](https://en.wikipedia.org/wiki/K-means_clustering)
- [DBSCAN Explained](https://en.wikipedia.org/wiki/DBSCAN)
- [Silhouette Analysis](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html)
