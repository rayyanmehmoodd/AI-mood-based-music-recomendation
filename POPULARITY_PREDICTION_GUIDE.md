# Popularity Prediction Guide

## Overview
This module uses **supervised learning** to predict song popularity (0-100 score) based on audio features like energy, danceability, valence, tempo, etc.

## 🤔 What is Supervised Learning?
Unlike unsupervised learning (clustering), supervised learning requires **labeled training data** - songs with known popularity scores. The model learns patterns from these examples to predict popularity of new, unseen songs.

---

## 🎯 Algorithms Implemented

### 1. **Random Forest Regressor** ✅ (Recommended)
**How it works:**
- Builds multiple decision trees (typically 100+)
- Each tree votes on the prediction
- Final prediction is the average of all trees
- Uses random subsets of features and data for each tree

**Pros:**
- ✅ Handles non-linear relationships
- ✅ Robust to outliers
- ✅ Provides feature importance
- ✅ Minimal hyperparameter tuning needed
- ✅ Works well with small-medium datasets

**Cons:**
- ❌ Can overfit on very noisy data
- ❌ Slower prediction than linear models
- ❌ Less interpretable than linear regression

**Typical Performance:**
- R² score: 0.15-0.30 (popularity is inherently noisy!)
- RMSE: ~15-20 points

**When to use:** Default choice for this task

---

### 2. **Gradient Boosting Regressor**
**How it works:**
- Builds trees sequentially
- Each new tree corrects errors of previous trees
- Uses gradient descent to minimize loss
- More sophisticated than Random Forest

**Pros:**
- ✅ Often slightly better accuracy than Random Forest
- ✅ Captures complex patterns
- ✅ Feature importance available

**Cons:**
- ❌ Slower to train
- ❌ More sensitive to hyperparameters
- ❌ Can overfit if not tuned properly

**Typical Performance:**
- R² score: 0.18-0.32
- RMSE: ~14-19 points

**When to use:** When you need best possible accuracy and have time for tuning

---

### 3. **Linear Regression** (with Ridge/Lasso variants)
**How it works:**
- Fits a linear equation: `popularity = w₁×feature₁ + w₂×feature₂ + ... + bias`
- **Ridge**: Adds L2 regularization (penalizes large weights)
- **Lasso**: Adds L1 regularization (can zero out features)

**Pros:**
- ✅ Fast training and prediction
- ✅ Highly interpretable coefficients
- ✅ Good baseline model
- ✅ Works well with linearly correlated features

**Cons:**
- ❌ Assumes linear relationships (rarely true for music)
- ❌ Lower accuracy than ensemble methods
- ❌ Sensitive to outliers (standard Linear Regression)

**Typical Performance:**
- R² score: 0.08-0.20
- RMSE: ~18-23 points

**When to use:** As a baseline or when interpretability is critical

---

### 4. **Artificial Neural Network (ANN)**
**How it works:**
- Multiple layers of neurons (nodes)
- Each layer learns increasingly complex patterns
- Uses backpropagation to learn weights
- Activation functions introduce non-linearity

**Architecture used:**
```
Input Layer → Dense(64) → Dropout → Dense(32) → Dropout → Output(1)
```

**Pros:**
- ✅ Can learn very complex patterns
- ✅ Scales well with large datasets
- ✅ Flexible architecture

**Cons:**
- ❌ Requires more training data
- ❌ Slower to train
- ❌ Black box (hard to interpret)
- ❌ Prone to overfitting on small datasets
- ❌ Needs careful hyperparameter tuning

**Typical Performance:**
- R² score: 0.12-0.28 (depends heavily on data size)
- RMSE: ~16-21 points

**When to use:** Large datasets (>10,000 songs), when interpretability doesn't matter

---

## ❓ Why Not Logistic Regression?

**Logistic Regression is for CLASSIFICATION, not regression!**

| Task | Output | Algorithm |
|------|--------|-----------|
| Predict category (e.g., "Hit" vs "Flop") | Discrete class | Logistic Regression ✅ |
| Predict continuous value (e.g., popularity 0-100) | Numerical score | Linear/Random Forest/etc. ✅ |

If you wanted to use Logistic Regression, you'd need to:
1. Convert popularity to categories (e.g., Low/Medium/High)
2. Train a classifier
3. Lose granularity (can't distinguish between scores like 45 vs 55)

**For popularity prediction, we use regression algorithms, not classification!**

---

## 📊 Evaluation Metrics

### R² Score (Coefficient of Determination)
- **Range**: -∞ to 1
- **Interpretation**:
  - 1.0: Perfect predictions
  - 0.0: Model no better than predicting the mean
  - Negative: Model worse than baseline
- **Typical values for popularity**: 0.15-0.30
- **Why so low?** Popularity is influenced by many external factors (marketing, artist fame, timing, luck) that audio features can't capture

### RMSE (Root Mean Squared Error)
- **Range**: 0 to ∞
- **Interpretation**: Average prediction error in popularity points
- **Example**: RMSE=18 means predictions are off by ±18 points on average
- **Good values**: <20 points

### MAE (Mean Absolute Error)
- **Range**: 0 to ∞
- **Interpretation**: Average absolute error (less sensitive to outliers than RMSE)
- **Good values**: <15 points

---

## 🎵 Audio Features Used

These Spotify audio features are used for prediction:

1. **danceability** (0-1): How suitable for dancing
2. **energy** (0-1): Intensity and activity level
3. **loudness** (dB): Overall loudness (-60 to 0)
4. **speechiness** (0-1): Presence of spoken words
5. **acousticness** (0-1): Likelihood of being acoustic
6. **instrumentalness** (0-1): Lack of vocals
7. **liveness** (0-1): Probability of live performance
8. **valence** (0-1): Musical positiveness (happy vs sad)
9. **tempo** (BPM): Speed/pace (typically 60-200)
10. **duration_ms** (ms): Song length in milliseconds

**Most Important Features** (from Random Forest feature importance):
1. Energy
2. Loudness
3. Danceability
4. Valence
5. Acousticness

---

## 🚀 Usage in Streamlit App

### Train Models
1. Go to **"📊 Data Analytics"** mode
2. Select **"Popularity Prediction"** tab
3. Choose algorithm or "Train All Models"
4. Set test split percentage (20% recommended)
5. Click **"🚀 Train Model(s)"**
6. View performance metrics and feature importance

### Predict New Song Popularity

**Option 1: Upload CSV**
1. Prepare CSV with same features as training data
2. Use template: `data/processed/new_song_template.csv`
3. Upload in "Predict New Song Popularity" section
4. View predictions and download results

**Option 2: Manual Input**
1. Enter audio feature values manually using sliders/inputs
2. Click **"🔮 Predict Popularity"**
3. View predicted score and gauge chart

---

## 📁 CSV Format for New Songs

```csv
track_name,danceability,energy,loudness,speechiness,acousticness,instrumentalness,liveness,valence,tempo,duration_ms
My Song,0.75,0.8,-5.5,0.05,0.15,0.0,0.12,0.85,128.0,210000
```

**Required columns:** All features used during training (detected automatically)

**Optional columns:** 
- `track_name`: Song title
- `artist_name`: Artist name
- Any other metadata (will be preserved in output)

---

## 💡 Tips for Better Predictions

1. **Train on more data**: R² improves with dataset size (aim for 1000+ songs)
2. **Feature engineering**: Try creating new features (e.g., energy × valence)
3. **Remove outliers**: Songs with extreme popularity (0 or 100) may hurt model
4. **Balance dataset**: Ensure variety of popularity ranges
5. **Use Random Forest or Gradient Boosting**: Usually outperform linear models
6. **Cross-validate**: Use CV to ensure model generalizes well
7. **Accept inherent noise**: R² of 0.25 is actually good for this problem!

---

## 🧪 Example Workflow

```python
from src.data_processor import DataProcessor
from src.popularity_predictor import PopularityPredictor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
processor = DataProcessor()
df = processor.load_data('data/raw/sample_music.csv')

# Prepare features and target
features = ['danceability', 'energy', 'loudness', 'valence', 'tempo', 'acousticness']
X = df[features].values
y = df['popularity'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest
predictor = PopularityPredictor()
predictor.train_random_forest(X_train_scaled, y_train)

# Evaluate
results = predictor.evaluate(X_test_scaled, y_test)
print(f"R² Score: {results['random_forest']['R2']:.3f}")
print(f"RMSE: {results['random_forest']['RMSE']:.2f}")

# Feature importance
importance = predictor.get_feature_importance('random_forest')
print(importance)

# Predict new song
new_song = [[0.75, 0.8, -5.5, 0.85, 128.0, 0.15]]  # Your audio features
new_song_scaled = scaler.transform(new_song)
prediction = predictor.predict(new_song_scaled, model_name='random_forest')
print(f"Predicted Popularity: {prediction[0]:.1f}")
```

---

## 📈 When to Use Each Algorithm

| Scenario | Best Algorithm |
|----------|---------------|
| Default/first choice | Random Forest |
| Need best accuracy | Gradient Boosting |
| Want interpretability | Linear Regression |
| Have 10,000+ songs | ANN |
| Need fast predictions | Linear Regression |
| Small dataset (<500 songs) | Random Forest + Cross-validation |
| Feature importance needed | Random Forest or Gradient Boosting |

---

## 🎓 Key Takeaways

1. **Supervised learning** = learns from labeled data (songs with known popularity)
2. **Random Forest** is the best default choice for this task
3. **R² of 0.25 is good** - popularity is inherently noisy due to external factors
4. **Logistic Regression is for classification**, not regression (discrete vs continuous)
5. **Feature importance** shows which audio features matter most
6. **Scale features** before training (especially for ANN and Linear Regression)
7. **Cross-validation** ensures model generalizes well
8. **Upload CSV or manual input** to predict new songs

---

## 🚨 Common Mistakes

❌ Using Logistic Regression for continuous values
✅ Use Linear/Random Forest/etc. for regression

❌ Expecting R² > 0.5 for popularity prediction
✅ R² of 0.20-0.30 is realistic due to external factors

❌ Not scaling features
✅ Always use StandardScaler before training

❌ Training on full dataset (no test set)
✅ Hold out 20% for testing

❌ Comparing models on different data splits
✅ Use same train/test split for fair comparison

---

## 📚 Further Reading

- [Scikit-learn Regression](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning)
- [Random Forest Explained](https://en.wikipedia.org/wiki/Random_forest)
- [Understanding R² Score](https://en.wikipedia.org/wiki/Coefficient_of_determination)
- [Gradient Boosting](https://en.wikipedia.org/wiki/Gradient_boosting)
- [Feature Importance Analysis](https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html)
