"""
Popularity Prediction Module
Supervised learning models to predict song popularity from audio features.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional, List, Any
import joblib

# Try to import TensorFlow for ANN
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class PopularityPredictor:
    """
    Popularity prediction system using multiple regression algorithms.
    
    Supports:
    - Linear Regression (with Ridge/Lasso variants)
    - Random Forest Regressor
    - Gradient Boosting Regressor  
    - Artificial Neural Network (ANN)
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the PopularityPredictor.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Dict] = {}
        self.best_model_name: Optional[str] = None
        self.feature_names: List[str] = []
        
    def train_linear_regression(self, X_train: np.ndarray, y_train: np.ndarray,
                                 variant: str = 'standard') -> 'LinearRegression':
        """
        Train Linear Regression model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            variant: 'standard', 'ridge', or 'lasso'
            
        Returns:
            Trained model
        """
        if variant == 'ridge':
            model = Ridge(alpha=1.0, random_state=self.random_state)
            name = 'ridge_regression'
        elif variant == 'lasso':
            model = Lasso(alpha=0.1, random_state=self.random_state)
            name = 'lasso_regression'
        else:
            model = LinearRegression()
            name = 'linear_regression'
            
        model.fit(X_train, y_train)
        self.models[name] = model
        
        print(f"✓ Trained {name}")
        return model
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray,
                            n_estimators: int = 100,
                            max_depth: Optional[int] = None,
                            tune_hyperparameters: bool = False) -> RandomForestRegressor:
        """
        Train Random Forest Regressor.
        
        Args:
            X_train: Training features
            y_train: Training targets
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            tune_hyperparameters: Whether to use GridSearchCV
            
        Returns:
            Trained model
        """
        if tune_hyperparameters:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            rf = RandomForestRegressor(random_state=self.random_state)
            grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='r2', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            
            model = grid_search.best_estimator_
            print(f"✓ Best RF params: {grid_search.best_params_}")
        else:
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=self.random_state,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            
        self.models['random_forest'] = model
        print(f"✓ Trained Random Forest ({model.n_estimators} trees)")
        
        return model
    
    def train_gradient_boosting(self, X_train: np.ndarray, y_train: np.ndarray,
                                 n_estimators: int = 100,
                                 learning_rate: float = 0.1) -> GradientBoostingRegressor:
        """
        Train Gradient Boosting Regressor.
        
        Args:
            X_train: Training features
            y_train: Training targets
            n_estimators: Number of boosting stages
            learning_rate: Learning rate
            
        Returns:
            Trained model
        """
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=5,
            random_state=self.random_state
        )
        
        model.fit(X_train, y_train)
        self.models['gradient_boosting'] = model
        
        print(f"✓ Trained Gradient Boosting ({n_estimators} stages)")
        return model
    
    def train_ann(self, X_train: np.ndarray, y_train: np.ndarray,
                  X_val: Optional[np.ndarray] = None,
                  y_val: Optional[np.ndarray] = None,
                  epochs: int = 100,
                  batch_size: int = 32) -> Any:
        """
        Train Artificial Neural Network for regression.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Training epochs
            batch_size: Batch size
            
        Returns:
            Trained Keras model
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not installed. Run: pip install tensorflow")
            
        input_dim = X_train.shape[1]
        
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(16, activation='relu'),
            Dense(1, activation='linear')  # Regression output
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        # Prepare validation data
        if X_val is None or y_val is None:
            validation_split = 0.2
            validation_data = None
        else:
            validation_split = None
            validation_data = (X_val, y_val)
            
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=0
        )
        
        self.models['ann'] = model
        self.models['ann_history'] = history
        
        print(f"✓ Trained ANN ({len(history.history['loss'])} epochs)")
        return model
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_val: Optional[np.ndarray] = None,
                         y_val: Optional[np.ndarray] = None) -> Dict:
        """
        Train all available models for comparison.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (for ANN)
            y_val: Validation targets (for ANN)
            
        Returns:
            Dictionary of trained models
        """
        print("Training all models...")
        
        # Linear models
        self.train_linear_regression(X_train, y_train, 'standard')
        self.train_linear_regression(X_train, y_train, 'ridge')
        self.train_linear_regression(X_train, y_train, 'lasso')
        
        # Ensemble models
        self.train_random_forest(X_train, y_train)
        self.train_gradient_boosting(X_train, y_train)
        
        # Neural network
        if TF_AVAILABLE:
            self.train_ann(X_train, y_train, X_val, y_val)
        else:
            print("⚠ Skipping ANN (TensorFlow not available)")
            
        return self.models
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                 model_name: Optional[str] = None) -> Dict:
        """
        Evaluate model(s) on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets
            model_name: Specific model to evaluate (None = all)
            
        Returns:
            Dictionary of evaluation metrics
        """
        models_to_eval = {model_name: self.models[model_name]} if model_name else self.models
        
        results = {}
        
        for name, model in models_to_eval.items():
            if name == 'ann_history':
                continue
                
            # Get predictions
            if hasattr(model, 'predict'):
                y_pred = model.predict(X_test)
                if hasattr(y_pred, 'flatten'):
                    y_pred = y_pred.flatten()
                    
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results[name] = {
                    'MSE': mse,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R2': r2,
                    'predictions': y_pred
                }
                
        self.results = results
        
        # Find best model
        r2_scores = {k: v['R2'] for k, v in results.items()}
        self.best_model_name = max(r2_scores, key=r2_scores.get)
        
        return results
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray,
                       model_name: str = 'random_forest',
                       cv: int = 5) -> Dict:
        """
        Perform cross-validation for a model.
        
        Args:
            X: Features
            y: Targets
            model_name: Model to validate
            cv: Number of folds
            
        Returns:
            Cross-validation results
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not trained")
            
        model = self.models[model_name]
        
        # Skip ANN (different validation approach)
        if model_name == 'ann':
            print("⚠ Use train_ann with validation data for ANN cross-validation")
            return {}
            
        scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        
        return {
            'model': model_name,
            'cv_scores': scores,
            'mean_r2': scores.mean(),
            'std_r2': scores.std()
        }
    
    def predict(self, X: np.ndarray, model_name: Optional[str] = None) -> np.ndarray:
        """
        Make predictions using trained model.
        
        Args:
            X: Features for prediction
            model_name: Model to use (None = best model)
            
        Returns:
            Predicted popularity scores
        """
        name = model_name or self.best_model_name
        
        if name is None or name not in self.models:
            raise ValueError(f"Model '{name}' not available")
            
        model = self.models[name]
        predictions = model.predict(X)
        
        # Clip predictions to valid range [0, 100]
        predictions = np.clip(predictions.flatten(), 0, 100)
        
        return predictions
    
    def get_feature_importance(self, model_name: str = 'random_forest') -> pd.DataFrame:
        """
        Get feature importance from tree-based models.
        
        Args:
            model_name: Model name ('random_forest' or 'gradient_boosting')
            
        Returns:
            DataFrame with feature importances
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not trained")
            
        model = self.models[model_name]
        
        if not hasattr(model, 'feature_importances_'):
            raise ValueError(f"Model '{model_name}' doesn't have feature importances")
            
        importances = model.feature_importances_
        
        if self.feature_names:
            feature_names = self.feature_names
        else:
            feature_names = [f'feature_{i}' for i in range(len(importances))]
            
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return df
    
    def compare_models(self) -> pd.DataFrame:
        """
        Create comparison table of all trained models.
        
        Returns:
            DataFrame comparing model metrics
        """
        if not self.results:
            raise ValueError("No evaluation results. Call evaluate() first.")
            
        comparison = []
        
        for name, metrics in self.results.items():
            comparison.append({
                'Model': name,
                'R²': metrics['R2'],
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE']
            })
            
        df = pd.DataFrame(comparison).sort_values('R²', ascending=False)
        return df
    
    def plot_comparison(self) -> plt.Figure:
        """
        Plot model comparison bar chart.
        
        Returns:
            Matplotlib figure
        """
        df = self.compare_models()
        
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        
        # R² Score
        sns.barplot(x='Model', y='R²', data=df, ax=axes[0], palette='viridis')
        axes[0].set_title('R² Score (higher is better)')
        axes[0].tick_params(axis='x', rotation=45)
        
        # RMSE
        sns.barplot(x='Model', y='RMSE', data=df, ax=axes[1], palette='magma')
        axes[1].set_title('RMSE (lower is better)')
        axes[1].tick_params(axis='x', rotation=45)
        
        # MAE
        sns.barplot(x='Model', y='MAE', data=df, ax=axes[2], palette='plasma')
        axes[2].set_title('MAE (lower is better)')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
    def plot_predictions(self, y_true: np.ndarray, 
                         model_name: Optional[str] = None) -> plt.Figure:
        """
        Plot actual vs predicted values.
        
        Args:
            y_true: True values
            model_name: Model to plot (None = best model)
            
        Returns:
            Matplotlib figure
        """
        name = model_name or self.best_model_name
        
        if name not in self.results:
            raise ValueError(f"No results for '{name}'")
            
        y_pred = self.results[name]['predictions']
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.scatter(y_true, y_pred, alpha=0.5)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        ax.set_xlabel('Actual Popularity')
        ax.set_ylabel('Predicted Popularity')
        ax.set_title(f'{name}: Actual vs Predicted')
        ax.legend()
        
        r2 = self.results[name]['R2']
        ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes,
                fontsize=12, verticalalignment='top')
        
        return fig
    
    def plot_feature_importance(self, model_name: str = 'random_forest') -> plt.Figure:
        """
        Plot feature importance bar chart.
        
        Args:
            model_name: Model to use
            
        Returns:
            Matplotlib figure
        """
        df = self.get_feature_importance(model_name)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.barplot(x='importance', y='feature', data=df, ax=ax, palette='viridis')
        ax.set_title(f'Feature Importance ({model_name})')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        
        return fig
    
    def save_model(self, model_name: str, filepath: str):
        """Save trained model to disk."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not trained")
            
        model = self.models[model_name]
        
        if model_name == 'ann':
            model.save(filepath)
        else:
            joblib.dump(model, filepath)
            
        print(f"✓ Saved {model_name} to {filepath}")
        
    def load_model(self, model_name: str, filepath: str):
        """Load trained model from disk."""
        if model_name == 'ann':
            if not TF_AVAILABLE:
                raise ImportError("TensorFlow not available")
            model = load_model(filepath)
        else:
            model = joblib.load(filepath)
            
        self.models[model_name] = model
        print(f"✓ Loaded {model_name} from {filepath}")
