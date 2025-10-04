"""
Machine learning models for exoplanet detection and classification
Implements Random Forest, XGBoost, and Neural Network models with astronomical optimizations
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import xgboost as xgb
from imblearn.over_sampling import SMOTE

# Deep learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (
    DISPOSITION_CONFIRMED,
    DISPOSITION_CANDIDATE,
    DISPOSITION_FALSE_POSITIVE,
    TRAINING_DISPOSITIONS
)


class ExoplanetClassifier:
    """
    Main class for exoplanet classification using multiple ML models
    
    Supports:
    - Random Forest
    - XGBoost
    - Neural Network
    
    Handles:
    - Data loading and preprocessing
    - Class imbalance with SMOTE
    - Model training and evaluation
    - Model persistence
    - Prediction with confidence scores
    """
    
    def __init__(self, model_type: str = 'random_forest', random_state: int = 42):
        """
        Initialize classifier
        
        Args:
            model_type: 'random_forest', 'xgboost', or 'neural_network'
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.class_names = []
        self.training_history = {}
        
    def load_data(
        self,
        data_path: str,
        dataset_name: str = 'kepler',
        test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and prepare data for training
        
        Args:
            data_path: Path to CSV file
            dataset_name: Name for logging
            test_size: Fraction of data for test set
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        print(f"\n{'='*60}")
        print(f"Loading {dataset_name.upper()} dataset")
        print(f"{'='*60}")
        
        # Load data
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df):,} observations")
        
        # Filter to training dispositions only
        df = df[df['disposition'].isin(TRAINING_DISPOSITIONS)].copy()
        print(f"After filtering: {len(df):,} training samples")
        
        # Define features to use
        self.feature_names = [
            'orbital_period',
            'transit_duration',
            'planetary_radius',
            'stellar_magnitude',
            'transit_depth',
            'impact_parameter',
            'equilibrium_temperature',
            'stellar_radius',
            'stellar_mass'
        ]
        
        # Extract features and target
        X = df[self.feature_names].values
        y = df['disposition'].values
        
        # Handle missing values
        print(f"\nHandling missing values...")
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(X)
        
        # Check class distribution
        print(f"\nClass distribution:")
        unique, counts = np.unique(y, return_counts=True)
        for label, count in zip(unique, counts):
            pct = (count / len(y)) * 100
            print(f"  {label:20s}: {count:6,} ({pct:5.1f}%)")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"\nTrain/Test split:")
        print(f"  Training samples: {len(X_train):,}")
        print(f"  Test samples:     {len(X_test):,}")
        
        return X_train, X_test, y_train, y_test
    
    def preprocess_data(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        use_smote: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess features and handle class imbalance
        
        Args:
            X_train, X_test, y_train, y_test: Train/test splits
            use_smote: Whether to apply SMOTE for class balancing
            
        Returns:
            Preprocessed X_train, X_test, y_train, y_test
        """
        print(f"\n{'='*60}")
        print("Preprocessing Data")
        print(f"{'='*60}")
        
        # Scale features
        print("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Encode labels
        print("Encoding labels...")
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        self.class_names = self.label_encoder.classes_
        
        # Handle class imbalance with SMOTE
        if use_smote:
            print("\nApplying SMOTE for class balancing...")
            print(f"  Before SMOTE: {len(X_train_scaled):,} samples")
            
            smote = SMOTE(random_state=self.random_state)
            X_train_scaled, y_train_encoded = smote.fit_resample(
                X_train_scaled, y_train_encoded
            )
            
            print(f"  After SMOTE:  {len(X_train_scaled):,} samples")
            
            # Show new distribution
            unique, counts = np.unique(y_train_encoded, return_counts=True)
            print(f"\n  Balanced class distribution:")
            for idx, count in zip(unique, counts):
                label = self.class_names[idx]
                pct = (count / len(y_train_encoded)) * 100
                print(f"    {label:20s}: {count:6,} ({pct:5.1f}%)")
        
        return X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded
    
    def build_random_forest(self, n_estimators: int = 200) -> RandomForestClassifier:
        """Build Random Forest classifier"""
        print(f"\nBuilding Random Forest with {n_estimators} trees...")
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight='balanced',
            verbose=1
        )
    
    def build_xgboost(self) -> xgb.XGBClassifier:
        """Build XGBoost classifier"""
        print(f"\nBuilding XGBoost classifier...")
        return xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
    
    def build_neural_network(self, input_dim: int, num_classes: int) -> keras.Model:
        """Build Neural Network classifier"""
        print(f"\nBuilding Neural Network...")
        print(f"  Input features: {input_dim}")
        print(f"  Output classes: {num_classes}")
        
        model = models.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        model.summary()
        return model
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None
    ):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (for neural network)
            y_val: Validation labels (for neural network)
        """
        print(f"\n{'='*60}")
        print(f"Training {self.model_type.upper()} Model")
        print(f"{'='*60}")
        
        if self.model_type == 'random_forest':
            self.model = self.build_random_forest()
            self.model.fit(X_train, y_train)
            
        elif self.model_type == 'xgboost':
            self.model = self.build_xgboost()
            self.model.fit(X_train, y_train)
            
        elif self.model_type == 'neural_network':
            num_classes = len(np.unique(y_train))
            self.model = self.build_neural_network(X_train.shape[1], num_classes)
            
            # Callbacks
            early_stop = callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            reduce_lr = callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
            
            # Train
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val) if X_val is not None else None,
                epochs=100,
                batch_size=32,
                callbacks=[early_stop, reduce_lr],
                verbose=1
            )
            
            self.training_history = history.history
        
        print(f"\n✅ Training complete!")
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        save_plots: bool = True,
        output_dir: str = '../models'
    ) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test labels
            save_plots: Whether to save visualization plots
            output_dir: Directory to save plots
            
        Returns:
            Dictionary of metrics
        """
        print(f"\n{'='*60}")
        print("Evaluating Model Performance")
        print(f"{'='*60}")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        if self.model_type == 'neural_network':
            y_pred_proba = self.model.predict(X_test)
            y_pred = np.argmax(y_pred_proba, axis=1)
        else:
            y_pred_proba = self.model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        # Print metrics
        print(f"\n📊 Overall Metrics:")
        print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        
        # Per-class metrics
        print(f"\n📋 Classification Report:")
        print(classification_report(
            y_test, y_pred,
            target_names=self.class_names,
            digits=4
        ))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n🔍 Confusion Matrix:")
        print(cm)
        
        # Save visualizations
        if save_plots:
            os.makedirs(output_dir, exist_ok=True)
            self._plot_confusion_matrix(cm, output_dir)
            self._plot_feature_importance(output_dir)
            if self.model_type == 'neural_network' and self.training_history:
                self._plot_training_history(output_dir)
        
        return metrics
    
    def _plot_confusion_matrix(self, cm: np.ndarray, output_dir: str):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title(f'Confusion Matrix - {self.model_type.upper()}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        filename = os.path.join(output_dir, f'{self.model_type}_confusion_matrix.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  💾 Saved confusion matrix: {filename}")
        plt.close()
    
    def _plot_feature_importance(self, output_dir: str):
        """Plot and save feature importance (for tree-based models)"""
        if self.model_type in ['random_forest', 'xgboost']:
            plt.figure(figsize=(10, 6))
            
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                plt.bar(range(len(importances)), importances[indices])
                plt.xticks(
                    range(len(importances)),
                    [self.feature_names[i] for i in indices],
                    rotation=45,
                    ha='right'
                )
                plt.title(f'Feature Importance - {self.model_type.upper()}')
                plt.ylabel('Importance')
                plt.xlabel('Feature')
                plt.tight_layout()
                
                filename = os.path.join(output_dir, f'{self.model_type}_feature_importance.png')
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"  💾 Saved feature importance: {filename}")
                plt.close()
    
    def _plot_training_history(self, output_dir: str):
        """Plot and save training history (for neural network)"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss
        ax1.plot(self.training_history['loss'], label='Training Loss')
        if 'val_loss' in self.training_history:
            ax1.plot(self.training_history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy
        ax2.plot(self.training_history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in self.training_history:
            ax2.plot(self.training_history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = os.path.join(output_dir, f'{self.model_type}_training_history.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  💾 Saved training history: {filename}")
        plt.close()
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions
        
        Args:
            X: Features to predict
            
        Returns:
            predictions, probabilities
        """
        X_scaled = self.scaler.transform(X)
        
        if self.model_type == 'neural_network':
            probabilities = self.model.predict(X_scaled, verbose=0)
            predictions = np.argmax(probabilities, axis=1)
        else:
            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)
        
        # Decode labels
        predictions_decoded = self.label_encoder.inverse_transform(predictions)
        
        return predictions_decoded, probabilities
    
    def save_model(self, output_dir: str = '../models'):
        """Save trained model and artifacts"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = f'{self.model_type}_{timestamp}'
        
        # Save model
        if self.model_type == 'neural_network':
            model_path = os.path.join(output_dir, f'{base_name}.h5')
            self.model.save(model_path)
        else:
            model_path = os.path.join(output_dir, f'{base_name}.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
        
        # Save scaler and encoder
        scaler_path = os.path.join(output_dir, f'{base_name}_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        encoder_path = os.path.join(output_dir, f'{base_name}_encoder.pkl')
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Save metadata
        metadata = {
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'class_names': self.class_names.tolist(),
            'timestamp': timestamp
        }
        
        metadata_path = os.path.join(output_dir, f'{base_name}_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n💾 Model saved:")
        print(f"  Model:    {model_path}")
        print(f"  Scaler:   {scaler_path}")
        print(f"  Encoder:  {encoder_path}")
        print(f"  Metadata: {metadata_path}")
        
        return model_path


def main():
    """Main training pipeline"""
    print("\n" + "█"*60)
    print("█" + " "*58 + "█")
    print("█" + "  EXOPLANET AI HUNTER - MODEL TRAINING".center(58) + "█")
    print("█" + " "*58 + "█")
    print("█"*60)
    
    # Configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    DATA_PATH = os.path.join(project_root, 'data', 'processed', 'kepler_standardized.csv')
    MODEL_DIR = os.path.join(project_root, 'models')
    MODEL_TYPE = 'random_forest'  # 'random_forest', 'xgboost', or 'neural_network'
    
    # Initialize classifier
    classifier = ExoplanetClassifier(model_type=MODEL_TYPE)
    
    # Load data
    X_train, X_test, y_train, y_test = classifier.load_data(DATA_PATH)
    
    # Preprocess
    X_train, X_test, y_train, y_test = classifier.preprocess_data(
        X_train, X_test, y_train, y_test, use_smote=True
    )
    
    # Train
    if MODEL_TYPE == 'neural_network':
        # Split train into train/val for neural network
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        classifier.train(X_train, y_train, X_val, y_val)
    else:
        classifier.train(X_train, y_train)
    
    # Evaluate
    metrics = classifier.evaluate(X_test, y_test, output_dir=MODEL_DIR)
    
    # Save model
    classifier.save_model(output_dir=MODEL_DIR)
    
    print("\n" + "█"*60)
    print("█" + " "*58 + "█")
    print("█" + "  TRAINING COMPLETE!".center(58) + "█")
    print("█" + " "*58 + "█")
    print("█"*60 + "\n")
    
    print(f"🎯 Final Results:")
    print(f"  Model Type: {MODEL_TYPE}")
    print(f"  Accuracy:   {metrics['accuracy']*100:.2f}%")
    print(f"  F1-Score:   {metrics['f1_score']:.4f}")
    print(f"\n✨ Model ready for deployment!")


if __name__ == "__main__":
    main()
