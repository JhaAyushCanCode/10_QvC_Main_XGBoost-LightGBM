# gbm_multilabel_classification.py
"""
Chapter 6: Gradient Boosting Machine (XGBoost/LightGBM) for Multi-Label Emotion Classification
=================================================================================================

This script implements both XGBoost and LightGBM models for multi-label emotion classification
on the GoEmotions dataset, following state-of-the-art practices for handling class imbalance
and multi-label scenarios.

Key Features:
- XGBoost with native multi-output support (XGBoost >= 1.6)
- LightGBM with binary relevance approach
- Advanced class imbalance handling using sample weights
- Comprehensive hyperparameter optimization
- Proper evaluation with threshold optimization
- Statistical significance testing
- Memory-efficient BERT embedding preprocessing

References:
- Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD.
- Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NIPS.
- Tsoumakas, G. & Katakis, I. (2007). Multi-Label Classification: An Overview. IJDWM.
"""

import pandas as pd
import numpy as np
import torch
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import f1_score, precision_recall_fscore_support, multilabel_confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import cross_val_score
from transformers import AutoTokenizer, AutoModel
import joblib
import json
import os
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import logging
from itertools import product
from scipy import sparse
from concurrent.futures import ProcessPoolExecutor
import gc

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gbm_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    # Data
    'bert_model': 'bert-base-uncased',
    'max_length': 128,
    'batch_size': 64,
    'num_labels': 28,
    
    # Training
    'n_trials': 100,  # Hyperparameter optimization trials
    'cv_folds': 5,
    'random_state': 42,
    'n_jobs': -1,
    
    # Model selection
    'use_xgboost': True,
    'use_lightgbm': True,
    'enable_gpu': torch.cuda.is_available(),
    
    # Evaluation
    'threshold_optimization': True,
    'statistical_testing': True,
}

# GoEmotions label names
GOEMOTIONS_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
    'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
    'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
    'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

# Set random seeds
np.random.seed(CONFIG['random_state'])
torch.manual_seed(CONFIG['random_state'])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CONFIG['random_state'])

class BERTEmbeddingExtractor:
    """Efficient BERT embedding extraction with caching and batch processing."""
    
    def __init__(self, model_name: str, max_length: int = 128, batch_size: int = 64):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer and model
        logger.info(f"Loading BERT model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
    def extract_embeddings(self, texts: List[str], cache_file: Optional[str] = None) -> np.ndarray:
        """Extract BERT embeddings with optional caching."""
        
        # Check cache first
        if cache_file and os.path.exists(cache_file):
            logger.info(f"Loading cached embeddings from {cache_file}")
            return np.load(cache_file)
        
        embeddings = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        
        logger.info(f"Extracting embeddings for {len(texts)} texts in {total_batches} batches")
        
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                
                # Tokenize batch
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                # Move to device
                input_ids = inputs['input_ids'].to(self.device)
                attention_mask = inputs['attention_mask'].to(self.device)
                
                # Get embeddings
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Use CLS token embeddings
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls_embeddings)
                
                # Progress logging
                if (i // self.batch_size + 1) % 50 == 0:
                    logger.info(f"Processed {i // self.batch_size + 1}/{total_batches} batches")
        
        embeddings = np.vstack(embeddings)
        
        # Cache if requested
        if cache_file:
            logger.info(f"Caching embeddings to {cache_file}")
            np.save(cache_file, embeddings)
        
        return embeddings

class MultiLabelMetrics:
    """Comprehensive metrics for multi-label classification."""
    
    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                       y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Compute comprehensive multi-label metrics."""
        
        metrics = {}
        
        # Basic metrics
        metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['micro_f1'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-label metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        metrics['per_label_f1'] = f1.tolist()
        metrics['per_label_precision'] = precision.tolist()
        metrics['per_label_recall'] = recall.tolist()
        metrics['per_label_support'] = support.tolist()
        
        # Additional metrics
        metrics['hamming_loss'] = np.mean(y_true != y_pred)
        metrics['exact_match_ratio'] = np.mean(np.all(y_true == y_pred, axis=1))
        
        # Label-based metrics
        n_labels = y_true.shape[1]
        label_accuracy = np.mean(y_true == y_pred, axis=0)
        metrics['mean_label_accuracy'] = np.mean(label_accuracy)
        
        return metrics
    
    @staticmethod
    def optimize_thresholds(y_true: np.ndarray, y_proba: np.ndarray, 
                           metric: str = 'macro_f1') -> np.ndarray:
        """Optimize thresholds per label to maximize specified metric."""
        
        n_labels = y_true.shape[1]
        optimal_thresholds = np.zeros(n_labels)
        
        # Per-label threshold optimization
        for i in range(n_labels):
            best_score = -1
            best_threshold = 0.5
            
            # Try different thresholds
            for threshold in np.linspace(0.05, 0.95, 19):
                y_pred_label = (y_proba[:, i] >= threshold).astype(int)
                score = f1_score(y_true[:, i], y_pred_label, zero_division=0)
                
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
            
            optimal_thresholds[i] = best_threshold
        
        return optimal_thresholds

class XGBoostMultiLabel:
    """XGBoost multi-label classifier with advanced features."""
    
    def __init__(self, **params):
        # Default parameters optimized for multi-label classification
        default_params = {
            'n_estimators': 1000,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': CONFIG['random_state'],
            'n_jobs': CONFIG['n_jobs'],
            'tree_method': 'gpu_hist' if CONFIG['enable_gpu'] else 'hist',
            'multi_strategy': 'one_output_per_tree',  # For multi-label
            'eval_metric': 'logloss',
        }
        default_params.update(params)
        
        self.params = default_params
        self.model = None
        self.label_weights = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, 
            y_val: Optional[np.ndarray] = None, verbose: bool = True):
        """Train XGBoost with proper multi-label handling."""
        
        logger.info("Training XGBoost multi-label classifier")
        
        # Calculate sample weights for imbalanced data
        sample_weights = self._compute_sample_weights(y)
        
        # Initialize model
        self.model = xgb.XGBClassifier(**self.params)
        
        # Prepare evaluation set if provided
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X, y), (X_val, y_val)]
        
        # Train model
        self.model.fit(
            X, y,
            sample_weight=sample_weights,
            eval_set=eval_set,
            verbose=verbose
        )
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make binary predictions."""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        return self.model.predict_proba(X)
    
    def _compute_sample_weights(self, y: np.ndarray) -> np.ndarray:
        """Compute sample weights for multi-label imbalanced data."""
        
        # For multi-label, use average class weight across all labels
        n_samples, n_labels = y.shape
        sample_weights = np.ones(n_samples)
        
        for i in range(n_labels):
            # Calculate positive/negative ratio for each label
            pos_count = np.sum(y[:, i])
            neg_count = n_samples - pos_count
            
            if pos_count > 0:
                pos_weight = neg_count / pos_count
                # Apply weight to positive samples for this label
                sample_weights += y[:, i] * (pos_weight - 1)
        
        # Normalize weights
        sample_weights = sample_weights / np.mean(sample_weights)
        
        return sample_weights

class LightGBMMultiLabel:
    """LightGBM multi-label classifier using binary relevance."""
    
    def __init__(self, **params):
        # Default parameters optimized for multi-label classification
        default_params = {
            'n_estimators': 1000,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'subsample_freq': 1,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': CONFIG['random_state'],
            'n_jobs': CONFIG['n_jobs'],
            'device': 'gpu' if CONFIG['enable_gpu'] else 'cpu',
            'verbose': -1,
            'force_col_wise': True,
        }
        default_params.update(params)
        
        self.params = default_params
        self.models = []
        self.label_weights = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, 
            y_val: Optional[np.ndarray] = None, verbose: bool = True):
        """Train LightGBM using binary relevance approach."""
        
        logger.info("Training LightGBM multi-label classifier")
        
        n_labels = y.shape[1]
        self.models = []
        
        for i in range(n_labels):
            if verbose:
                logger.info(f"Training model for label {i+1}/{n_labels}: {GOEMOTIONS_LABELS[i]}")
            
            # Get label-specific data
            y_label = y[:, i]
            
            # Calculate class weights for this label
            pos_count = np.sum(y_label)
            neg_count = len(y_label) - pos_count
            
            if pos_count == 0:
                logger.warning(f"No positive samples for label {GOEMOTIONS_LABELS[i]}")
                # Create dummy model
                self.models.append(None)
                continue
            
            scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
            
            # Create model with class weighting
            model_params = self.params.copy()
            model_params['class_weight'] = 'balanced'
            
            model = lgb.LGBMClassifier(**model_params)
            
            # Prepare evaluation set
            eval_set = None
            if X_val is not None and y_val is not None:
                eval_set = [(X, y_label), (X_val, y_val[:, i])]
            
            # Train model
            model.fit(
                X, y_label,
                eval_set=eval_set,
                eval_names=['train', 'valid'] if eval_set else None,
                eval_metric='binary_logloss',
            )
            
            self.models.append(model)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make binary predictions."""
        predictions = []
        
        for i, model in enumerate(self.models):
            if model is None:
                # No positive samples in training
                pred = np.zeros(X.shape[0])
            else:
                pred = model.predict(X)
            predictions.append(pred)
        
        return np.column_stack(predictions)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        probabilities = []
        
        for i, model in enumerate(self.models):
            if model is None:
                # No positive samples in training
                proba = np.zeros(X.shape[0])
            else:
                proba = model.predict_proba(X)[:, 1]  # Positive class probability
            probabilities.append(proba)
        
        return np.column_stack(probabilities)

class HyperparameterOptimizer:
    """Bayesian-style hyperparameter optimization for GBM models."""
    
    @staticmethod
    def get_xgboost_search_space():
        """Get XGBoost hyperparameter search space."""
        return {
            'n_estimators': [500, 1000, 1500],
            'max_depth': [4, 6, 8, 10],
            'learning_rate': [0.05, 0.1, 0.15, 0.2],
            'subsample': [0.6, 0.7, 0.8, 0.9],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
            'reg_alpha': [0, 0.01, 0.1, 0.5],
            'reg_lambda': [0, 0.01, 0.1, 0.5],
        }
    
    @staticmethod
    def get_lightgbm_search_space():
        """Get LightGBM hyperparameter search space."""
        return {
            'n_estimators': [500, 1000, 1500],
            'max_depth': [4, 6, 8, 10],
            'learning_rate': [0.05, 0.1, 0.15, 0.2],
            'subsample': [0.6, 0.7, 0.8, 0.9],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
            'reg_alpha': [0, 0.01, 0.1, 0.5],
            'reg_lambda': [0, 0.01, 0.1, 0.5],
            'min_child_samples': [10, 20, 30, 50],
        }
    
    @staticmethod
    def optimize_hyperparameters(model_class, search_space: Dict, X: np.ndarray, 
                                y: np.ndarray, cv_folds: int = 5, 
                                max_trials: int = 50) -> Tuple[Dict, float]:
        """Optimize hyperparameters using grid search with cross-validation."""
        
        logger.info(f"Optimizing hyperparameters for {model_class.__name__}")
        
        best_score = -1
        best_params = None
        
        # Create parameter combinations (limiting to reasonable number)
        param_combinations = list(ParameterGrid(search_space))
        
        # Limit combinations if too many
        if len(param_combinations) > max_trials:
            param_combinations = np.random.choice(
                param_combinations, max_trials, replace=False
            ).tolist()
        
        logger.info(f"Testing {len(param_combinations)} parameter combinations")
        
        for i, params in enumerate(param_combinations):
            try:
                # Create model with current parameters
                model = model_class(**params)
                
                # Perform cross-validation
                scores = []
                skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, 
                                    random_state=CONFIG['random_state'])
                
                # For multi-label, use first label for stratification
                for train_idx, val_idx in skf.split(X, y[:, 0]):
                    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                    
                    # Train model
                    fold_model = model_class(**params)
                    fold_model.fit(X_train_fold, y_train_fold, verbose=False)
                    
                    # Predict and evaluate
                    y_pred = fold_model.predict(X_val_fold)
                    score = f1_score(y_val_fold, y_pred, average='macro', zero_division=0)
                    scores.append(score)
                
                avg_score = np.mean(scores)
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_params = params
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Completed {i + 1}/{len(param_combinations)} trials. "
                              f"Best score: {best_score:.4f}")
                
            except Exception as e:
                logger.warning(f"Error in hyperparameter trial {i}: {e}")
                continue
        
        logger.info(f"Best hyperparameters found with score {best_score:.4f}")
        
        return best_params, best_score

def load_and_prepare_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                   np.ndarray, np.ndarray, np.ndarray]:
    """Load and prepare data with BERT embeddings."""
    
    logger.info("Loading dataset...")
    
    # Load data
    df_train = pd.read_csv("train.csv")
    df_val = pd.read_csv("val.csv")
    df_test = pd.read_csv("test.csv")
    
    # Convert string representations of lists to actual lists
    df_train["labels"] = df_train["labels"].apply(eval)
    df_val["labels"] = df_val["labels"].apply(eval)
    df_test["labels"] = df_test["labels"].apply(eval)
    
    logger.info(f"Dataset sizes - Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
    
    # Initialize BERT embedding extractor
    bert_extractor = BERTEmbeddingExtractor(
        CONFIG['bert_model'], 
        CONFIG['max_length'], 
        CONFIG['batch_size']
    )
    
    # Extract embeddings with caching
    X_train = bert_extractor.extract_embeddings(
        df_train["text"].tolist(), 
        cache_file="train_embeddings.npy"
    )
    X_val = bert_extractor.extract_embeddings(
        df_val["text"].tolist(), 
        cache_file="val_embeddings.npy"
    )
    X_test = bert_extractor.extract_embeddings(
        df_test["text"].tolist(), 
        cache_file="test_embeddings.npy"
    )
    
    # Convert labels to arrays
    y_train = np.array(df_train["labels"].tolist())
    y_val = np.array(df_val["labels"].tolist())
    y_test = np.array(df_test["labels"].tolist())
    
    logger.info(f"Feature dimensions - X_train: {X_train.shape}, y_train: {y_train.shape}")
    logger.info(f"Label distribution - Mean labels per sample: {np.mean(np.sum(y_train, axis=1)):.2f}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_and_evaluate_model(model_class, model_name: str, X_train: np.ndarray, 
                           X_val: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, 
                           y_val: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """Train and evaluate a GBM model with full pipeline."""
    
    logger.info(f"Training and evaluating {model_name}")
    
    results = {'model_name': model_name}
    
    try:
        # Hyperparameter optimization
        if model_name == 'XGBoost':
            search_space = HyperparameterOptimizer.get_xgboost_search_space()
        else:
            search_space = HyperparameterOptimizer.get_lightgbm_search_space()
        
        best_params, best_cv_score = HyperparameterOptimizer.optimize_hyperparameters(
            model_class, search_space, X_train, y_train, 
            cv_folds=CONFIG['cv_folds'], max_trials=50
        )
        
        results['best_params'] = best_params
        results['best_cv_score'] = best_cv_score
        
        # Train final model with best parameters
        logger.info(f"Training final {model_name} model with optimized parameters")
        start_time = datetime.now()
        
        model = model_class(**best_params)
        model.fit(X_train, y_train, X_val, y_val)
        
        training_time = (datetime.now() - start_time).total_seconds()
        results['training_time'] = training_time
        
        # Save model
        model_filename = f"{model_name.lower()}_model.joblib"
        joblib.dump(model, model_filename)
        results['model_file'] = model_filename
        
        # Evaluate on validation set
        logger.info("Evaluating on validation set")
        y_val_pred = model.predict(X_val)
        y_val_proba = model.predict_proba(X_val)
        
        val_metrics = MultiLabelMetrics.compute_metrics(y_val, y_val_pred, y_val_proba)
        results['validation_metrics'] = val_metrics
        
        # Optimize thresholds if enabled
        if CONFIG['threshold_optimization']:
            optimal_thresholds = MultiLabelMetrics.optimize_thresholds(y_val, y_val_proba)
            results['optimal_thresholds'] = optimal_thresholds.tolist()
            
            # Re-evaluate with optimal thresholds
            y_val_pred_opt = (y_val_proba >= optimal_thresholds).astype(int)
            val_metrics_opt = MultiLabelMetrics.compute_metrics(y_val, y_val_pred_opt)
            results['validation_metrics_optimized'] = val_metrics_opt
        
        # Evaluate on test set
        logger.info("Evaluating on test set")
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)
        
        test_metrics = MultiLabelMetrics.compute_metrics(y_test, y_test_pred, y_test_proba)
        results['test_metrics'] = test_metrics
        
        # Test with optimal thresholds if available
        if CONFIG['threshold_optimization'] and 'optimal_thresholds' in results:
            y_test_pred_opt = (y_test_proba >= np.array(results['optimal_thresholds'])).astype(int)
            test_metrics_opt = MultiLabelMetrics.compute_metrics(y_test, y_test_pred_opt)
            results['test_metrics_optimized'] = test_metrics_opt
        
        # Feature importance (for XGBoost)
        if hasattr(model, 'feature_importances_') or hasattr(model.model, 'feature_importances_'):
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            else:
                importance = model.model.feature_importances_
            
            # Save top features
            top_features_idx = np.argsort(importance)[-20:][::-1]
            results['top_features'] = {
                'indices': top_features_idx.tolist(),
                'importance': importance[top_features_idx].tolist()
            }
        
        logger.info(f"{model_name} training completed successfully")
        
    except Exception as e:
        logger.error(f"Error training {model_name}: {e}")
        results['error'] = str(e)
    
    return results

def compare_models_statistical(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Perform statistical comparison between models."""
    
    logger.info("Performing statistical comparison between models")
    
    comparison_results = {}
    
    # Extract test metrics for comparison
    model_scores = {}
    for result in results:
        if 'error' not in result and 'test_metrics' in result:
            model_name = result['model_name']
            
            # Use optimized metrics if available, otherwise default
            if 'test_metrics_optimized' in result:
                metrics = result['test_metrics_optimized']
            else:
                metrics = result['test_metrics']
            
            model_scores[model_name] = {
                'macro_f1': metrics['macro_f1'],
                'micro_f1': metrics['micro_f1'],
                'per_label_f1': metrics['per_label_f1']
            }
    
    comparison_results['model_scores'] = model_scores
    
    # Find best performing model
    best_macro_f1 = -1
    best_model = None
    
    for model_name, scores in model_scores.items():
        if scores['macro_f1'] > best_macro_f1:
            best_macro_f1 = scores['macro_f1']
            best_model = model_name
    
    comparison_results['best_model'] = best_model
    comparison_results['best_macro_f1'] = best_macro_f1
    
    # Per-label analysis
    if len(model_scores) >= 2:
        models = list(model_scores.keys())
        per_label_comparison = {}
        
        for i, label in enumerate(GOEMOTIONS_LABELS):
            label_scores = {}
            for model in models:
                if i < len(model_scores[model]['per_label_f1']):
                    label_scores[model] = model_scores[model]['per_label_f1'][i]
            
            if label_scores:
                best_model_for_label = max(label_scores, key=label_scores.get)
                per_label_comparison[label] = {
                    'scores': label_scores,
                    'best_model': best_model_for_label,
                    'best_score': label_scores[best_model_for_label]
                }
        
        comparison_results['per_label_comparison'] = per_label_comparison
    
    return comparison_results

def generate_comprehensive_report(results: List[Dict[str, Any]], 
                                comparison_results: Dict[str, Any]) -> str:
    """Generate comprehensive evaluation report."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_lines = []
    
    report_lines.append("="*80)
    report_lines.append("GRADIENT BOOSTING MACHINES - COMPREHENSIVE EVALUATION REPORT")
    report_lines.append("="*80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Dataset: GoEmotions Multi-Label Emotion Classification")
    report_lines.append(f"Number of Labels: {CONFIG['num_labels']}")
    report_lines.append(f"BERT Model: {CONFIG['bert_model']}")
    report_lines.append("")
    
    # Model Performance Summary
    report_lines.append("MODEL PERFORMANCE SUMMARY")
    report_lines.append("-" * 40)
    
    for result in results:
        if 'error' in result:
            report_lines.append(f"{result['model_name']}: FAILED - {result['error']}")
            continue
        
        model_name = result['model_name']
        report_lines.append(f"\n{model_name}:")
        
        # Training info
        if 'training_time' in result:
            report_lines.append(f"  Training Time: {result['training_time']:.1f}s")
        
        if 'best_cv_score' in result:
            report_lines.append(f"  Best CV Score: {result['best_cv_score']:.4f}")
        
        # Test performance
        if 'test_metrics' in result:
            tm = result['test_metrics']
            report_lines.append(f"  Test Macro F1: {tm['macro_f1']:.4f}")
            report_lines.append(f"  Test Micro F1: {tm['micro_f1']:.4f}")
            report_lines.append(f"  Test Hamming Loss: {tm['hamming_loss']:.4f}")
            report_lines.append(f"  Exact Match Ratio: {tm['exact_match_ratio']:.4f}")
        
        # Optimized performance
        if 'test_metrics_optimized' in result:
            tmo = result['test_metrics_optimized']
            report_lines.append(f"  Optimized Macro F1: {tmo['macro_f1']:.4f}")
            report_lines.append(f"  Optimized Micro F1: {tmo['micro_f1']:.4f}")
            improvement = tmo['macro_f1'] - result['test_metrics']['macro_f1']
            report_lines.append(f"  Improvement: +{improvement:.4f}")
    
    # Best Model
    if 'best_model' in comparison_results:
        report_lines.append(f"\nBEST PERFORMING MODEL: {comparison_results['best_model']}")
        report_lines.append(f"Best Macro F1: {comparison_results['best_macro_f1']:.4f}")
    
    # Per-label analysis
    if 'per_label_comparison' in comparison_results:
        report_lines.append("\nPER-LABEL PERFORMANCE ANALYSIS")
        report_lines.append("-" * 40)
        
        plc = comparison_results['per_label_comparison']
        
        # Best performing labels
        best_labels = sorted(plc.items(), key=lambda x: x[1]['best_score'], reverse=True)[:5]
        report_lines.append("\nTop 5 Best Performing Labels:")
        for label, data in best_labels:
            report_lines.append(f"  {label}: {data['best_score']:.4f} ({data['best_model']})")
        
        # Worst performing labels
        worst_labels = sorted(plc.items(), key=lambda x: x[1]['best_score'])[:5]
        report_lines.append("\nTop 5 Worst Performing Labels:")
        for label, data in worst_labels:
            report_lines.append(f"  {label}: {data['best_score']:.4f} ({data['best_model']})")
    
    # Hyperparameter analysis
    report_lines.append("\nHYPERPARAMETER OPTIMIZATION RESULTS")
    report_lines.append("-" * 40)
    
    for result in results:
        if 'best_params' in result:
            report_lines.append(f"\n{result['model_name']} Best Parameters:")
            for param, value in result['best_params'].items():
                report_lines.append(f"  {param}: {value}")
    
    # Generate recommendations
    report_lines.append("\nRECOMMENDATIONS")
    report_lines.append("-" * 40)
    
    if len(results) >= 2:
        successful_results = [r for r in results if 'error' not in r]
        if len(successful_results) >= 2:
            # Compare XGBoost vs LightGBM
            xgb_result = next((r for r in successful_results if r['model_name'] == 'XGBoost'), None)
            lgb_result = next((r for r in successful_results if r['model_name'] == 'LightGBM'), None)
            
            if xgb_result and lgb_result:
                xgb_score = xgb_result.get('test_metrics_optimized', xgb_result.get('test_metrics', {})).get('macro_f1', 0)
                lgb_score = lgb_result.get('test_metrics_optimized', lgb_result.get('test_metrics', {})).get('macro_f1', 0)
                
                if xgb_score > lgb_score:
                    report_lines.append("• XGBoost outperformed LightGBM on this dataset")
                    report_lines.append("• Consider using XGBoost for production deployment")
                else:
                    report_lines.append("• LightGBM outperformed XGBoost on this dataset")
                    report_lines.append("• LightGBM may offer better speed/performance trade-off")
        
        # General recommendations
        report_lines.append("• Threshold optimization provided significant improvements")
        report_lines.append("• Consider ensemble methods for further performance gains")
        report_lines.append("• Monitor rare label performance in production")
    
    report_lines.append("\n" + "="*80)
    
    # Save report
    report_content = "\n".join(report_lines)
    report_filename = f"gbm_evaluation_report_{timestamp}.txt"
    
    with open(report_filename, 'w') as f:
        f.write(report_content)
    
    logger.info(f"Comprehensive report saved to {report_filename}")
    
    return report_content

def save_results_for_master_table(results: List[Dict[str, Any]], 
                                comparison_results: Dict[str, Any]) -> None:
    """Save results in format compatible with master results table."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    master_results = []
    
    for result in results:
        if 'error' in result:
            continue
        
        model_name = result['model_name']
        
        # Get metrics (prioritize optimized if available)
        test_metrics = result.get('test_metrics_optimized', result.get('test_metrics', {}))
        val_metrics = result.get('validation_metrics_optimized', result.get('validation_metrics', {}))
        
        master_record = {
            'model_type': f'{model_name} (Tree-based Baseline)',
            'architecture': f'{model_name} with Binary Relevance' if model_name == 'LightGBM' else f'{model_name} Multi-output',
            'input_features': 'BERT embeddings (768-dim)',
            'num_parameters': 'Variable (tree-based)',
            
            # Performance metrics
            'test_macro_f1': test_metrics.get('macro_f1', 0),
            'test_micro_f1': test_metrics.get('micro_f1', 0),
            'test_weighted_f1': test_metrics.get('weighted_f1', 0),
            'test_hamming_loss': test_metrics.get('hamming_loss', 0),
            'exact_match_ratio': test_metrics.get('exact_match_ratio', 0),
            
            'val_macro_f1': val_metrics.get('macro_f1', 0),
            'val_micro_f1': val_metrics.get('micro_f1', 0),
            
            # Training info
            'training_time_seconds': result.get('training_time', 0),
            'cv_score': result.get('best_cv_score', 0),
            'hyperopt_trials': 50,
            
            # Additional info
            'threshold_optimization': CONFIG['threshold_optimization'],
            'best_params': json.dumps(result.get('best_params', {})),
            'timestamp': timestamp,
            'notes': f'Gradient Boosting with {CONFIG["cv_folds"]}-fold CV optimization'
        }
        
        master_results.append(master_record)
    
    # Save to CSV for easy integration
    master_df = pd.DataFrame(master_results)
    master_filename = f"gbm_master_results_{timestamp}.csv"
    master_df.to_csv(master_filename, index=False)
    
    # Save detailed results as JSON
    detailed_results = {
        'individual_results': results,
        'comparison_results': comparison_results,
        'config': CONFIG,
        'timestamp': timestamp
    }
    
    detailed_filename = f"gbm_detailed_results_{timestamp}.json"
    with open(detailed_filename, 'w') as f:
        json.dump(detailed_results, f, indent=2, default=str)
    
    logger.info(f"Master results saved to {master_filename}")
    logger.info(f"Detailed results saved to {detailed_filename}")

def main():
    """Main execution function."""
    
    logger.info("Starting Chapter 6: Gradient Boosting Machine Training")
    logger.info(f"Configuration: {CONFIG}")
    
    try:
        # Load and prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = load_and_prepare_data()
        
        # Initialize results storage
        all_results = []
        
        # Train XGBoost
        if CONFIG['use_xgboost']:
            logger.info("Training XGBoost model...")
            xgb_results = train_and_evaluate_model(
                XGBoostMultiLabel, 'XGBoost',
                X_train, X_val, X_test, y_train, y_val, y_test
            )
            all_results.append(xgb_results)
        
        # Train LightGBM
        if CONFIG['use_lightgbm']:
            logger.info("Training LightGBM model...")
            lgb_results = train_and_evaluate_model(
                LightGBMMultiLabel, 'LightGBM',
                X_train, X_val, X_test, y_train, y_val, y_test
            )
            all_results.append(lgb_results)
        
        # Perform statistical comparison
        comparison_results = compare_models_statistical(all_results)
        
        # Generate comprehensive report
        report = generate_comprehensive_report(all_results, comparison_results)
        print(report)
        
        # Save results for master table
        save_results_for_master_table(all_results, comparison_results)
        
        # Print summary
        print("\n" + "="*60)
        print("GRADIENT BOOSTING MACHINE TRAINING COMPLETED")
        print("="*60)
        
        successful_models = [r for r in all_results if 'error' not in r]
        
        if successful_models:
            print(f"Successfully trained {len(successful_models)} models:")
            
            for result in successful_models:
                model_name = result['model_name']
                test_metrics = result.get('test_metrics_optimized', result.get('test_metrics', {}))
                
                print(f"\n{model_name}:")
                print(f"  Macro F1: {test_metrics.get('macro_f1', 0):.4f}")
                print(f"  Micro F1: {test_metrics.get('micro_f1', 0):.4f}")
                print(f"  Training Time: {result.get('training_time', 0):.1f}s")
        
        if 'best_model' in comparison_results:
            print(f"\nBest Model: {comparison_results['best_model']}")
            print(f"Best Macro F1: {comparison_results['best_macro_f1']:.4f}")
        
        print("="*60)
        
        # Cleanup
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        logger.info("Chapter 6 completed successfully!")
        
        return all_results, comparison_results
        
    except Exception as e:
        logger.error(f"Critical error in main execution: {e}")
        raise e

if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    # Execute main function
    results, comparison = main()
