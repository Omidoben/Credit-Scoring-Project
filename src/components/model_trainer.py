import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any
import joblib
import json
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.model_selection import cross_val_score

from src.utils.logger import get_logger
from src.utils.exception import CreditScoringException
from src.utils.config_loader import ConfigLoader

logger = get_logger(__name__)


@dataclass
class ModelTrainerConfig:
    """Configuration for model training"""
    model_dir: str
    cv_folds: int
    random_state: int
    metric: str
    models_to_train: list
    
    def __post_init__(self):
        """Ensure model directory exists"""
        os.makedirs(self.model_dir, exist_ok=True)


class ModelTrainer:
    """
    Train multiple ML models, evaluate performance, and select the best one
    Uses cross-validation for model selection
    """
    
    def __init__(self, config: Optional[ConfigLoader] = None):
        """
        Initialize ModelTrainer
        
        Args:
            config (ConfigLoader, optional): Configuration loader instance
        """
        if config is None:
            config = ConfigLoader()
        
        self.config = config
        model_config = config.get_model_config()
        training_config = config.get_training_config()
        
        self.trainer_config = ModelTrainerConfig(
            model_dir=model_config.get("model_dir", "models/"),
            cv_folds=model_config.get("cv_folds", 5),
            random_state=model_config.get("random_state", 42),
            metric=training_config.get("metric", "f1"),
            models_to_train=training_config.get("models_to_train", 
                                               ["logistic_regression", "random_forest", "xgboost", "lightgbm"])
        )
        
        self.models = {}
        self.results = {}
        
        logger.info("ModelTrainer initialized")
    
    def get_model_instance(self, model_name: str) -> Any:
        """
        Get model instance based on name
        
        Args:
            model_name (str): Name of the model
        
        Returns:
            Model instance
        """
        models_dict = {
            "logistic_regression": LogisticRegression(
                random_state=self.trainer_config.random_state,
                max_iter=1000,
                class_weight="balanced"
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.trainer_config.random_state,
                class_weight="balanced",
                n_jobs=-1
            ),
            "xgboost": XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.trainer_config.random_state,
                scale_pos_weight=3,
                eval_metric="logloss"
            ),
            "lightgbm": LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.trainer_config.random_state,
                class_weight="balanced",
                verbose=-1
            )
        }
        
        if model_name not in models_dict:
            raise ValueError(f"Model '{model_name}' not recognized. Available: {list(models_dict.keys())}")
        
        return models_dict[model_name]
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict]:
        """
        Train and evaluate all configured models using CV for selection
        
        Process:
        1. Train each model and evaluate with CV
        2. Select best model based on CV scores
        3. Evaluate all models on test set
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training targets
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test targets
        
        Returns:
            Dict: Results for all models
        """
        try:
            logger.info("Training all models with cross-validation")
            
            # Train all models and get CV scores
            for model_name in self.trainer_config.models_to_train:
                try:
                    logger.info(f"Training {model_name}...")
                    
                    model = self.get_model_instance(model_name)
                    
                    # Cross-validation
                    cv_scores = cross_val_score(
                        model, X_train, y_train,
                        cv=self.trainer_config.cv_folds,
                        scoring=self.trainer_config.metric
                    )
                    
                    logger.info(f"  CV {self.trainer_config.metric}: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
                    
                    # Train model on full training set
                    model.fit(X_train, y_train)
                    self.models[model_name] = model
                    
                    # Store CV results
                    self.results[model_name] = {
                        "cv_mean": cv_scores.mean(),
                        "cv_std": cv_scores.std()
                    }
                    
                except Exception as e:
                    logger.error(f"Error training {model_name}: {e}")
                    continue
            
            logger.info(f"Trained {len(self.models)} models successfully")
            
            # Select best model based on CV scores
            if not self.results:
                raise ValueError("No models trained successfully")
            
            best_model_name = max(self.results, key=lambda x: self.results[x]["cv_mean"])
            logger.info(f"Best model selected: {best_model_name} (CV {self.trainer_config.metric}: {self.results[best_model_name]['cv_mean']:.4f})")
            
            # Evaluate ALL models on test set
            logger.info("Evaluating all models on test set...")
            
            for model_name in self.models.keys():
                model = self.models[model_name]
                
                # Predictions
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                
                # Probabilities (for ROC-AUC)
                y_train_proba = model.predict_proba(X_train)[:, 1]
                y_test_proba = model.predict_proba(X_test)[:, 1]
                
                #  metrics
                metrics = {
                    "cv_mean": self.results[model_name]["cv_mean"],
                    "cv_std": self.results[model_name]["cv_std"],
                    "train_accuracy": accuracy_score(y_train, y_train_pred),
                    "test_accuracy": accuracy_score(y_test, y_test_pred),
                    "train_precision": precision_score(y_train, y_train_pred, zero_division=0),
                    "test_precision": precision_score(y_test, y_test_pred, zero_division=0),
                    "train_recall": recall_score(y_train, y_train_pred, zero_division=0),
                    "test_recall": recall_score(y_test, y_test_pred, zero_division=0),
                    "train_f1": f1_score(y_train, y_train_pred, zero_division=0),
                    "test_f1": f1_score(y_test, y_test_pred, zero_division=0),
                    "train_roc_auc": roc_auc_score(y_train, y_train_proba),
                    "test_roc_auc": roc_auc_score(y_test, y_test_proba)
                }
                
                # Confusion matrix
                cm = confusion_matrix(y_test, y_test_pred)
                metrics["confusion_matrix"] = cm.tolist()
                
                # Update results with full metrics
                self.results[model_name] = metrics
                
                logger.info(f"  {model_name} - Test Accuracy: {metrics['test_accuracy']:.4f}, Test F1: {metrics['test_f1']:.4f}, Test ROC-AUC: {metrics['test_roc_auc']:.4f}")
            
            logger.info("All models evaluated")
            
            return self.results
        
        except Exception as e:
            raise CreditScoringException(e, sys)
    
    def get_best_model(self) -> Tuple[str, Any, Dict]:
        """
        Get the best model selected during training (based on CV scores)
        
        Returns:
            Tuple[str, Any, Dict]: (model_name, model, metrics)
        """
        try:
            if not self.results:
                raise ValueError("No models have been trained yet")
            
            # Best model was selected based on CV during training
            best_model_name = max(self.results, key=lambda x: self.results[x]["cv_mean"])
            best_model = self.models[best_model_name]
            best_metrics = self.results[best_model_name]
            
            logger.info(f"Best model: {best_model_name}")
            logger.info(f"  CV {self.trainer_config.metric}: {best_metrics['cv_mean']:.4f} (±{best_metrics['cv_std']:.4f})")
            logger.info(f"  Test {self.trainer_config.metric}: {best_metrics[f'test_{self.trainer_config.metric}']:.4f}")
            
            return best_model_name, best_model, best_metrics
        
        except Exception as e:
            raise CreditScoringException(e, sys)
    
    def save_model(self, model: Any, model_name: str, metrics: Dict, version: str = "1.0") -> str:
        """
        Save model with metadata
        
        Args:
            model: Trained model
            model_name (str): Name of the model
            metrics (Dict): Performance metrics
            version (str): Model version
        
        Returns:
            str: Path to saved model
        """
        try:
            logger.info(f"Saving model: {model_name}")
            
            # Save model
            model_filename = f"{model_name}_v{version}.pkl"
            model_path = os.path.join(self.trainer_config.model_dir, model_filename)
            joblib.dump(model, model_path)
            
            # Save metadata
            metadata = {
                "model_name": model_name,
                "version": version,
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics,
                "config": {
                    "random_state": self.trainer_config.random_state,
                    "cv_folds": self.trainer_config.cv_folds,
                    "metric": self.trainer_config.metric
                }
            }
            
            metadata_path = os.path.join(self.trainer_config.model_dir, f"{model_name}_v{version}_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)
            
            logger.info(f"Model saved to: {model_path}")
            logger.info(f"Metadata saved to: {metadata_path}")
            
            return model_path
        
        except Exception as e:
            raise CreditScoringException(e, sys)
    
    def save_best_model(self) -> str:
        """
        Save the best performing model
        
        Returns:
            str: Path to saved best model
        """
        try:
            best_model_name, best_model, best_metrics = self.get_best_model()
            
            # Save with version
            model_path = self.save_model(best_model, best_model_name, best_metrics)
            
            # Also save as "best_model.pkl" for easy loading
            best_model_path = os.path.join(self.trainer_config.model_dir, "best_model.pkl")
            joblib.dump(best_model, best_model_path)
            
            # Save best model metadata
            metadata = {
                "model_name": best_model_name,
                "timestamp": datetime.now().isoformat(),
                "metrics": best_metrics
            }
            
            best_metadata_path = os.path.join(self.trainer_config.model_dir, "best_model_metadata.json")
            with open(best_metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)
            
            logger.info(f"Best model also saved as: {best_model_path}")
            
            return best_model_path
        
        except Exception as e:
            raise CreditScoringException(e, sys)
    
    def save_comparison_report(self) -> str:
        """
        Save model comparison report as CSV
        
        Returns:
            str: Path to comparison report
        """
        try:
            logger.info("Saving model comparison report...")
            
            # Create comparison dataframe
            comparison_data = []
            
            for model_name, metrics in self.results.items():
                row = {
                    "model": model_name,
                    "cv_mean": metrics["cv_mean"],
                    "cv_std": metrics["cv_std"],
                    "train_accuracy": metrics["train_accuracy"],
                    "test_accuracy": metrics["test_accuracy"],
                    "train_f1": metrics["train_f1"],
                    "test_f1": metrics["test_f1"],
                    "train_roc_auc": metrics["train_roc_auc"],
                    "test_roc_auc": metrics["test_roc_auc"]
                }
                comparison_data.append(row)
            
            df_comparison = pd.DataFrame(comparison_data)
            
            # Sort by CV mean (how models were selected)
            df_comparison = df_comparison.sort_values("cv_mean", ascending=False)
            
            # Save to CSV
            report_path = os.path.join(self.trainer_config.model_dir, "model_comparison.csv")
            df_comparison.to_csv(report_path, index=False)
            
            logger.info(f"Comparison report saved to: {report_path}")
            
            return report_path
        
        except Exception as e:
            raise CreditScoringException(e, sys)
    
    def print_results_summary(self):
        """Print formatted results summary"""
        print("\n" + "="*70)
        print("MODEL TRAINING RESULTS")
        print("="*70)
        
        # Sort models by CV score (how they were selected)
        sorted_models = sorted(
            self.results.items(), 
            key=lambda x: x[1]["cv_mean"], 
            reverse=True
        )
        
        for model_name, metrics in sorted_models:
            print(f"\n{model_name.upper()}")
            print("-" * 70)
            print(f"  CV {self.trainer_config.metric.upper()}:     {metrics['cv_mean']:.4f} (±{metrics['cv_std']:.4f}) ← Selection Metric")
            print(f"  Train Accuracy: {metrics['train_accuracy']:.4f}")
            print(f"  Test Accuracy:  {metrics['test_accuracy']:.4f}")
            print(f"  Train F1:       {metrics['train_f1']:.4f}")
            print(f"  Test F1:        {metrics['test_f1']:.4f}")
            print(f"  Train ROC-AUC:  {metrics['train_roc_auc']:.4f}")
            print(f"  Test ROC-AUC:   {metrics['test_roc_auc']:.4f}")
        
        print("\n" + "="*70)
        best_model_name, _, best_metrics = self.get_best_model()
        print(f"BEST MODEL: {best_model_name.upper()}")
        print(f"  Selected based on CV {self.trainer_config.metric.upper()}: {best_metrics['cv_mean']:.4f}")
        print(f"  Final Test {self.trainer_config.metric.upper()}: {best_metrics[f'test_{self.trainer_config.metric}']:.4f}")
        print("="*70)


# Testing
if __name__ == "__main__":
    try:
        from src.components.data_ingestion import DataIngestion
        from src.components.data_transformation import DataTransformation
        
        # Load and transform data
        logger.info("Loading and transforming data...")
        data_ingestion = DataIngestion()
        demographics_df, financial_history_df, transactions_df = data_ingestion.load_all_data()
        
        data_transformation = DataTransformation()
        X_train, X_test, y_train, y_test, _ = data_transformation.transform_all(
            demographics_df,
            financial_history_df,
            transactions_df
        )
        
        # Train models
        model_trainer = ModelTrainer()
        results = model_trainer.train_all_models(X_train, y_train, X_test, y_test)
        
        # Print results
        model_trainer.print_results_summary()
        
        # Save best model
        best_model_path = model_trainer.save_best_model()
        
        # Save comparison report
        report_path = model_trainer.save_comparison_report()
        
        print(f"\n✓ Model training complete!")
        print(f"  Best model saved to: {best_model_path}")
        print(f"  Comparison report: {report_path}")
        
    except Exception as e:
        logger.error(f"Error in model training test: {e}")
        raise