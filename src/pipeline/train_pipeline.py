import sys
from typing import Optional, Tuple
import numpy as np

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.utils.logger import get_logger
from src.utils.exception import CreditScoringException
from src.utils.config_loader import ConfigLoader

logger = get_logger(__name__)


class TrainPipeline:
    """
    Orchestrates the complete training pipeline:
    1. Data Ingestion
    2. Data Transformation (Feature Engineering)
    3. Model Training (CV-based selection)
    4. Model Evaluation
    5. Model Saving
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize training pipeline
        
        Args:
            config_path (str, optional): Path to config file
        """
        # Load configuration
        if config_path:
            self.config = ConfigLoader(config_path)
        else:
            self.config = ConfigLoader()
        
        # Initialize components
        self.data_ingestion = DataIngestion(self.config)
        self.data_transformation = DataTransformation(self.config)
        self.model_trainer = ModelTrainer(self.config)
        
        logger.info("TrainPipeline initialized")
    
    def run(self) -> Tuple[str, dict]:
        """
        Run the complete training pipeline
        
        Returns:
            Tuple[str, dict]: (best_model_path, results_summary)
        """
        try:
            logger.info("="*70)
            logger.info("starting credit scoring training pipeline")
            logger.info("="*70)
            
            # Step 1: Data Ingestion
            logger.info("step 1: data ingestion")
            
            demographics_df, financial_history_df, transactions_df = (
                self.data_ingestion.load_all_data()
            )
            
            logger.info("Data ingestion complete")
            logger.info(f"  Demographics: {demographics_df.shape}")
            logger.info(f"  Financial History: {financial_history_df.shape}")
            logger.info(f"  Transactions: {transactions_df.shape}")
            
            # Step 2: Data Transformation
            logger.info("step 2: data transformation")
            
            X_train, X_test, y_train, y_test, features_df = (
                self.data_transformation.transform_all(
                    demographics_df,
                    financial_history_df,
                    transactions_df
                )
            )
            
            logger.info("Data transformation complete")
            logger.info(f"  Training set: {X_train.shape}")
            logger.info(f"  Test set: {X_test.shape}")
            logger.info(f"  Features: {len(self.data_transformation.feature_columns)}")
            
            # Step 3: Model Training
            logger.info("step 3: model training")
            
            results = self.model_trainer.train_all_models(
                X_train, y_train,
                X_test, y_test
            )
            
            logger.info("Model training complete")
            logger.info(f"  Models trained: {len(results)}")
            
            # Step 4: Model Selection & Saving
            logger.info("step 4: model selection & saving")
            
            # Get best model
            best_model_name, best_model, best_metrics = self.model_trainer.get_best_model()
            
            logger.info(f"Best model selected: {best_model_name} (based on CV {self.model_trainer.trainer_config.metric})")
            logger.info(f"  CV {self.model_trainer.trainer_config.metric}: {best_metrics['cv_mean']:.4f} (±{best_metrics['cv_std']:.4f})")
            logger.info(f"  Test Accuracy: {best_metrics['test_accuracy']:.4f}")
            logger.info(f"  Test F1 Score: {best_metrics['test_f1']:.4f}")
            logger.info(f"  Test ROC-AUC: {best_metrics['test_roc_auc']:.4f}")
            
            # Save best model
            best_model_path = self.model_trainer.save_best_model()
            
            # Save comparison report
            report_path = self.model_trainer.save_comparison_report()
            
            logger.info("Model artifacts saved")
            logger.info(f"  Best model: {best_model_path}")
            logger.info(f"  Comparison report: {report_path}")
            logger.info(f"  Scaler: {self.data_transformation.transformation_config.scaler_path}")
            logger.info(f"  Feature columns: {self.data_transformation.transformation_config.feature_columns_path}")
            
            # Step 5: Pipeline Summary
            logger.info("="*70)
            logger.info("training pipeline completed successfully")
            logger.info("="*70)
            
            # Create summary
            summary = {
                "status": "success",
                "best_model": best_model_name,
                "best_model_path": best_model_path,
                "metrics": {
                    "cv_mean": best_metrics["cv_mean"],
                    "cv_std": best_metrics["cv_std"],
                    "test_accuracy": best_metrics["test_accuracy"],
                    "test_precision": best_metrics["test_precision"],
                    "test_recall": best_metrics["test_recall"],
                    "test_f1": best_metrics["test_f1"],
                    "test_roc_auc": best_metrics["test_roc_auc"]
                },
                "data_info": {
                    "train_samples": X_train.shape[0],
                    "test_samples": X_test.shape[0],
                    "num_features": len(self.data_transformation.feature_columns),
                    "target_distribution": {
                        "train_repaid": int((y_train == 1).sum()),
                        "train_default": int((y_train == 0).sum()),
                        "test_repaid": int((y_test == 1).sum()),
                        "test_default": int((y_test == 0).sum())
                    }
                },
                "artifacts": {
                    "model": best_model_path,
                    "scaler": self.data_transformation.transformation_config.scaler_path,
                    "feature_columns": self.data_transformation.transformation_config.feature_columns_path,
                    "comparison_report": report_path
                }
            }
            
            # Print detailed summary
            self._print_summary(summary)
            
            return best_model_path, summary
        
        except Exception as e:
            logger.error("Training Pipeline Failed")
            raise CreditScoringException(e, sys)
    
    def _print_summary(self, summary: dict):
        """
        Print formatted pipeline summary
        
        Args:
            summary (dict): Pipeline execution summary
        """
        print("="*70)
        print("training pipeline summary")
        print("="*70)
        
        print(f"\nbest model: {summary['best_model']} (selected by cv)")
        
        metrics = summary["metrics"]
        print(f"\nperformance metrics:")
        print(f"  cv {self.model_trainer.trainer_config.metric}:      {metrics['cv_mean']:.4f} (±{metrics['cv_std']:.4f}) <- selection metric")
        print(f"  test accuracy:  {metrics['test_accuracy']:.4f}")
        print(f"  test precision: {metrics['test_precision']:.4f}")
        print(f"  test recall:    {metrics['test_recall']:.4f}")
        print(f"  test f1 score:  {metrics['test_f1']:.4f}")
        print(f"  test roc-auc:   {metrics['test_roc_auc']:.4f}")
        
        print(f"\ndataset information:")
        data_info = summary["data_info"]
        print(f"  training samples:   {data_info['train_samples']}")
        print(f"  test samples:       {data_info['test_samples']}")
        print(f"  number of features: {data_info['num_features']}")
        
        print(f"\ntarget distribution (train):")
        print(f"  repaid (1):  {data_info['target_distribution']['train_repaid']}")
        print(f"  default (0): {data_info['target_distribution']['train_default']}")
        
        print(f"\ntarget distribution (test):")
        print(f"  repaid (1):  {data_info['target_distribution']['test_repaid']}")
        print(f"  default (0): {data_info['target_distribution']['test_default']}")
        
        print(f"\nsaved artifacts:")
        artifacts = summary["artifacts"]
        print(f"  model:           {artifacts['model']}")
        print(f"  scaler:          {artifacts['scaler']}")
        print(f"  feature columns: {artifacts['feature_columns']}")
        print(f"  report:          {artifacts['comparison_report']}")
        
        print("="*70)
        print("pipeline execution completed successfully")
        print("="*70)
    
    def run_quick_validation(self) -> bool:
        """
        Quick validation to check if all artifacts exist
        
        Returns:
            bool: True if all artifacts exist, False otherwise
        """
        try:
            import os
            
            logger.info("Running quick validation...")
            
            # Check if model exists
            model_path = os.path.join(
                self.model_trainer.trainer_config.model_dir,
                "best_model.pkl"
            )
            
            # Check if scaler exists
            scaler_path = self.data_transformation.transformation_config.scaler_path
            
            # Check if feature columns exist
            feature_columns_path = self.data_transformation.transformation_config.feature_columns_path
            
            artifacts = {
                "model": model_path,
                "scaler": scaler_path,
                "feature_columns": feature_columns_path
            }
            
            all_exist = True
            for name, path in artifacts.items():
                if os.path.exists(path):
                    logger.info(f"  ✓ {name}: Found")
                else:
                    logger.warning(f"  ✗ {name}: Not found")
                    all_exist = False
            
            if all_exist:
                logger.info("all artifacts exist")
            else:
                logger.warning("some artifacts are missing. run training pipeline.")
            
            return all_exist
        
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False


# Testing
if __name__ == "__main__":
    try:
        print("="*70)
        print("testing training pipeline")
        print("="*70)
        
        # Initialize and run pipeline
        pipeline = TrainPipeline()
        best_model_path, summary = pipeline.run()
        
        # Validate artifacts
        print("="*70)
        print("validating artifacts")
        print("="*70)
        
        validation_passed = pipeline.run_quick_validation()
        
        if validation_passed:
            print("all checks passed! pipeline is ready for deployment.")
        else:
            print("some checks failed. please review the logs.")
        
    except Exception as e:
        logger.error(f"Error in training pipeline test: {e}")
        raise