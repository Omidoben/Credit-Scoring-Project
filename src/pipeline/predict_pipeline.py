import os
import sys
import pandas as pd
import numpy as np
import joblib
import json
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

from src.data.feature_extractor import FeatureExtractor
from src.utils.logger import get_logger
from src.utils.exception import CreditScoringException
from src.utils.config_loader import ConfigLoader

logger = get_logger(__name__)


@dataclass
class PredictPipelineConfig:
    """Configuration for prediction pipeline"""
    model_path: str
    scaler_path: str
    feature_columns_path: str
    
    def __post_init__(self):
        """Validate that all artifacts exist"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scaler not found: {self.scaler_path}")
        if not os.path.exists(self.feature_columns_path):
            raise FileNotFoundError(f"Feature columns not found: {self.feature_columns_path}")


class PredictPipeline:
    """
    Prediction pipeline for credit scoring
    
    Handles two types of predictions:
    1. From raw data (demographics + financial_history + transactions)
    2. From pre-computed features
    """
    
    def __init__(self, config: Optional[ConfigLoader] = None):
        """
        Initialize prediction pipeline
        
        Args:
            config (ConfigLoader, optional): Configuration loader instance
        """
        try:
            if config is None:
                config = ConfigLoader()
            
            self.config = config
            model_config = config.get_model_config()
            model_dir = model_config.get("model_dir", "models/")
            
            # Setup paths
            self.predict_config = PredictPipelineConfig(
                model_path=os.path.join(model_dir, "best_model.pkl"),
                scaler_path=os.path.join(model_dir, "scaler.pkl"),
                feature_columns_path=os.path.join(model_dir, "feature_columns.pkl")
            )
            
            # Load artifacts
            logger.info("Loading model artifacts...")
            self.model = joblib.load(self.predict_config.model_path)
            self.scaler = joblib.load(self.predict_config.scaler_path)
            self.feature_columns = joblib.load(self.predict_config.feature_columns_path)
            
            # Initialize feature extractor
            self.feature_extractor = FeatureExtractor()
            
            logger.info("PredictPipeline initialized successfully")
            logger.info(f"  Model: {self.predict_config.model_path}")
            logger.info(f"  Features: {len(self.feature_columns)}")
        
        except Exception as e:
            raise CreditScoringException(e, sys)
    
    def predict_from_raw_data(self, applicant_data: Dict) -> Dict:
        """
        Predict from raw applicant data (full pipeline)
        
        Args:
            applicant_data (Dict): Dictionary containing:
                - demographics (dict): age, employment_type, years_employed, etc.
                - financial_history (dict): past_defaults_count, credit_score, etc.
                - transactions (list): List of transaction dictionaries
        
        Returns:
            Dict: Prediction results with probability and risk assessment
        
        Example:
            >>> applicant_data = {
            ...     "demographics": {
            ...         "age": 35,
            ...         "employment_type": "Salaried",
            ...         "years_employed": 8,
            ...         "education_level": "Secondary"
            ...     },
            ...     "financial_history": {
            ...         "past_defaults_count": 0,
            ...         "past_repaid_loans": 3,
            ...         "credit_score": 680,
            ...         "account_age_months": 48,
            ...         "overdraft_history": 2
            ...     },
            ...     "transactions": [
            ...         {"date": "2024-01-01", "transaction_type": "Income", "amount": 50000, ...},
            ...         ...
            ...     ]
            ... }
            >>> result = pipeline.predict_from_raw_data(applicant_data)
        """
        try:
            logger.info("Making prediction from raw data...")
            
            # Validate input
            self._validate_raw_input(applicant_data)
            
            # Extract transaction features
            transactions_df = pd.DataFrame(applicant_data["transactions"])
            transactions_df["date"] = pd.to_datetime(transactions_df["date"])
            
            transaction_features = self.feature_extractor.extract_features(transactions_df)
            
            # Combine all features
            all_features = {**transaction_features}
            
            # Add demographics
            demographics = applicant_data["demographics"]
            all_features["age"] = demographics.get("age")
            all_features["years_employed"] = demographics.get("years_employed")
            
            # Encode education level
            education_mapping = {"Primary": 0, "Secondary": 1, "Tertiary": 2}
            all_features["education_level_encoded"] = education_mapping.get(
                demographics.get("education_level", "Secondary"), 1
            )
            
            # One-hot encode employment type
            employment_type = demographics.get("employment_type", "Self-employed")
            all_features["employment_Salaried"] = 1 if employment_type == "Salaried" else 0
            all_features["employment_Self-employed"] = 1 if employment_type == "Self-employed" else 0
            all_features["employment_Student/Other"] = 1 if employment_type == "Student/Other" else 0
            
            # Add financial history
            financial_history = applicant_data["financial_history"]
            all_features["account_age_months"] = financial_history.get("account_age_months")
            all_features["past_defaults_count"] = financial_history.get("past_defaults_count")
            all_features["past_repaid_loans"] = financial_history.get("past_repaid_loans")
            all_features["credit_score"] = financial_history.get("credit_score")
            all_features["overdraft_history"] = financial_history.get("overdraft_history")
            
            # Make prediction
            result = self._make_prediction(all_features)
            
            logger.info(f"Prediction complete: {result['prediction_label']}")
            
            return result
        
        except Exception as e:
            raise CreditScoringException(e, sys)
    
    def predict_from_features(self, features: Dict) -> Dict:
        """
        Predict from pre-computed features
        
        Args:
            features (Dict): Dictionary of feature values
        
        Returns:
            Dict: Prediction results with probability and risk assessment
        
        Example:
            >>> features = {
            ...     "monthly_avg_income": 45000,
            ...     "income_volatility_cv": 0.15,
            ...     "fuliza_frequency": 2.5,
            ...     ...
            ... }
            >>> result = pipeline.predict_from_features(features)
        
        Raises:
            CreditScoringException: If prediction fails
        """
        try:
            logger.info("Making prediction from pre-computed features...")
            
            # Validate features
            missing_features = [f for f in self.feature_columns if f not in features]
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            
            # Make prediction
            result = self._make_prediction(features)
            
            logger.info(f"Prediction complete: {result['prediction_label']}")
            
            return result
        
        except Exception as e:
            raise CreditScoringException(e, sys)
    
    def _make_prediction(self, features: Dict) -> Dict:
        """
        Internal method to make prediction from features
        
        Args:
            features (Dict): Feature dictionary
        
        Returns:
            Dict: Prediction results
        """
        try:
            # Create feature vector in correct order
            feature_vector = [features.get(col, 0) for col in self.feature_columns]
            feature_array = np.array(feature_vector).reshape(1, -1)
            
            # Scale features
            features_scaled = self.scaler.transform(feature_array)
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            # Interpret results
            prediction_label = "Repaid" if prediction == 1 else "Default"
            risk_level = self._assess_risk_level(probabilities[1])
            
            result = {
                "prediction": int(prediction),
                "prediction_label": prediction_label,
                "probability_default": float(probabilities[0]),
                "probability_repay": float(probabilities[1]),
                "risk_level": risk_level,
                "recommendation": self._get_recommendation(prediction, probabilities[1])
            }
            
            return result
        
        except Exception as e:
            raise CreditScoringException(e, sys)
    
    def _assess_risk_level(self, repay_probability: float) -> str:
        """
        Assess risk level based on repayment probability
        
        Args:
            repay_probability (float): Probability of repayment
        
        Returns:
            str: Risk level (Low, Medium, High)
        """
        if repay_probability >= 0.75:
            return "Low"
        elif repay_probability >= 0.50:
            return "Medium"
        else:
            return "High"
    
    def _get_recommendation(self, prediction: int, repay_probability: float) -> str:
        """
        Get loan recommendation based on prediction
        
        Args:
            prediction (int): Binary prediction (0 or 1)
            repay_probability (float): Probability of repayment
        
        Returns:
            str: Recommendation text
        """
        if prediction == 1:
            if repay_probability >= 0.85:
                return "Approve - High confidence"
            elif repay_probability >= 0.70:
                return "Approve - Standard terms"
            else:
                return "Approve with caution - Monitor closely"
        else:
            if repay_probability < 0.30:
                return "Reject - High default risk"
            elif repay_probability < 0.50:
                return "Reject - Moderate default risk"
            else:
                return "Review manually - Borderline case"
    
    def _validate_raw_input(self, applicant_data: Dict):
        """
        Validate raw applicant data structure
        
        Args:
            applicant_data (Dict): Applicant data dictionary
        
        Raises:
            ValueError: If required fields are missing
        """
        required_keys = ["demographics", "financial_history", "transactions"]
        missing_keys = [key for key in required_keys if key not in applicant_data]
        
        if missing_keys:
            raise ValueError(f"Missing required keys: {missing_keys}")
        
        # Validate demographics
        required_demo = ["age", "employment_type", "years_employed", "education_level"]
        demographics = applicant_data["demographics"]
        missing_demo = [field for field in required_demo if field not in demographics]
        
        if missing_demo:
            raise ValueError(f"Missing demographics fields: {missing_demo}")
        
        # Validate financial history
        required_history = ["past_defaults_count", "past_repaid_loans", "account_age_months"]
        financial_history = applicant_data["financial_history"]
        missing_history = [field for field in required_history if field not in financial_history]
        
        if missing_history:
            raise ValueError(f"Missing financial history fields: {missing_history}")
        
        # Validate transactions
        if not isinstance(applicant_data["transactions"], list):
            raise ValueError("Transactions must be a list")
        
        if len(applicant_data["transactions"]) == 0:
            raise ValueError("Transactions list cannot be empty")
    
    def predict_batch(self, applicants_data: List[Dict]) -> List[Dict]:
        """
        Make predictions for multiple applicants
        
        Args:
            applicants_data (List[Dict]): List of applicant data dictionaries
        
        Returns:
            List[Dict]: List of prediction results
        """
        try:
            logger.info(f"Making batch predictions for {len(applicants_data)} applicants...")
            
            results = []
            for idx, applicant_data in enumerate(applicants_data, 1):
                try:
                    result = self.predict_from_raw_data(applicant_data)
                    result["applicant_index"] = idx
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error predicting applicant {idx}: {e}")
                    results.append({
                        "applicant_index": idx,
                        "error": str(e),
                        "prediction": None
                    })
            
            logger.info(f"Batch prediction complete: {len(results)} results")
            
            return results
        
        except Exception as e:
            raise CreditScoringException(e, sys)


# Testing
if __name__ == "__main__":
    try:
        from src.components.data_ingestion import DataIngestion
        
        print("="*70)
        print("testing prediction pipeline")
        print("="*70)
        
        # Initialize prediction pipeline
        pipeline = PredictPipeline()
        
        # Load sample applicant data
        data_ingestion = DataIngestion()
        applicant_data = data_ingestion.get_applicant_data(applicant_id=1)
        
        # Make prediction
        print("\nmaking prediction for applicant 1...")
        result = pipeline.predict_from_raw_data(applicant_data)
        
        # Print results
        print("\n" + "="*70)
        print("prediction results")
        print("="*70)
        print(f"\nprediction: {result['prediction_label']}")
        print(f"risk level: {result['risk_level']}")
        print(f"\nprobabilities:")
        print(f"  repay:   {result['probability_repay']:.4f} ({result['probability_repay']*100:.2f}%)")
        print(f"  default: {result['probability_default']:.4f} ({result['probability_default']*100:.2f}%)")
        print(f"\nrecommendation: {result['recommendation']}")
        print("="*70)
        
        # Test with pre-computed features
        print("\ntesting prediction from features...")
        
        sample_features = {
            "monthly_avg_income": 20000,
            "income_volatility_cv": 0.15,
            "income_trend_6months": 10.0,
            "num_income_sources": 2,
            "monthly_net_cashflow": 15000,
            "months_negative_cashflow": 0,
            "balance_min": 1000,
            "months_balance_below_500": 2,
            "fuliza_frequency": 3.5,
            "fuliza_repayment_rate": 0.95,
            "debt_to_income_ratio": 0.15,
            "avg_fuliza_amount": 2000,
            "transaction_frequency": 50,
            "savings_rate": 0.10,
            "expense_consistency_cv": 0.20,
            "age": 32,
            "years_employed": 8,
            "education_level_encoded": 1,
            "employment_Salaried": 1,
            "employment_Self-employed": 0,
            "employment_Student/Other": 0,
            "account_age_months": 48,
            "past_defaults_count": 1,
            "past_repaid_loans": 2,
            "credit_score": 580,
            "overdraft_history": 4
        }
        
        result2 = pipeline.predict_from_features(sample_features)
        
        print("\n" + "="*70)
        print("prediction from features")
        print("="*70)
        print(f"\nprediction: {result2['prediction_label']}")
        print(f"risk level: {result2['risk_level']}")
        print(f"\nprobabilities:")
        print(f"  repay:   {result2['probability_repay']:.4f}")
        print(f"  default: {result2['probability_default']:.4f}")
        print(f"\nrecommendation: {result2['recommendation']}")
        print("="*70)
        
        print("\nprediction pipeline test completed successfully")
        
    except Exception as e:
        logger.error(f"Error in prediction pipeline test: {e}")
        raise