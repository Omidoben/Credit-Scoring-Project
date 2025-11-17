import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

from src.utils.logger import get_logger
from src.utils.exception import CreditScoringException
from src.utils.config_loader import ConfigLoader
from src.data.feature_extractor import FeatureExtractor

logger = get_logger(__name__)


@dataclass
class DataTransformationConfig:
    """Configuration for data transformation"""
    processed_data_path: str
    model_dir: str
    scaler_path: str
    feature_columns_path: str
    test_size: float
    random_state: int
    
    def __post_init__(self):
        """Ensure directories exist"""
        os.makedirs(self.processed_data_path, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)


class DataTransformation:
    """
    Handle data transformation:
    1. Extract features from transactions
    2. Merge with demographics and financial history
    3. Encode categorical variables
    4. Scale numerical features
    5. Split into train/test sets
    """
    
    def __init__(self, config: Optional[ConfigLoader] = None):
        """
        Initialize DataTransformation
        
        Args:
            config (ConfigLoader, optional): Configuration loader instance
        """
        if config is None:
            config = ConfigLoader()
        
        self.config = config
        data_config = config.get_data_config()
        model_config = config.get_model_config()
        
        self.transformation_config = DataTransformationConfig(
            processed_data_path=data_config.get("processed_data_path", "data/processed/"),
            model_dir=model_config.get("model_dir", "models/"),
            scaler_path=os.path.join(model_config.get("model_dir", "models/"), "scaler.pkl"),
            feature_columns_path=os.path.join(model_config.get("model_dir", "models/"), "feature_columns.pkl"),
            test_size=model_config.get("test_size", 0.2),
            random_state=model_config.get("random_state", 42)
        )
        
        self.feature_extractor = FeatureExtractor()
        self.scaler = StandardScaler()
        self.feature_columns = None
        
        logger.info("DataTransformation initialized")
    
    def extract_features_bulk(self, 
                             demographics_df: pd.DataFrame,
                             financial_history_df: pd.DataFrame,
                             transactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features for all applicants
        
        Args:
            demographics_df: Demographics data
            financial_history_df: Financial history data
            transactions_df: Transaction data
        
        Returns:
            pd.DataFrame: Feature dataframe with all extracted features
        
        """
        try:
            logger.info("Starting feature extraction for all applicants")
            
            all_features = []
            total_applicants = demographics_df["applicant_id"].nunique()
            
            for idx, applicant_id in enumerate(demographics_df["applicant_id"].unique(), 1):
                if idx % 100 == 0:
                    logger.info(f"Processing applicant {idx}/{total_applicants}...")
                
                # Get transactions for this applicant
                applicant_trans = transactions_df[transactions_df["applicant_id"] == applicant_id]
                
                # Extract transaction features
                features = self.feature_extractor.extract_features(applicant_trans)
                features["applicant_id"] = applicant_id
                
                all_features.append(features)
            
            # Convert to DataFrame
            features_df = pd.DataFrame(all_features)
            
            # Merge with demographics
            features_df = features_df.merge(demographics_df, on="applicant_id", how="left")
            
            # Merge with financial history
            features_df = features_df.merge(financial_history_df, on="applicant_id", how="left")
            
            logger.info(f"Feature extraction complete: {features_df.shape[0]} rows × {features_df.shape[1]} columns")
            
            return features_df
        
        except Exception as e:
            raise CreditScoringException(e, sys)
    
    def process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process features: encode categoricals, handle missing values
        
        Args:
            df: Raw feature dataframe
        
        Returns:
            pd.DataFrame: Processed feature dataframe
        """
        try:
            logger.info("Processing features")
            
            df_processed = df.copy()
            
            # One-hot encode employment_type (drop first to avoid multicollinearity)
            if "employment_type" in df_processed.columns:
                employment_dummies = pd.get_dummies(
                    df_processed["employment_type"], 
                    prefix="employment",
                    drop_first=True
                )
                df_processed = pd.concat([df_processed, employment_dummies], axis=1)
                df_processed.drop("employment_type", axis=1, inplace=True)
                logger.info(f"One-hot encoded employment_type: {len(employment_dummies.columns)} features")
            
            # Ordinal encode education_level
            if "education_level" in df_processed.columns:
                education_mapping = {
                    "Primary": 0,
                    "Secondary": 1,
                    "Tertiary": 2
                }
                df_processed["education_level_encoded"] = df_processed["education_level"].map(education_mapping)
                df_processed.drop("education_level", axis=1, inplace=True)
                logger.info("Ordinal encoded education_level")
            
            # Handle missing values in credit_score (fill with median if exists)
            if "credit_score" in df_processed.columns:
                if df_processed["credit_score"].isnull().sum() > 0:
                    median_score = df_processed["credit_score"].median()
                    df_processed["credit_score"] = df_processed["credit_score"].fillna(median_score)
                    logger.info(f"Filled missing credit_score with median: {median_score}")

            columns_to_drop = []
            for col in df_processed.columns:
                if df_processed[col].dtype == 'object':
                    columns_to_drop.append(col)
        
            if columns_to_drop:
                logger.info(f"Dropping non-numeric columns: {columns_to_drop}")
                df_processed.drop(columns_to_drop, axis=1, inplace=True)
            
            logger.info("Feature processing complete")
            
            return df_processed
        
        except Exception as e:
            raise CreditScoringException(e, sys)
    
    def generate_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate target variable (loan_status) based on features
        
        Logic:
        - Strong income + low Fuliza → Repaid (1)
        - Volatile income + high Fuliza → Default (0)
        - Target: ~75% Repaid, 25% Default
        
        Args:
            df: Feature dataframe
        
        Returns:
            pd.DataFrame: Dataframe with loan_status column
        """
        try:
            logger.info("Generating target variable...")
            
            df_target = df.copy()
            
            # Normalize features for scoring (0-1 scale)
            def normalize(series, higher_is_better=True):
                min_val = series.min()
                max_val = series.max()
                if max_val == min_val:
                    return pd.Series([0.5] * len(series), index=series.index)
                normalized = (series - min_val) / (max_val - min_val)
                if not higher_is_better:
                    normalized = 1 - normalized
                return normalized
            
            # Income strength (higher = better)
            income_score = normalize(df_target['monthly_avg_income'], higher_is_better=True)
            
            # Income stability (lower volatility = better)
            income_stability_score = normalize(df_target['income_volatility_cv'], higher_is_better=False)
            
            # Cash flow health (higher = better)
            cashflow_score = normalize(df_target['monthly_net_cashflow'], higher_is_better=True)
            
            # Balance health (higher min balance = better)
            balance_score = normalize(df_target['balance_min'], higher_is_better=True)
            
            # Debt burden (lower = better)
            fuliza_freq_score = normalize(df_target['fuliza_frequency'], higher_is_better=False)
            debt_ratio_score = normalize(df_target['debt_to_income_ratio'], higher_is_better=False)
            
            # Repayment behavior (higher = better)
            repayment_score = df_target['fuliza_repayment_rate']
            
            # Past behavior (lower defaults = better)
            past_defaults_score = normalize(df_target['past_defaults_count'], higher_is_better=False)
            past_repaid_score = normalize(df_target['past_repaid_loans'], higher_is_better=True)
            
            # Composite creditworthiness score (0-100)
            creditworthiness = (
                income_score * 0.15 +
                income_stability_score * 0.10 +
                cashflow_score * 0.10 +
                balance_score * 0.05 +
                fuliza_freq_score * 0.10 +
                debt_ratio_score * 0.10 +
                repayment_score * 0.15 +
                past_defaults_score * 0.15 +
                past_repaid_score * 0.10
            ) * 100
            
            # Add randomness
            np.random.seed(self.transformation_config.random_state)
            noise = np.random.normal(0, 10, len(df_target))
            creditworthiness = np.clip(creditworthiness + noise, 0, 100)
            
            # Convert to binary outcome (threshold for ~25% default rate)
            threshold = np.percentile(creditworthiness, 25)
            df_target["loan_status"] = (creditworthiness > threshold).astype(int)
            
            df_target["creditworthiness_score"] = creditworthiness.round(2)
            
            # Log distribution
            repaid_count = (df_target["loan_status"] == 1).sum()
            default_count = (df_target["loan_status"] == 0).sum()
            logger.info(f"Target generated:")
            logger.info(f"Repaid (1): {repaid_count} ({repaid_count/len(df_target)*100:.1f}%)")
            logger.info(f"Default (0): {default_count} ({default_count/len(df_target)*100:.1f}%)")
            
            return df_target
        
        except Exception as e:
            raise CreditScoringException(e, sys)
    
    def prepare_for_training(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for model training
        Select only numeric features needed for ML
        
        Args:
            df: Processed feature dataframe with target
        
        Returns:
            Tuple[pd.DataFrame, pd.Series]: (X, y) for training
        """
        try:
            logger.info("Preparing data for training...")
            
            # Define features to keep for ML
            feature_groups = self.config.get_feature_config()
            
            ml_features = []
            
            # Transaction features
            for group in ["income", "cashflow", "debt", "behavioral"]:
                if group in feature_groups:
                    ml_features.extend(feature_groups[group])
            
            # Demographics features
            ml_features.extend(["age", "years_employed", "education_level_encoded"])
            
            # Employment type
            employment_cols = [col for col in df.columns if col.startswith("employment_")]
            ml_features.extend(employment_cols)
            
            # Financial history features
            history_features = ["account_age_months", "past_defaults_count", 
                              "past_repaid_loans", "overdraft_history"]
            
            # Add credit_score if available
            if "credit_score" in df.columns:
                history_features.append("credit_score")
            
            ml_features.extend(history_features)
            
            # Filter to only features that exist in dataframe
            ml_features = [f for f in ml_features if f in df.columns]

            non_numeric = []
            for col in ml_features:
                if df[col].dtype == 'object':
                    non_numeric.append(col)
        
            if non_numeric:
                logger.warning(f"Removing non-numeric columns: {non_numeric}")
                ml_features = [f for f in ml_features if f not in non_numeric]
            
            # Store feature columns for later use
            self.feature_columns = ml_features
            
            # Extract X and y
            X = df[ml_features]
            y = df["loan_status"]
            
            logger.info(f"Selected {len(ml_features)} features for training")
            logger.info(f"  Shape: {X.shape}")
            
            return X, y
        
        except Exception as e:
            raise CreditScoringException(e, sys)
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple:
        """
        Split data into train and test sets
        
        Args:
            X: Feature matrix
            y: Target vector
        """
        try:
            logger.info(f"Splitting data (test_size={self.transformation_config.test_size})...")
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.transformation_config.test_size,
                random_state=self.transformation_config.random_state,
                stratify=y
            )
            
            logger.info(f"Train set: {X_train.shape[0]} samples")
            logger.info(f"Test set: {X_test.shape[0]} samples")
            
            return X_train, X_test, y_train, y_test
        
        except Exception as e:
            raise CreditScoringException(e, sys)
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scale features using StandardScaler
        Fit on train, transform both train and test
        
        Args:
            X_train: Training features
            X_test: Test features
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (X_train_scaled, X_test_scaled)
        """
        try:
            logger.info("Scaling features...")
            
            # Fit scaler on training data
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            # Transform test data
            X_test_scaled = self.scaler.transform(X_test)
            
            logger.info("Features scaled using StandardScaler")
            
            # Save scaler
            joblib.dump(self.scaler, self.transformation_config.scaler_path)
            logger.info(f"Scaler saved to: {self.transformation_config.scaler_path}")
            
            # Save feature columns
            joblib.dump(self.feature_columns, self.transformation_config.feature_columns_path)
            logger.info(f"Feature columns saved to: {self.transformation_config.feature_columns_path}")
            
            return X_train_scaled, X_test_scaled
        
        except Exception as e:
            raise CreditScoringException(e, sys)
    
    def transform_all(self,
                     demographics_df: pd.DataFrame,
                     financial_history_df: pd.DataFrame,
                     transactions_df: pd.DataFrame) -> Tuple:
        """
        Complete transformation pipeline
        
        Args:
            demographics_df: Demographics data
            financial_history_df: Financial history data
            transactions_df: Transaction data
        
        Returns:
            Tuple: (X_train_scaled, X_test_scaled, y_train, y_test, feature_df)
        """
        try:
            logger.info("Starting Data transformation pipeline")
            
            # Extract features
            features_df = self.extract_features_bulk(
                demographics_df, 
                financial_history_df, 
                transactions_df
            )
            
            # Process features
            features_df = self.process_features(features_df)
            
            # Generate target variable
            features_df = self.generate_target_variable(features_df)
            
            # Save processed data
            output_path = os.path.join(
                self.transformation_config.processed_data_path,
                self.config.get("data.ml_data_file", "ml_training_data.csv")
            )
            features_df.to_csv(output_path, index=False)
            logger.info(f"Processed data saved to: {output_path}")
            
            # Prepare for training
            X, y = self.prepare_for_training(features_df)
            
            # Split data
            X_train, X_test, y_train, y_test = self.split_data(X, y)
            
            # Scale features
            X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
            
            logger.info("Data transformation complete")
            
            return X_train_scaled, X_test_scaled, y_train, y_test, features_df
        
        except Exception as e:
            raise CreditScoringException(e, sys)


# Testing
if __name__ == "__main__":
    try:
        from src.components.data_ingestion import DataIngestion
        
        # Load data
        logger.info("Testing Data Transformation...")
        data_ingestion = DataIngestion()
        demographics_df, financial_history_df, transactions_df = data_ingestion.load_all_data()
        
        # Transform data
        data_transformation = DataTransformation()
        X_train, X_test, y_train, y_test, features_df = data_transformation.transform_all(
            demographics_df,
            financial_history_df,
            transactions_df
        )
        
        print("\n" + "="*70)
        print("Data transformation test results")
        print("="*70)
        print(f"\nTraining set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        print(f"\nTarget distribution (train):")
        print(f"  Repaid (1): {(y_train == 1).sum()}")
        print(f"  Default (0): {(y_train == 0).sum()}")
        print(f"\nFeature columns ({len(data_transformation.feature_columns)}):")
        for i, col in enumerate(data_transformation.feature_columns[:10], 1):
            print(f"  {i}. {col}")
        if len(data_transformation.feature_columns) > 10:
            print(f"  ... and {len(data_transformation.feature_columns) - 10} more")
        
    except Exception as e:
        logger.error(f"Error in data transformation test: {e}")
        raise
