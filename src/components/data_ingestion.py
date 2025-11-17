import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass
from typing import Optional, Tuple

from src.utils.logger import get_logger
from src.utils.exception import CreditScoringException
from src.utils.config_loader import ConfigLoader

logger = get_logger(__name__)

@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion"""
    raw_data_path: str
    demographics_file: str
    financial_history_file: str
    transactions_file: str

    def __post_init__(self):
        """Construct full file paths"""
        self.demographics_path = os.path.join(self.raw_data_path, self.demographics_file)
        self.financial_history_path = os.path.join(self.raw_data_path, self.financial_history_file)
        self.transactions_path = os.path.join(self.raw_data_path, self.transactions_file)

class DataIngestion:
    """
    Handles data ingestion from raw csv files
    - Loads demographics, financial history, and transactions data
    """
    def __init__(self, config: Optional[ConfigLoader] = None):
        """Initializes data ingestion"""
        if config is None:
            config = ConfigLoader()

        self.config = config
        data_config = config.get_data_config()

        self.ingestion_config = DataIngestionConfig(
            raw_data_path=data_config.get("raw_data_path", "data/raw"),
            demographics_file=data_config.get("demographics_file", "demographics.csv"),
            financial_history_file=data_config.get("financial_history_file", "financial_history.csv"),
            transactions_file=data_config.get("transactions_file", "transactions.csv")
        )

        logger.info("Data Ingestion Initialized")

    def load_demographics(self) -> pd.DataFrame:
        """"
        Load demographic data

        Returns:
            pd.DataFrame - demographics data
        """
        try:
            logger.info(f"Loading demographics data from {self.ingestion_config.demographics_path}")

            if not os.path.exists(self.ingestion_config.demographics_path):
                raise FileNotFoundError(f"Demographics file not found: {self.ingestion_config.demographics_path}")
            
            df = pd.read_csv(self.ingestion_config.demographics_path)
            logger.info(f"Loaded demographics: {df.shape[0]} rows × {df.shape[1]} columns")

            # validation
            required_cols = ["applicant_id", "age", "employment_type"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            return df

        except Exception as e:
            raise CreditScoringException(e, sys)
        
    def load_financial_history(self) -> pd.DataFrame:
        """
        Load financial history data
        
        Returns:
            pd.DataFrame: Financial history data
        """
        try:
            logger.info(f"Loading financial history data from {self.ingestion_config.financial_history_path}")
            
            if not os.path.exists(self.ingestion_config.financial_history_path):
                raise FileNotFoundError(f"Financial history file not found: {self.ingestion_config.financial_history_path}")
            
            df = pd.read_csv(self.ingestion_config.financial_history_path)
            
            logger.info(f"Loaded financial history: {df.shape[0]} rows × {df.shape[1]} columns")
            
            required_columns = ["applicant_id", "past_defaults_count"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            return df
        
        except Exception as e:
            raise CreditScoringException(e, sys)
    
    def load_transactions(self, applicant_id: Optional[int] = None) -> pd.DataFrame:
        """
        Load transaction data (all or for specific applicant)
        
        Args:
            applicant_id (int, optional): Load transactions for specific applicant only
        
        Returns:
            pd.DataFrame: Transactions dataframe
        """
        try:
            logger.info(f"Loading transactions from {self.ingestion_config.transactions_path}")
            
            if not os.path.exists(self.ingestion_config.transactions_path):
                raise FileNotFoundError(f"Transactions file not found: {self.ingestion_config.transactions_path}")
            
            df = pd.read_csv(self.ingestion_config.transactions_path)
            
            # Convert date column to datetime
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            
            # Filter by applicant_id if provided
            if applicant_id is not None:
                df = df[df["applicant_id"] == applicant_id]
                logger.info(f"Loaded {len(df)} transactions for applicant {applicant_id}")
            else:
                logger.info(f"Loaded transactions: {df.shape[0]} rows × {df.shape[1]} columns")
            
            required_columns = ["applicant_id", "date", "transaction_type", "amount"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            return df
        
        except Exception as e:
            raise CreditScoringException(e, sys)
        
    def load_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all data sources
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 
                (demographics, financial_history, transactions)
        """
        try:
            logger.info("Loading all the three files")
            demographics_df = self.load_demographics()
            financial_history_df = self.load_financial_history()
            transactions_df = self.load_transactions()
            
            # Validate applicant_id alignment across the three files
            demo_ids = set(demographics_df["applicant_id"].unique())
            history_ids = set(financial_history_df["applicant_id"].unique())
            trans_ids = set(transactions_df["applicant_id"].unique())
            
            if demo_ids != history_ids or demo_ids != trans_ids:
                logger.warning("Applicant IDs mismatch across datasets")
                logger.warning(f"Demographics: {len(demo_ids)} applicants")
                logger.warning(f"Financial History: {len(history_ids)} applicants")
                logger.warning(f"Transactions: {len(trans_ids)} applicants")
            else:
                logger.info(f"All datasets aligned: {len(demo_ids)} applicants")
            
            logger.info("Data Ingestion Complete")
            
            return demographics_df, financial_history_df, transactions_df
        
        except Exception as e:
            raise CreditScoringException(e, sys)

    def get_applicant_data(self, applicant_id: int) -> dict:
        """
        Get all data for a specific applicant
        
        Args:
            applicant_id (int): Applicant ID
        
        Returns:
            Dict: Dictionary containing all applicant data
        """
        try:
            demographics_df = self.load_demographics()
            financial_history_df = self.load_financial_history()
            transactions_df = self.load_transactions(applicant_id=applicant_id)
            
            demo_data = demographics_df[demographics_df["applicant_id"] == applicant_id]
            if demo_data.empty:
                raise ValueError(f"Applicant {applicant_id} not found in demographics")
            
            history_data = financial_history_df[financial_history_df["applicant_id"] == applicant_id]
            if history_data.empty:
                raise ValueError(f"Applicant {applicant_id} not found in financial history")
            
            return {
                "applicant_id": applicant_id,
                "demographics": demo_data.to_dict("records")[0],
                "financial_history": history_data.to_dict("records")[0],
                "transactions": transactions_df.to_dict("records")
            }
        
        except Exception as e:
            raise CreditScoringException(e, sys)


# Testing
if __name__ == "__main__":
    try:
        # Initialize data ingestion
        data_ingestion = DataIngestion()
        
        # Load all data
        demographics_df, financial_history_df, transactions_df = data_ingestion.load_all_data()
        
        print("\n" + "="*70)
        print("Data Ingestion test")
        print("="*70)
        print(f"\nDemographics shape: {demographics_df.shape}")
        print(f"Financial History shape: {financial_history_df.shape}")
        print(f"Transactions shape: {transactions_df.shape}")
        
        # Test single applicant loading
        print("\n" + "="*70)
        print("Testing Single Applicant Loading")
        print("="*70)
        applicant_data = data_ingestion.get_applicant_data(applicant_id=432)
        print(f"\nApplicant 1 data loaded:")
        print(f"  - Demographics: {len(applicant_data['demographics'])} fields")
        print(f"  - Financial History: {len(applicant_data['financial_history'])} fields")
        print(f"  - Transactions: {len(applicant_data['transactions'])} records")
        
    except Exception as e:
        logger.error(f"Error in data ingestion test: {e}")
        raise