import pandas as pd
import numpy as np
import sys
from typing import Dict
from src.utils.logger import get_logger
from src.utils.exception import CreditScoringException

logger = get_logger(__name__)


class FeatureExtractor:
    """
    Extract ML features from transaction data
    Works for both bulk processing (training) and single applicant (production)
    """
    
    def __init__(self):
        """Initialize feature extractor"""
        logger.info("FeatureExtractor initialized")
    
    def extract_features(self, transactions_df: pd.DataFrame) -> Dict[str, float]:
        """
        Extract all features from transaction data for a single applicant
        
        Args:
            transactions_df (pd.DataFrame): Transaction data for one applicant
        
        Returns:
            Dict[str, float]: Dictionary of extracted features
        """
        try:
            if transactions_df.empty:
                logger.warning("Empty transaction dataframe provided")
                return self._get_zero_features()
            
            features = {}
            
            # Extract feature groups
            features.update(self._extract_income_features(transactions_df))
            features.update(self._extract_cashflow_features(transactions_df))
            features.update(self._extract_debt_features(transactions_df))
            features.update(self._extract_behavioral_features(transactions_df))
            
            return features
        
        except Exception as e:
            raise CreditScoringException(e, sys)
    
    ## 1) Income features
    
    def _extract_income_features(self, trans_df: pd.DataFrame) -> Dict[str, float]:
        """
        Extract income-related features
        
        Returns:
            - monthly_avg_income: Average monthly income
            - income_volatility_cv: Coefficient of variation (std/mean)
            - income_trend_6months: % change from first 6 to last 6 months
            - num_income_sources: Estimated number of income sources
        """
        try:
            income_trans = trans_df[trans_df["transaction_type"] == "Income"].copy()
            
            if income_trans.empty:
                return {
                    "monthly_avg_income": 0.0,
                    "income_volatility_cv": 0.0,
                    "income_trend_6months": 0.0,
                    "num_income_sources": 0
                }
            
            # Get monthly income
            income_trans = income_trans.copy()
            income_trans["month"] = income_trans["date"].dt.to_period("M")
            monthly_income = income_trans.groupby("month")["amount"].sum()
            
            # Monthly average income
            monthly_avg_income = monthly_income.mean()
            
            # Income volatility 
            income_std = monthly_income.std()
            income_volatility_cv = (income_std / monthly_avg_income) if monthly_avg_income > 0 else 0.0
            
            # Income trend
            if len(monthly_income) >= 6:
                first_6_avg = monthly_income.iloc[:6].mean()
                last_6_avg = monthly_income.iloc[-6:].mean()
                income_trend_6months = ((last_6_avg - first_6_avg) / first_6_avg * 100) if first_6_avg > 0 else 0.0
            else:
                income_trend_6months = 0.0
            
            # Number of income sources (unique income amounts pattern)
            unique_income_amounts = income_trans["amount"].round(-2).nunique()
            num_income_sources = min(unique_income_amounts, 5)
            
            return {
                "monthly_avg_income": round(monthly_avg_income, 2),
                "income_volatility_cv": round(income_volatility_cv, 4),
                "income_trend_6months": round(income_trend_6months, 2),
                "num_income_sources": int(num_income_sources)
            }
        
        except Exception as e:
            logger.error(f"Error extracting income features: {e}")
            return {
                "monthly_avg_income": 0.0,
                "income_volatility_cv": 0.0,
                "income_trend_6months": 0.0,
                "num_income_sources": 0
            }
    

    # 2) CashFlow Features

    def _extract_cashflow_features(self, trans_df: pd.DataFrame) -> Dict[str, float]:
        """
        Extract cash flow features
        
        Returns:
            - monthly_net_cashflow: Average monthly net cashflow
            - months_negative_cashflow: Number of months with negative flow
            - balance_min: Minimum balance reached
            - months_balance_below_500: Months with balance < 500
        """
        try:
            if trans_df.empty:
                return {
                    "monthly_net_cashflow": 0.0,
                    "months_negative_cashflow": 0,
                    "balance_min": 0.0,
                    "months_balance_below_500": 0
                }
            
            # monthly net cashflow
            trans_df = trans_df.copy()
            trans_df["month"] = trans_df["date"].dt.to_period("M")
            monthly_flow = trans_df.groupby("month")["amount"].sum()

            # Average monthly net cashflow
            monthly_net_cashflow = monthly_flow.mean()
            
            # Months with negative cashflow
            months_negative_cashflow = (monthly_flow < 0).sum()
            
            # Minimum balance
            balance_min = trans_df["balance"].min()
            
            # Months where balance dropped below 500
            trans_df["below_500"] = trans_df["balance"] < 500
            months_below_500 = trans_df.groupby("month")["below_500"].any().sum()
            
            return {
                "monthly_net_cashflow": round(monthly_net_cashflow, 2),
                "months_negative_cashflow": int(months_negative_cashflow),
                "balance_min": round(balance_min, 2),
                "months_balance_below_500": int(months_below_500)
            }
        
        except Exception as e:
            logger.error(f"Error extracting cashflow features: {e}")
            return {
                "monthly_net_cashflow": 0.0,
                "months_negative_cashflow": 0,
                "balance_min": 0.0,
                "months_balance_below_500": 0
            }
    
    # 4) Debt Features
    
    def _extract_debt_features(self, trans_df: pd.DataFrame) -> Dict[str, float]:
        """
        Extract debt-related features
        
        Returns:
            - fuliza_frequency: Fuliza usage per month
            - fuliza_repayment_rate: % of borrowed amount repaid
            - debt_to_income_ratio: Total debt / Total income
            - avg_fuliza_amount: Average Fuliza amount
        """
        try:
            fuliza_credit = trans_df[trans_df["transaction_type"] == "Fuliza Credit"]
            fuliza_repay = trans_df[trans_df["transaction_type"] == "Loan Repayment"]
            income_trans = trans_df[trans_df["transaction_type"] == "Income"]
            
            # Fuliza frequency (per month)
            num_months = 12
            fuliza_frequency = len(fuliza_credit) / num_months
            
            # Fuliza repayment rate
            total_borrowed = fuliza_credit["amount"].sum() if len(fuliza_credit) > 0 else 0
            total_repaid = abs(fuliza_repay["amount"].sum()) if len(fuliza_repay) > 0 else 0
            fuliza_repayment_rate = (total_repaid / total_borrowed) if total_borrowed > 0 else 1.0
            fuliza_repayment_rate = min(fuliza_repayment_rate, 1.0)  # Cap at 1.0
            
            # Debt to income ratio
            total_income = income_trans["amount"].sum() if len(income_trans) > 0 else 1
            debt_to_income_ratio = (total_borrowed / total_income) if total_income > 0 else 0.0
            
            # Average Fuliza amount
            avg_fuliza_amount = fuliza_credit["amount"].mean() if len(fuliza_credit) > 0 else 0.0
            
            return {
                "fuliza_frequency": round(fuliza_frequency, 2),
                "fuliza_repayment_rate": round(fuliza_repayment_rate, 4),
                "debt_to_income_ratio": round(debt_to_income_ratio, 4),
                "avg_fuliza_amount": round(avg_fuliza_amount, 2)
            }
        
        except Exception as e:
            logger.error(f"Error extracting debt features: {e}")
            return {
                "fuliza_frequency": 0.0,
                "fuliza_repayment_rate": 1.0,
                "debt_to_income_ratio": 0.0,
                "avg_fuliza_amount": 0.0
            }
    
    
    # Behavioral Features
    
    def _extract_behavioral_features(self, trans_df: pd.DataFrame) -> Dict[str, float]:
        """
        Extract behavioral features
        
        Returns:
            - transaction_frequency: Transactions per month
            - savings_rate: Savings / Income
            - expense_consistency_cv: CV of monthly expenses
        """
        try:
            if trans_df.empty:
                return {
                    "transaction_frequency": 0.0,
                    "savings_rate": 0.0,
                    "expense_consistency_cv": 0.0
                }
            
            # Transaction frequency (transactions per month)
            num_months = 12
            transaction_frequency = len(trans_df) / num_months
            
            # Savings rate
            income_trans = trans_df[trans_df["transaction_type"] == "Income"]
            savings_trans = trans_df[trans_df["transaction_type"] == "Savings In"]
            
            total_income = income_trans["amount"].sum() if len(income_trans) > 0 else 1
            total_savings = savings_trans["amount"].sum() if len(savings_trans) > 0 else 0
            savings_rate = (total_savings / total_income) if total_income > 0 else 0.0
            
            # Expense consistency (CV of monthly expenses)
            expense_types = ["Merchant Payment", "Transfer", "Withdrawal", "Airtime"]
            expense_trans = trans_df[trans_df["transaction_type"].isin(expense_types)].copy()
            
            if len(expense_trans) > 0:
                expense_trans = expense_trans.copy()
                expense_trans["month"] = expense_trans["date"].dt.to_period("M")
                monthly_expenses = expense_trans.groupby("month")["amount"].sum().abs()
                
                expense_mean = monthly_expenses.mean()
                expense_std = monthly_expenses.std()
                expense_consistency_cv = (expense_std / expense_mean) if expense_mean > 0 else 0.0
            else:
                expense_consistency_cv = 0.0
            
            return {
                "transaction_frequency": round(transaction_frequency, 2),
                "savings_rate": round(savings_rate, 4),
                "expense_consistency_cv": round(expense_consistency_cv, 4)
            }
        
        except Exception as e:
            logger.error(f"Error extracting behavioral features: {e}")
            return {
                "transaction_frequency": 0.0,
                "savings_rate": 0.0,
                "expense_consistency_cv": 0.0
            }
    

    
    def _get_zero_features(self) -> Dict[str, float]:
        """Return dictionary of all features with zero values"""
        return {
            # Income features
            'monthly_avg_income': 0.0,
            'income_volatility_cv': 0.0,
            'income_trend_6months': 0.0,
            'num_income_sources': 0,
            # Cashflow features
            'monthly_net_cashflow': 0.0,
            'months_negative_cashflow': 0,
            'balance_min': 0.0,
            'months_balance_below_500': 0,
            # Debt features
            'fuliza_frequency': 0.0,
            'fuliza_repayment_rate': 1.0,
            'debt_to_income_ratio': 0.0,
            'avg_fuliza_amount': 0.0,
            # Behavioral features
            'transaction_frequency': 0.0,
            'savings_rate': 0.0,
            'expense_consistency_cv': 0.0
        }


# Testing
if __name__ == "__main__":
    try:
        from src.components.data_ingestion import DataIngestion
        
        # Load data
        data_ingestion = DataIngestion()
        applicant_data = data_ingestion.get_applicant_data(applicant_id=1)
        
        # Convert transactions to DataFrame
        transactions_df = pd.DataFrame(applicant_data['transactions'])
        transactions_df['date'] = pd.to_datetime(transactions_df['date'])
        
        # Extract features
        extractor = FeatureExtractor()
        features = extractor.extract_features(transactions_df)
        
        print("\n" + "="*70)
        print("Feature extraction test - Applicant 1")
        print("="*70)
        print(f"\nExtracted {len(features)} features:")
        for feature_name, value in features.items():
            print(f"  {feature_name}: {value}")
        
    except Exception as e:
        logger.error(f"Error in feature extraction test: {e}")
        raise
