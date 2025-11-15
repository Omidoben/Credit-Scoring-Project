import yaml
import os
from pathlib import Path
from typing import Any, Dict
from src.utils.logger import get_logger
from src.utils.exception import CreditScoringException
import sys

logger = get_logger(__name__)


class ConfigLoader:
    """
    Load and manage project configuration from YAML file
    """
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize config loader
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config_path = config_path
        self.config = self._load_config()
        logger.info(f"Configuration loaded from {config_path}")
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file
        
        Returns:
            Dict: Configuration dictionary
        
        Raises:
            CreditScoringException: If config file not found or invalid
        """
        try:
            if not os.path.exists(self.config_path):
                raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            if config is None:
                raise ValueError("Config file is empty")
            
            return config
        
        except Exception as e:
            raise CreditScoringException(e, sys)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key (supports nested keys with dot notation)
        
        Args:
            key (str): Configuration key (e.g., 'data.raw_data_path')
            default (Any): Default value if key not found
        
        Returns:
            Any: Configuration value
        
        Example:
            config = ConfigLoader()
            config.get('data.raw_data_path')
            'data/raw/'
        """
        try:
            keys = key.split('.')
            value = self.config
            
            for k in keys:
                if isinstance(value, dict):
                    value = value.get(k)
                else:
                    return default
            
            return value if value is not None else default
        
        except Exception as e:
            logger.warning(f"Error getting config key '{key}': {e}")
            return default
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration"""
        return self.config.get('data', {})    # Returns {} (default value), no error. If config file might be missing a section
        # return self.config["data"]
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return self.config.get('model', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration"""
        return self.config.get('training', {})
    
    def get_feature_config(self) -> Dict[str, Any]:
        """Get feature configuration"""
        return self.config.get('features', {})
    
    def get_all_features(self) -> list:
        """Get list of all feature names"""
        features = []
        feature_config = self.get_feature_config()
        
        for feature_group in feature_config.values():
            if isinstance(feature_group, list):
                features.extend(feature_group)
        
        return features
    
    def __repr__(self):
        return f"ConfigLoader(config_path='{self.config_path}')"