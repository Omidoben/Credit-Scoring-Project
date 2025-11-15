import sys
from src.utils.config_loader import ConfigLoader
from src.utils.exception import CreditScoringException

def main():
    try:
        # Initialize the ConfigLoader
        config_loader = ConfigLoader(config_path="configs/config.yaml")
        print("Config loaded successfully!")

        # Test getting an existing key
        raw_data_path = config_loader.get("data.raw_data_path")
        print(f"data.raw_data_path: {raw_data_path}")

        # Test getting a missing key with default
        missing_value = config_loader.get("data.non_existing_key")
        print(f"Missing key returned default: {missing_value}")

        # Test getting data config
        data_config = config_loader.get_data_config()
        print(f"Data config: {data_config}")

        # Test getting all features
        all_features = config_loader.get_all_features()
        print(f"All features: {all_features}")

    except Exception as e:
        raise CreditScoringException("ConfigLoader test failed", sys) from e

if __name__ == "__main__":
    main()
