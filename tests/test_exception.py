import sys
from src.utils.exception import CreditScoringException
from src.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    logger.info("Testing custom exception")
    try:
        a = 10 / 0
        logger.info(f"Computation result: {a}")
    except Exception as e:
        raise CreditScoringException("Test division error", sys)
    
if __name__=="__main__":
    main()