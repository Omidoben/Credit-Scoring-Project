from src.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    logger.info("Test Started")

    try:
        x = 10 / 2
        logger.info(f"Computation result: {x}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
    
    logger.info("Test completed successfully")

if __name__ == "__main__":
    main()