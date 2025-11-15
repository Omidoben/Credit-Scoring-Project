import sys
from src.utils.logger import get_logger

logger = get_logger(__name__)

class CreditScoringException(Exception):
    """
    Custome Exception class for Credit Scoring Project
    Provides detailed error information including filename and file number
    """
    def __init__(self, error_message: str, error_detail: sys):
        """
        Initialize custom exception
        
        Args:
            error_message (str): Error message
            error_detail (sys): System exception info
        """
        super().__init__(error_message)
        self.error_message = self._get_detailed_error_message(error_message, error_detail)
        logger.error(self.error_message)
    
    def _get_detailed_error_message(self, error: str, error_detail: sys) -> str:
        """
        Extract detailed error information
        
        Args:
            error (str): Error message
            error_detail (sys): System exception info
        """
        _, _, exc_tb = error_detail.exc_info()
        
        if exc_tb is not None:
            file_name = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno
            
            return (
                f"Error occurred in script: [{file_name}] "
                f"at line number: [{line_number}] "
                f"error message: [{str(error)}]"
            )
        else:
            return f"Error: {str(error)}"
    
    def __str__(self):
        return self.error_message
    
    def __repr__(self):
        return f"CreditScoringException({self.error_message})"
    

# This module defines a custom exception class for the Credit Scoring project.
#
# Key features:
# - Wraps Python exceptions inside `CreditScoringException` for cleaner errors.
# - Automatically logs the full error details using the project logger.
# - Extracts and displays:
#       • filename where the error occurred
#       • line number of the error
#       • original error message