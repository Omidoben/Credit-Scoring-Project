from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path: str) -> List[str]:
    """Read requirements from file"""
    requirements = []
    with open(file_path) as f:
        requirements = f.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        
        # Remove -e . if present
        if "-e ." in requirements:
            requirements.remove("-e .")
    
    return requirements

setup(
    name='credit_scoring',
    version='0.1.0',
    author='Your Name',
    author_email='omidobenard@gmail.com',
    description='ML-based credit scoring system using M-Pesa transaction data',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)