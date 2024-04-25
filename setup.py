import pathlib
from setuptools import find_packages, setup

# Current directory reference
here = pathlib.Path(__file__).parent.resolve()

# Read long description from a file
long_description = (here / "README.md").read_text(encoding="utf-8")

# Setup function for your package
setup(
    name="test_confounding",  # Package name
    version="0.0.1",  # Package version
    description="Python implementation of the testing procedures introduced in the paper: Hidden yet quantifiable: A lower bound for confounding strength using randomized trials",  # Short description
    long_description=long_description,  # Long description from README
    long_description_content_type="text/markdown",  # Description type (markdown)
    url="https://github.com/jaabmar/confounder-lower-bound",  # Project URL
    author="Javier Abad & Piersilvio de Bartolomeis",  # Author(s)
    author_email="javier.abadmartinez@ai.ethz.ch",  # Author email(s)
    
    # Classifiers for metadata
    classifiers=[
        "License :: OSI Approved :: MIT License",  # License
        "Programming Language :: Python :: 3",  # Python language support
        "Operating System :: OS Independent",  # OS compatibility
    ],
    
    # Keywords for project relevance
    keywords="machine learning, hidden confounding, hypothesis testing, causal inference, falsification, AISTATS",
    
    # Package directory information
    package_dir={"": "src"},  # Custom source directory
    packages=find_packages(where="src"),  # Find all packages in 'src'
    
    # Dependencies required for installation
    install_requires=[
        "numpy==1.24.3",
        "scipy==1.10.1",
        "mlinsights==0.4.664",
        "scikit-learn==1.3.0",
        "statsmodels==0.13.5",
        "pandas==1.5.3",
        "xgboost==1.7.3",
        "scikit-uplift==0.5.1",
        "quantile-forest==1.2.0",
        "torch==2.0.1",
        "cvxpy==1.3.1",
    ],
    
    # Python version requirement
    python_requires="==3.11.5",  # Ensure correct Python version
    
    # Additional optional dependencies (extra requirements)
    extras_require={
        "tests": ["pytest==7.2.1", "pytest-mock==3.10.0"],  # Additional packages for testing
    },
    
    # Include package data in the distribution
    include_package_data=True,
)
