from setuptools import setup, find_packages

setup(
    name="src.my_package",
    version="0.1",
    packages=find_packages(),
    description="A Python package for analyzing Air Traffic Passenger Statistics",
    author="Afnan",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/your_package_repo",
    install_requires=["pandas", "numpy", "matplotlib", "seaborn", "scikit-learn"],
)
