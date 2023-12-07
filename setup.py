from setuptools import setup, find_packages

setup(
    name="src.my_package",
    version="0.1",
    packages=find_packages(),
    description="A Python package for analyzing Air Traffic Passenger Statistics",
    author="Afnan",
    author_email="amohamm7@mail.yu.edu",
    url="https://github.com/Afnanfaaz/PROJECT-4",
    install_requires=["pandas", "numpy", "matplotlib", "seaborn", "scikit-learn"],
)
