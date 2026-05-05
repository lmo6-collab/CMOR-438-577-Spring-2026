from setuptools import setup, find_packages

setup(
    name="mlpkg",
    version="0.1.0",
    description="A small machine learning package for INDE 577 / CMOR 438 (Spring 2026)",
    author="lmo6-collab",
    packages=find_packages(exclude=["tests", "notebooks"]),
    install_requires=[
        "numpy>=1.20",
        "scikit-learn>=1.0",
    ],
    python_requires=">=3.9",
)
