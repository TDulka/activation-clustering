from setuptools import setup, find_packages

setup(
    name="activation-clustering",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.0.0",
        "datasets>=2.0.0",
        "numpy>=1.20.0",
        "tqdm>=4.65.0",
    ],
    python_requires=">=3.8",
)
