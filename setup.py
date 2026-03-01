from setuptools import setup, find_packages

setup(
    name="fraud_detection",
    version="2.0.0",
    description="A BERT-based financial fraud detection system with Gradio and LINE Bot interfaces.",
    author="jerrychen428",
    packages=find_packages(),
    install_requires=[
        "transformers==4.40.1",
        "torch==2.2.2",
        "scikit-learn==1.4.2",
        "pandas==2.2.2",
        "gradio==4.26.0",
        "numpy==1.26.4",
        "accelerate==0.29.1",
        "flask>=3.0.0",
        "line-bot-sdk>=3.0.0",
        "python-dotenv>=1.0.0",
    ],
    python_requires=">=3.10",
)
