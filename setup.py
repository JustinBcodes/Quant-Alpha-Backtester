from setuptools import setup, find_packages

setup(
    name="quant-alpha-aws",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "yfinance>=0.2.0",
        "boto3>=1.26.0",
        "fastapi>=0.95.0",
        "uvicorn>=0.22.0",
        "streamlit>=1.22.0",
        "plotly>=5.14.0",
        "python-dotenv>=1.0.0"
    ],
    python_requires=">=3.9",
    author="Your Name",
    author_email="your.email@example.com",
    description="Multi-Factor Alpha Signal Generator and Backtesting Engine",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
    ],
) 