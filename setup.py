"""
FedFIM Setup Script
"""
from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name='fedfim',
    version='1.0.0',
    author='FedFIM Research Team',
    author_email='research@fedfim.org',
    description='Drift-Aware, Incentive-Compatible, Multimodal Personalized Federated Learning for Financial Intelligence',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/fedfim/fedfim',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Office/Business :: Financial',
    ],
    python_requires='>=3.8',
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.24.0',
        'pandas>=2.0.0',
        'scikit-learn>=1.3.0',
        'matplotlib>=3.7.0',
        'seaborn>=0.12.0',
        'plotly>=5.15.0',
        'streamlit>=1.28.0',
        'yfinance>=0.2.0',
        'textblob>=0.17.0',
        'scipy>=1.11.0',
        'tqdm>=4.65.0',
        'joblib>=1.3.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'jupyter>=1.0.0',
        ],
        'api': [
            'praw>=7.7.0',
            'requests>=2.31.0',
            'python-dotenv>=1.0.0',
        ]
    },
    entry_points={
        'console_scripts': [
            'fedfim-train=src.training.train_fedfim:main',
            'fedfim-dashboard=app:main',
        ],
    },
)