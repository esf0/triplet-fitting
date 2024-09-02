from setuptools import setup, find_packages

setup(
    name='conversion_triple_fit',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas>=1.0',
        'numpy>=1.18',
        'scipy>=1.4',
        'scikit-learn>=0.22',
        'tqdm>=4.0',
        'matplotlib>=3.0',
        'seaborn>=0.10',
        'plotly>=4.0',
    ],
    python_requires='>=3.10',
)
