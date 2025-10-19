
from setuptools import setup
from setuptools import find_packages

# FIXME: dask xgboost fails when using pip install... fix.

setup(
    name='noclouds',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'dask',
        'numba',
        'xarray',
        'rioxarray',
        'scikit-learn',
        'xgboost'
    ],
    python_requires='>=3.9',
    url='https://github.com/lewistrotter/noclouds',
    author='Lewis Trotter',
    author_email='mrlewie@outlook.com',
    description='A small python package for detecting, removing and/or filling clouded pixels in satellite images.',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
