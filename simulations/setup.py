#!/usr/bin/env python3

from setuptools import setup, find_packages
import os

# Read README for long description
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='neptune-prediction',
    version='1.0.0',
    author='Anton Thiriaux and Don van Beusichem',
    url='https://github.com/beusy/neptune-prediction',
    description="Le Verrier's algorithm for predicting Neptune from Uranus perturbations",
    long_description=long_description,
    long_description_content_type='text/markdown',
    
    packages=find_packages(exclude=['tests', 'tests.*', 'examples', 'examples.*']),
    
    python_requires='>=3.10',
    
    install_requires=[
        'numpy>=1.24.0',
        'scipy>=1.10.0',
    ],
    
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'mypy>=1.0.0',
        ],
        'viz': [
            'matplotlib>=3.7.0',
        ],
        'notebook': [
            'jupyter>=1.0.0',
            'ipywidgets>=8.0.0',
        ],
        'all': [
            'matplotlib>=3.7.0',
            'jupyter>=1.0.0',
            'ipywidgets>=8.0.0',
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
        ],
    },
    
    package_data={
        'leverrier': ['data/*.csv'],
    },
    
    include_package_data=True,
    
    entry_points={
        'console_scripts': [
            'neptune-predict=leverrier.inverse:main',
        ],
    },
    
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
    
    keywords=[
        'astronomy',
        'celestial mechanics',
        'neptune',
        'le verrier',
        'inverse problem',
        'perturbation theory',
        'orbital mechanics',
    ],
)
