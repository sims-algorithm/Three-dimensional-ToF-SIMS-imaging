"""
Setup script for the ToF-SIMS 3D Imaging Toolkit.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name='tofsims',
    version='0.1.0',
    author='ToF-SIMS Research Team',
    description='Python toolkit for 3D ToF-SIMS imaging analysis',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ZhangYuxuanMUC/Three-dimensional-ToF-SIMS-imaging',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Visualization',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.20.0',
        'scipy>=1.8.0',
        'matplotlib>=3.4.0',
        'Pillow>=10.2.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov',
            'black',
            'flake8',
        ],
    },
    keywords='ToF-SIMS imaging mass-spectrometry 3D-imaging visualization',
)
