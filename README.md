## Three-dimensional ToF-SIMS Imaging

This repository provides a Python toolkit for three-dimensional ToF-SIMS imaging.  
It aims to reconstruct 3D molecular distributions from serial 2D ToF-SIMS images and includes utilities for data parsing, preprocessing, ROI-based line-profile analysis, and 3D visualization for tissue- and cell-scale studies.

ℹ️ **Project status**
The codebase is currently being organized and refined.  
Some components and example scripts are available, while additional modules and documentation are still under preparation.  
Further updates will be released as the project progresses.\(^o^)/

## Environment Requirements

This project is developed and tested under the following environment.  
The versions listed below are **recommended** rather than strictly required.  
Using similar versions is strongly advised for better compatibility.

### Operating System
- Windows 10 / Windows 11 (64-bit)

### Python
- Python **3.8** (recommended)
- Python 3.9 may work, but has not been fully tested

> Note: Newer Python versions (e.g. 3.10+) may cause dependency resolution or build issues for some scientific libraries.

### Environment Manager
- Anaconda or Miniconda (recommended)
- Conda channel: `conda-forge` (recommended for visualization libraries)

### Recommended Dependencies

| Package | Recommended Version |
|------|------|
| numpy | ~1.23 |
| pandas | >=1.3, <2.0 |
| matplotlib | ~3.5 |
| opencv-python | ~4.9 |
| Pillow | ~9.1 |
| natsort | ~8.1 |
| pyvista | ~0.25 |

### Recommended Installation Workflow

It is recommended to create an isolated Conda environment for this project.

```bash
conda create -n sims3d python=3.8 -y
conda activate sims3d
