# Spline Interpolation with PyTorch

This repository contains implementations for cubic spline interpolation using PyTorch. The primary components include:

1. `Cubic_Spline_Interpolation.ipynb`: A Jupyter notebook demonstrating the usage and implementation of cubic spline interpolation.
2. `torch_spline_interpolation.py`: A Python module providing functions for 1D and 2D spline interpolation.
3. `qptransformlinear.py`: A Python module for Q-transform linear interpolation using PyTorch.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Jupyter Notebook](#jupyter-notebook)
  - [Python Modules](#python-modules)
- [File Descriptions](#file-descriptions)
  - [Cubic_Spline_Interpolation.ipynb](#cubic_spline_interpolationipynb)
  - [torch_spline_interpolation.py](#torch_spline_interpolationpy)
  - [qptransformlinear.py](#qptransformlinearpy)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/spline-interpolation-pytorch.git
    cd spline-interpolation-pytorch
    ```

## Usage

### Jupyter Notebook

To run the Jupyter notebook:

1. Start Jupyter Notebook:
    ```sh
    jupyter notebook
    ```

2. Open and run `Cubic_Spline_Interpolation.ipynb` to see the implementation and examples of cubic spline interpolation.

### Python Modules

You can use the provided Python modules in your own scripts:

```python
import torch
from torch_spline_interpolation import spline_interpolate, spline_interpolate_2d
from qptransformlinear import QScan, SingleQTransformLinear

# Example usage
# Define your input tensor
input_tensor = torch.rand(100)

# Perform 1D spline interpolation
interpolated_tensor = spline_interpolate(input_tensor, num_x_bins=200)

# For 2D interpolation
input_2d_tensor = torch.rand(100, 100)
interpolated_2d_tensor = spline_interpolate_2d(input_2d_tensor, num_t_bins=200, num_f_bins=200)
