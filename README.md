# Spline Interpolation with PyTorch

This repository contains implementations for cubic spline interpolation using PyTorch. The primary components include:

1. `Cubic_Spline_Interpolation.ipynb`: A Jupyter notebook demonstrating the usage and implementation of cubic spline interpolation.
2. `GPU_qtransform_benchmark.ipynb`: A jupyter notebook to test inputs with batch dimenions for interpolation and for speed benchmarking the qtransform against gwpy's qtransform on batched input.
3. `torch_spline_interpolation.py`: A Python module providing functions for 1D and 2D spline interpolation.
4. `qptransformlinear.py`: A Python module for Q-transform linear interpolation using PyTorch.
5. `qtransform.py`: integration of https://github.com/ML4GW/ml4gw/blob/dev/ml4gw/transforms/qtransform.py with `torch_spline_interpolation.py`
6. `torch_smoothing_spline_interpolation.py`: A Python module for smoothing spline interpolation using PyTorch.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Jupyter Notebook](#jupyter-notebook)
  - [Python Modules](#python-modules)
- [File Descriptions](#file-descriptions)
  - [Cubic_Spline_Interpolation.ipynb](#cubic_spline_interpolationipynb)
  - [GPU_qtransform_benchmark.ipynb](#GPU_qtransform_benchmarkipynb)
  - [torch_spline_interpolation.py](#torch_spline_interpolationpy)
  - [qptransformlinear.py](#qptransformlinearpy)
  - [qtransform.py](#qtransformpy)
  - [torch_smoothing_spline_interpolation.py](#torch_smoothing_spline_interpolationpy)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/dottormale/Qtransform_torch.git
    cd Qtransform_torch
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
from torch_smoothing_spline_interpolation import smoothing_spline_interpolate
from qptransformlinear import QScan, SingleQTransformLinear

# Example usage
# Define your input tensor
input_tensor = torch.rand(100)

# Perform 1D spline interpolation
interpolated_tensor = spline_interpolate(input_tensor, num_x_bins=200)

# For 2D interpolation
input_2d_tensor = torch.rand(100, 100)
interpolated_2d_tensor = spline_interpolate_2d(input_2d_tensor, num_t_bins=200, num_f_bins=200)

# For smoothing spline interpolation
smoothed_tensor = smoothing_spline_interpolate(input_tensor, smooth_factor=0.5)
```

## File Descriptions

### Cubic_Spline_Interpolation.ipynb

The Jupyter notebook contains several sections demonstrating different aspects of cubic spline interpolation:

- **Set GPU**: Setup for running the notebook on a GPU.
- **Scipy Example**: Example of bivariate cubic spline interpolation using `scipy.interpolate.RectBivariateSpline`.
- **CPU For Loop Implementation**: Custom implementation of bivariate cubic natural spline interpolation using PyTorch, showcasing the algorithm with for loops.
- **GPU Tensorized Implementation**: More optimized implementation using tensor operations for GPU support.
- **Final: GPU Tensorized Alternative Matrix System Solution**: Further optimized implementation focusing on solving the linear system for control points computation.
- **GWs Qplots Interpolation**: Applying the interpolation methods to spectrograms and comparing with `gwpy.q_transform()`.
- **Smoothing Spline (Addressing Border Effects in 1D Interpolation)**: Analysis of different 1D interpolation methods using cubic splines and addressing border effects.

### GPU_qtransform_benchmark.ipynb

The Jupyter notebook contains several sections demonstrating different aspects of cubic spline interpolation:

- **Set GPU**: Setup for running the notebook on a GPU.
- **Load data**: Load open data from interferometer of choice and organise it in batches.
- **Test transform with interpolation**: Speed benchmark of custom torch based qtransform against gwpy's on batched input.
- **Test interpolation**: Test custom torch based 1D and 2D spline interpolation on batched input
  
### torch_spline_interpolation.py

This module provides functions for performing cubic spline interpolation in 1D and 2D using PyTorch:

- `spline_interpolate(input_tensor, num_x_bins)`: Performs 1D cubic spline interpolation.
- `spline_interpolate_2d(input_tensor, num_t_bins, num_f_bins)`: Performs 2D cubic spline interpolation.

### qptransformlinear.py

This module provides functionality for Q-transform linear interpolation using PyTorch:

- `QScan`: Class for handling Q-scan operations.
- `SingleQTransformLinear`: Class for performing single Q-transform linear interpolation.

### qtransform.py

This module provides functionality for Q-transform interpolation using PyTorch.
The package is the same as 
- https://github.com/ML4GW/ml4gw/blob/dev/ml4gw/transforms/qtransform.py
with the only difference being the interpolation methods implemented in torch_spline_interpolation.py replacing pytorch's standart interpolation methods.

### torch_smoothing_spline_interpolation.py

This module provides functions for performing smoothing spline interpolation using PyTorch:

- `smoothing_spline_interpolate(input_tensor, smooth_factor)`: Performs smoothing spline interpolation with a specified smoothing factor.

## Dependencies

- Python 3.x
- PyTorch
- SciPy
- NumPy
- Matplotlib
- Jupyter Notebook

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.
Contact: lorenzo.asprea@to.infn.it
