This folder contains a Jupyter Notebook which provides examples on how to use the **Q-Transform with Amplitude Modulation** (QTAM),
together with two Python scripts containing the QTAM class together with other accessory classes and functions needed for the analytis 
of time series in the time-frequency space.

## Installation

1.  You can clone this folder with the command 
    ```
    git clone https://github.com/dottormale/Qtransform_torch.git
    cd Qtransform_torch
## Files Description

- **qtransform_gpu_2_5_0.py** : this script contains the implementation of the QTAM. The structure of the classes takes inspiration from the GWPy code, adapted for the new transform and ancillary functions. The class *SingleQTransform* computes the
- QTAM of a batch of 1D timeseries at a fixed value of q, while the class *QScan* computes the QTAM for multiple values of q and returns the interpolated Q-transform with the highest energy.

- **Annalisa_2_1_0.py**: ths script contains many classes and utilities used for the pre-processing of GW signals. It contains the class *STFTWhiten* which is used to whiten the timeseries in the example usage of the QTAM.

- **QTAM_Example_Usage.ipynb**: this Jupyter notebook contains examples on how to use the QTAM on a real GW signal. It includes a description of some of the mathematical features of the QTAM and can be customised by the user.
