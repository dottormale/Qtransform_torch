This folder contains a Jupyter Notebook which provides examples on how to use the Q-Transform with Amplitude Modulation (QTAM), together with two Python scripts containing the QTAM class together with other accessory classes and functions needed for the analysis of time series in the time-frequency space.

Installation
You can clone this folder with the command

git clone https://github.com/dottormale/Qtransform_torch.git
cd Qtransform_torch

Files Description

- **QTAM.py** : this script contains the implementation of the QTAM. The structure of the classes takes inspiration from the GWPy code, adapted for the new transform and ancillary functions. The class SingleQTransform computes the QTAM of a batch of 1D timeseries at a fixed value of q. 

- **QScan.py**: this script contains the classes and fuctions to compute the QTAMs for different values of q and with different windows.

- **QTAM_Example_Usage.ipynb**: this Jupyter notebook contains examples on how to use the QTAM on a real GW signal. It shows that the transform is invertible and can be used to perform de-noising on a signal containing glitches. It also shows examples on how to customise the transform to best fit the data that is being analysed.

- **glitch_for_testing.pkl**: file containig a 0.2 seconds long _Scattered Light_ glitch which is imported into QTAM_Example_Usage.ipynb to demontrate a de-noising application. 


## LICENSE
Copyright (c) [2025] [Lorenzo Asprea].

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License v3.0 as published by the Free Software Foundation.
