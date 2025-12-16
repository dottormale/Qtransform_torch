# QTAM (Q-Transform Amplitude Modulation)

This folder contains a Jupyter Notebook which provides examples on how to use the **Q-Transform Amplitude Modulation (QTAM)**, together with two Python scripts containing the QTAM class and other accessory classes and functions needed for the analysis of time series in the time-frequency space.

## 📥 Installation

You can clone this repository with the following commands:

```bash
git clone https://github.com/dottormale/Qtransform_torch.git
cd Qtransform_torch
```
## 📝 Files Description

* **`QTAM.py`**
  This script contains the implementation of the QTAM. The structure of the classes takes inspiration from the *GWPy* code, adapted for the new transform and ancillary functions. The class `SingleQTransform` computes the QTAM of a batch of 1D timeseries at a fixed value of $q$.

* **`QScan.py`**
  This script contains the classes and functions to compute the QTAMs for different values of $q$ and with different windows.

* **`QTAM_Example_Usage.ipynb`**
  This Jupyter notebook contains examples on how to use the QTAM on a real GW signal. It shows that the transform is invertible and can be used to perform de-noising on a signal containing glitches. It also shows examples on how to customize the transform to best fit the data being analyzed.

* **`glitch_for_testing.pkl`**
  File containing a $0.2$-second long *Scattered Light* glitch which is imported into `QTAM_Example_Usage.ipynb` to demonstrate a de-noising application.

## 📄 License

**Copyright (c) 2025 Lorenzo Asprea.**

This program is free software: you can redistribute it and/or modify it under the terms of the **GNU General Public License v3.0** as published by the Free Software Foundation.






