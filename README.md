# Qtransform_torch

**Qtransform\_torch** is a global repository containing two distinct yet related projects focused on advanced signal processing and time-series analysis: the **Q-Transform with Amplitude Modulation (QTAM)** implementation and tools for **Spline Interpolation**.

## 🚀 Quick Start

To begin working with the code in this repository, clone it using the following command:

```bash
git clone [https://github.com/dottormale/Qtransform_torch.git](https://github.com/dottormale/Qtransform_torch.git)
cd Qtransform_torch
```


The repository is organized into the following main subfolders:

* **`/QTAM`**: Implementation of the Q-Transform with Amplitude Modulation for time-frequency analysis.
* **`/Spline_interpolation`**: Code related to spline interpolation techniques.

---

## 📂 Subproject: QTAM (Q-Transform Amplitude Modulation)

Located in the `QTAM/` folder, this project contains the core implementation and examples for the Q-Transform with Amplitude Modulation (QTAM). This is an advanced method for analyzing time-series data in the time-frequency domain and is particularly useful for signals like Gravitational Waves (GWs).

### 📝 Files Description

| File | Description |
| :--- | :--- |
| **`QTAM.py`** | Contains the main implementation of the `QTAM` class. The structure is inspired by *GWPy*, adapted for the transform and ancillary functions. The `SingleQTransform` class computes the QTAM of a batch of 1D timeseries at a fixed $q$ value. |
| **`QScan.py`** | Contains classes and functions necessary to compute the QTAMs for different values of the quality factor $q$ and various windowing functions. |
| **`QTAM_Example_Usage.ipynb`** | A Jupyter notebook demonstrating how to use QTAM on a real GW signal. It shows applications like the invertibility of the transform, de-noising of signals containing glitches, and customization of the transform parameters. |
| **`glitch_for_testing.pkl`** | A $0.2$-second long Scattered Light glitch file used by the example notebook to demonstrate de-noising capabilities. |

### ✨ Key Applications

The QTAM implementation is highly effective for tasks that require isolating transient features in noisy data, such as:

* **De-noising:** Cleaning signals that contain glitches.
* **Analysis:** Advanced time-frequency localization.

---

## 📂 Subproject: Spline_interpolation

Located in the `Spline_interpolation/` folder, this subproject hosts code related to general spline interpolation techniques.

> **ℹ️ Note:** Please refer to the `README.md` file inside the `Spline_interpolation/` subfolder for specific details on the files and usage instructions.

---

## 📄 License

Both subprojects in this repository are open source and licensed under the **GNU General Public License v3.0**.

**Copyright (c) 2025 Lorenzo Asprea.**

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License v3.0 as published by the Free Software Foundation.

See the `LICENSE` file in the repository root for the full text.
