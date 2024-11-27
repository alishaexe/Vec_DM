#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 11:12:26 2024

@author: alisha
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

# Example data: an oscillatory signal
x = np.linspace(0, 10, 1000)
y = np.sin(4*x) * np.exp(-0.1 * x)  # Damped oscillation

# 1. Find peaks
peaks, _ = find_peaks(y)

# 2. Get peak values and their corresponding x-values
x_peaks = x[peaks]
y_peaks = y[peaks]

# 3. Interpolate between peaks using spline interpolation
f = interp1d(x_peaks, y_peaks, kind='linear', fill_value="extrapolate")

# 4. Create a smooth curve using the interpolated function
y_smooth = f(x)

# Plot the original data and the smooth curve over the peaks
plt.plot(x, y, label="Oscillatory Response", alpha=0.6)
plt.plot(x, y_smooth, 'r--', label="Smooth Curve over Peaks")
plt.scatter(x_peaks, y_peaks, color='red', label="Detected Peaks")
plt.legend()
plt.show()
