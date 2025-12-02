

import numpy as np
from pathlib import Path
from typing import Union, Any
import time
import os
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    GPU_AVAILABLE = cp.cuda.is_available()
except ImportError:
    CUPY_AVAILABLE = False
    GPU_AVAILABLE = False

print("CuPy installed:", CUPY_AVAILABLE)
print("CUDA GPU available:", GPU_AVAILABLE)
'''
if GPU_AVAILABLE:
    num_cuda = cp.cuda.runtime.getDeviceCount()
    print("Found CUDA devices:", [f"cuda:{i}" for i in range(num_cuda)])
    xp = cp
else:
    xp = np  # fallback to NumPy

def hyper_spec_calc(image_path: Union[str, Path],
                   spectrometer_mat: Union[np.ndarray, Any],
                   vector: Union[np.ndarray, Any]) -> Union[np.ndarray, Any]:
    if not isinstance(image_path, str):
        image_path = str(image_path)
    image = xp.load(image_path)  # Load the hyperspectral image
    if image.ndim == 3 and image.shape[0] < 100:
        image = xp.transpose(image, (1, 2, 0)) # WxHxC array
    percentage_mat = (spectrometer_mat / xp.sum(spectrometer_mat, axis=1))  # Normalize the spectrometer matrix
    result = image[:, :, None, :] * percentage_mat  # shape (H, W, 87, 1)

    # Sum across the last axis (channels)
    return xp.sum(result, axis=-1)  # shape (H, W, 87)
'''










