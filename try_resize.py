import numpy as np
import cv2


def resize_array(array: np.ndarray, target_shape: tuple, interpolation=cv2.INTER_LINEAR) -> np.ndarray:
    """
    Resize a NumPy array to the target shape.

    Parameters:
        array (np.ndarray): The input array to resize.
        target_shape (tuple): The desired shape (height, width).
        interpolation (int): Interpolation method (default is cv2.INTER_LINEAR).

    Returns:
        np.ndarray: The resized array.
    """
    if array.ndim == 2:  # Grayscale image
        resized = cv2.resize(array, (target_shape[1], target_shape[0]), interpolation=interpolation)
    elif array.ndim == 3:  # Multi-channel image
        resized = cv2.resize(array, (target_shape[1], target_shape[0]), interpolation=interpolation)
    else:
        raise ValueError("Input array must be 2D or 3D.")

    return resized


# Example usage
array = np.random.rand(100, 200, 3)  # Example 3D array
resized_array = resize_array(array, (50, 100))  # Resize to 50x100
print("Original shape:", array.shape)
print("Resized shape:", resized_array.shape)