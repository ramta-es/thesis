import numpy as np
import pandas as pd
import os
from typing import Tuple, Dict, Any

csv_path = '/Volumes/Extreme Pro/20250605_121437_results_raw.csv'




def create_test_data(image_size: tuple, spec_matrix_size: int, folder_name: str, csv_path: str = csv_path):
    """
    Create synthetic test data for a dataset viewer.

    Parameters:
        image_size (tuple): Size of the synthetic image (height, width, channels).
        spec_matrix_size (int): Number of spectral bands in the synthetic spectrometer matrix.
        folder_name (str): Name of the folder to save the synthetic data.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Synthetic image and spectrometer matrix.
    """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    df = pd.read_csv(csv_path)

    # Create a synthetic image
    image = np.random.rand(*image_size).astype(np.float32)
    print('Synthetic image shape:', image.shape)

    # Create a synthetic spectrometer matrix
    spec_df = df.iloc[0: (spec_matrix_size), 0:image.shape[0] + 2]

    # Save the synthetic data
    np.save(os.path.join(folder_name, 'synthetic_image'), image)
    spec_df.to_csv(os.path.join(folder_name, 'synthetic_spectrometer_matrix.csv'), index=False)

    print('saved synthetic data to', folder_name)


if __name__ == '__main__':
    # Example usage
    create_test_data(image_size=(5, 100, 100), spec_matrix_size=11, folder_name='/Volumes/Extreme Pro/test_data')
    print("Test data created successfully.")
    print(pd.read_csv('/Volumes/Extreme Pro/test_data/synthetic_spectrometer_matrix.csv'))
    print(pd.read_csv('/Volumes/Extreme Pro/test_data/synthetic_spectrometer_matrix.csv').columns)



