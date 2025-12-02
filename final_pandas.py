
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def read_and_filter_csv(csv_path):
    """
    Read a CSV file containing spectral data, filter wavelengths between 400-750nm,
    and calculate percentage matrix.

    Parameters:
        csv_path (str): Path to the CSV file containing spectral data

    Returns:
        dict: Dictionary with wavelengths and normalized spectrometer matrix
    """
    # Check if file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # First read the headers to get column names
    col_names = pd.read_csv(csv_path, nrows=0).columns

    # Filter column names
    v_columns = [col for col in col_names if "[V]" in col]
    nm_columns = [col for col in col_names if "[nm]" in col]

    # Validate columns exist
    if not v_columns or not nm_columns:
        raise ValueError("CSV does not contain expected columns with '[V]' or '[nm]'")

    # Read the CSV only once with just the columns we need
    df_selected = pd.read_csv(csv_path, usecols=v_columns + nm_columns).to_numpy()

    # Filter rows where wavelength is between 400-750nm
    filtered = df_selected[(df_selected[:, 0] > 400) & (df_selected[:, 0] < 750), :]

    # Extract spectrometer matrix (all columns except the first wavelength column)
    spectrometer_mat = filtered[:, 1:]

    # Calculate column sums for normalization (avoid division by zero)
    col_sums = np.sum(spectrometer_mat, axis=0)
    col_sums[col_sums == 0] = 1.0

    return {
        'wave_lengths': filtered[:, 0],
        'percentage_mat': spectrometer_mat / col_sums
    }


# Example usage
if __name__ == "__main__":
    csv_path = '/Volumes/Extreme Pro/20250605_121437_results_raw.csv'
    try:
        result = read_and_filter_csv(csv_path)
        print(f"Extracted {(result['wave_lengths'].shape)} wavelengths")
        print(f"Percentage matrix shape: {result['percentage_mat'].shape}")
    except Exception as e:
        print(f"Error: {e}")