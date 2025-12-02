import numpy as np
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt
from tqdm import tqdm


def calculate_mahalanobis_distance(data_cloud: np.ndarray, measured_dist: np.ndarray) -> list:
    """
    Calculate Mahalanobis distance between two sets of pixels.

    Parameters:
        data_cloud (np.ndarray): First set of pixels with shape (N, C)
        measured_dist (np.ndarray): Second set of pixels with shape (N, C)

    Returns:
        float: Mean Mahalanobis distance across all pixels
    """

    # Calculate mean spectral signature of data cloud
    mean1 = np.mean(data_cloud, axis=0)

    # Calculate covariance matrix
    cov_matrix = np.cov(data_cloud.T)

    try:
        # Calculate inverse covariance matrix
        inv_cov = np.linalg.inv(cov_matrix)
        # Calculate Mahalanobis distance for each pixel in measured_dist
        distances = [mahalanobis(pixel, mean1, inv_cov) for pixel in measured_dist]
    # dist = np.mean(distances)
    # except np.linalg.LinAlgError:
    #     # If covariance matrix is singular, use pseudo-inverse
    #     print(f"Warning: Singular matrix, using pseudo-inverse")
    #     inv_cov = np.linalg.pinv(cov_matrix)
    #     diff = measured_dist - mean1
    #     distances = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))
    #     dist = np.mean(distances)

    return distances


def calculate_mahalanobis_exclude_channels(image1: np.ndarray, image2: np.ndarray) -> tuple:
    """
    Calculate Mahalanobis distance between two images by excluding one channel at a time.

    Parameters:
        image1 (np.ndarray): First image with shape (H, W, 840)
        image2 (np.ndarray): Second image with shape (H, W, 840)

    Returns:
        tuple: (distances array, channel indices)
            - distances: Array of 840 mean Mahalanobis distances
            - channels: Array of excluded channel indices [0, 1, ..., 839]
    """
    # Validate input shapes
    assert image1.shape == image2.shape, "Images must have the same shape!"
    assert image1.shape[-1] == 840, "Images must have 840 channels!"

    height, width, num_channels = image1.shape
    distances = np.zeros(num_channels)

    # Flatten images to (H*W, 840)
    data_cloud = image1.reshape(-1, num_channels)
    measured_dist = image2.reshape(-1, num_channels)

    print(f"Calculating Mahalanobis distances for {num_channels} channels...")

    # Iterate through each channel to exclude
    for excluded_channel in tqdm(range(num_channels)):
        # Create mask for all channels except the excluded one
        channel_mask = np.ones(num_channels, dtype=bool)
        channel_mask[excluded_channel] = False

        # Select 839 channels (excluding one)
        selected_data_cloud = data_cloud[:, channel_mask]
        selected_measured_dist = measured_dist[:, channel_mask]

        # Calculate Mahalanobis distance (returns list of per-pixel distances)
        pixel_distances = calculate_mahalanobis_distance(selected_data_cloud, selected_measured_dist)

        # Store the mean distance across all pixels
        distances[excluded_channel] = np.mean(pixel_distances)

    print("Calculation complete!")

    # Display results
    display_results(distances)

    return distances, np.arange(num_channels)


def display_results(distances: np.ndarray):
    """
    Display the calculated Mahalanobis distances.

    Parameters:
        distances (np.ndarray): Array of 840 distances
    """
    # Print statistics
    print("\n" + "=" * 60)
    print("MAHALANOBIS DISTANCE RESULTS")
    print("=" * 60)
    print(f"Total channels analyzed: {len(distances)}")
    print(f"Mean distance: {np.mean(distances):.4f}")
    print(f"Std deviation: {np.std(distances):.4f}")
    print(f"Min distance: {np.min(distances):.4f} (Channel {np.argmin(distances)})")
    print(f"Max distance: {np.max(distances):.4f} (Channel {np.argmax(distances)})")
    print("=" * 60)

    # Find top 10 and bottom 10 channels
    sorted_indices = np.argsort(distances)
    print("\nTop 10 channels with HIGHEST distance (removing these decreases similarity):")
    for i in range(10):
        idx = sorted_indices[-(i + 1)]
        print(f"  Channel {idx}: {distances[idx]:.4f}")

    print("\nTop 10 channels with LOWEST distance (removing these increases similarity):")
    for i in range(10):
        idx = sorted_indices[i]
        print(f"  Channel {idx}: {distances[idx]:.4f}")

    # Plot results
    plot_distances(distances)


def plot_distances(distances: np.ndarray):
    """
    Plot the Mahalanobis distances for all channels.

    Parameters:
        distances (np.ndarray): Array of 840 distances
    """
    plt.figure(figsize=(14, 6))

    # Main plot
    plt.subplot(1, 2, 1)
    plt.plot(distances, linewidth=1, alpha=0.7)
    plt.xlabel('Excluded Channel Index', fontsize=12)
    plt.ylabel('Mahalanobis Distance', fontsize=12)
    plt.title('Mahalanobis Distance vs Excluded Channel', fontsize=14)
    plt.grid(True, alpha=0.3)

    # Histogram
    plt.subplot(1, 2, 2)
    plt.hist(distances, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Mahalanobis Distance', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Distances', fontsize=14)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def analyze_images_from_paths(image1_path: str, image2_path: str, output_path: str = 'mahalanobis_distances.npy'):
    """
    Load two images and calculate Mahalanobis distances by excluding channels.

    Parameters:
        image1_path (str): Path to first .npy image file
        image2_path (str): Path to second .npy image file
        output_path (str): Path to save results (default: 'mahalanobis_distances.npy')

    Returns:
        tuple: (distances array, channel indices)
    """
    # Load images
    print(f"Loading image 1: {image1_path}")
    image1 = np.load(image1_path)
    print(f"  Shape: {image1.shape}")

    print(f"Loading image 2: {image2_path}")
    image2 = np.load(image2_path)
    print(f"  Shape: {image2.shape}")

    # Validate
    if image1.shape != image2.shape:
        raise ValueError(f"Shape mismatch: {image1.shape} vs {image2.shape}")

    if len(image1.shape) != 3:
        raise ValueError(f"Expected 3D arrays, got shape: {image1.shape}")

    # Calculate distances
    distances, channels = calculate_mahalanobis_exclude_channels(image1, image2)

    # Save results
    np.save(output_path, distances)
    print(f"\nResults saved to '{output_path}'")

    return distances, channels


# Example usage
if __name__ == "__main__":
    # Simple usage - provide two image paths
    distances, channels = analyze_images_from_paths(
        "/Volumes/Extreme Pro/pair_new/pair_data/box_4_class_A_num_36/before.npy",
        "/Volumes/Extreme Pro/pair_new/pair_data/box_5_class_D_num_20/before.npy"
    )

    # Or with custom output path:
    # distances, channels = analyze_images_from_paths(
    #     "/path/to/image1.npy",
    #     "/path/to/image2.npy",
    #     "my_custom_results.npy"
    # )
