import numpy as np
import cv2
import matplotlib.pyplot as plt

def mask_with_opencv(image: np.ndarray, vertices: list) -> np.ndarray:
    """
    Create a mask for pixels inside a polygon using OpenCV.

    Parameters:
        image (np.ndarray): The input image as a 2D or 3D NumPy array.
        vertices (list): A list of (x, y) tuples representing the polygon vertices.

    Returns:
        np.ndarray: A 2D or 3D NumPy array containing the pixel values inside the polygon.
    """
    # Create a blank mask with the same height and width as the image
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint16)

    # Convert vertices to a NumPy array of integer type
    polygon = np.array([vertices], dtype=np.int32)

    # Fill the polygon on the mask
    cv2.fillPoly(mask, polygon, 1)


    # Apply the mask to the image
    if image.ndim == 3:  # For 3D images (e.g., multi-channel)
        mask = mask[:, :, np.newaxis]
        masked_image = mask * image
        mean = np.mean(image[mask.squeeze() == 1], axis=0).reshape(1, 1, -1)
        print('mean shape', (mean.shape))
        print('masked image shape', masked_image.shape)
        print('pixels inside', (image[mask.squeeze() == 1].shape))
    else:  # For 2D images (grayscale)
        masked_image = np.zeros_like(image)
        masked_image[mask == 1] = image[mask == 1]
        print('pixels inside', (image[mask==1]))

    return masked_image



# Example usage
if __name__ == "__main__":
    a = [[3, 0, 1, 3, 1, 1, 4, 5],
         [1, 1, 4, 1, 4, 2, 4, 6],
         [2, 2, 2, 2, 3, 0, 2, 7],
         [0, 1, 3, 4, 5, 2, 3, 8],
         [1, 4, 2, 1, 5, 0, 1, 4],
         [4, 1, 4, 0, 4, 4, 0, 3],
         [2, 4, 0, 2, 4, 0, 4, 2],
         [5, 6, 2, 3, 1, 7, 8, 6]]

    sub_a = [
        [1, 4, 1, 4],
        [2, 2, 2, 3],
        [1, 3, 4, 5],
        [4, 2, 1, 5]
    ]

    array = np.array(a, dtype=np.uint8)
    b = np.random.randint(0, 11, size=(8, 8, 84))

    # Define polygon coordinates
    polygon_coords = [(1, 1), (1, 4), (4, 4), (4, 1)]

    # Apply the mask
    masked_pixels = mask_with_opencv(b, polygon_coords)
    print("Masked Pixels Inside Polygon:", masked_pixels.shape)

    # Display the result
    plt.imshow(masked_pixels[:, :, 0], cmap='gray')
    plt.title("Masked Pixels Inside Polygon")
    plt.show()


    # Display the polygon on the array


    image_path = '/Volumes/Extreme Pro/pair_new/pair_data/box_1_class_B_num_1/before.npy'
    image = np.load(image_path).astype(np.uint16)  # Example 2D or 3D image

    polygon_coords = [(0, 0), (300, 0), (300, 300), (0, 300)]

    masked_pixels = mask_with_opencv(image, polygon_coords)
    print("Masked Pixels Inside Polygon:", masked_pixels.shape)

    # Display the result
    plt.imshow(masked_pixels[:, :, 400], cmap='gray')
    plt.title("Masked Pixels Inside Polygon")
    plt.show()