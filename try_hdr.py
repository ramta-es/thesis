import spectral as spy
from pathlib import Path
from typing import Any, Dict
import matplotlib.pyplot as plt
import numpy as np
'''
# Path to the HDR file
hdr_file_path = '/Volumes/Extreme Pro/first_cube_2025-04-06_09-10-49/capture/first_cube_2025-04-06_09-10-49.hdr'
hdr_file_path2 ='/Users/ramtahor/Desktop/Vnir_Parsimons__hand_2021-02-21_13-04-16/capture/Vnir_Parsimons__hand_2021-02-21_13-04-16.hdr'
spy.settings.envi_support_nonlowercase_params = True
# Open the HDR file
'''
'''
hdr_image = spy.open_image(hdr_file_path2)

# Access the image data
image_data = hdr_image.load()

# Print some information about the image
print(f"Image shape: {image_data.shape}")
print(f"Data type: {image_data.dtype}")
print(f'Image data: {(hdr_image.bands.centers)}')
'''
'''
file_path = '/Volumes/Extreme Pro/first_cube_2025-04-06_09-10-49'
# def open_image(image_folder: Path) -> dict{'image': tuple[Any, Any], 'whiteref': tuple[Any, Any], 'blackref': tuple[Any, Any]}
#     png_path = list((Path(image_folder).glob('*.png')))
#     cap = Path(image_folder).joinpath('capture')
#     print('cap', cap)
#     if len(list(cap.glob('*.raw'))) > 0:
#         try:
#             raw_file = list(cap.glob('*.raw'))
#             hdr_file = list(cap.glob('*.hdr'))
#             print(f'raw file list{[i for i in raw_file]}')
#             print('hdr file list: ', list(cap.glob('*.hdr')))
#             spec_img = spy.io.envi.open(hdr_file.as_posix(), raw_file.as_posix())
#             print('spec image type: ', type(spec_img))
#         return {'image': str(png_path[0]), spec_img, 'whiteref': str(png_path[0]), spec_img, 'blackref': str(png_path[0]), spec_img}
#     return None, None


def open_image(image_folder: Path) -> Dict[str, Any]:
    png_files = list(image_folder.glob('*.png'))
    cap = image_folder / 'capture'
    raw_files = list(cap.glob('*.raw'))
    hdr_files = list(cap.glob('*.hdr'))

    if raw_files and hdr_files:
        files = list(zip(hdr_files, raw_files))
        try:
            darkref_files = next((hdr, raw) for hdr, raw in files if 'DARKREF' in hdr.name)
            whiteref_files = next((hdr, raw) for hdr, raw in files if 'WHITEREF' in hdr.name)
            image_files = next((hdr, raw) for hdr, raw in files if 'DARKREF' not in hdr.name and 'WHITEREF' not in hdr.name)

            spec_img = spy.io.envi.open(image_files[0].as_posix(), image_files[1].as_posix())
            darkref = spy.io.envi.open(darkref_files[0].as_posix(), darkref_files[1].as_posix())

            whiteref = spy.io.envi.open(whiteref_files[0].as_posix(), whiteref_files[1].as_posix())
            return {
                'image': (str(png_files[0]), spec_img),
                'whiteref': whiteref,
                'darkref': darkref
            }
        except:
            image_files = next(
                (hdr, raw) for hdr, raw in files if 'DARKREF' not in hdr.name and 'WHITEREF' not in hdr.name)
            spec_img = spy.io.envi.open(image_files[0].as_posix(), image_files[1].as_posix())

        return {
            'image': (str(png_files[0]), spec_img)
        }
    return None


def normalize_image(spectral_cube: Dict[str, Any]) -> np.ndarray:
    if 'darkref' in spectral_cube and 'whiteref' in spectral_cube:
        spec_img = spectral_cube['image'][1].load().astype(np.uint16)
        darkref = spectral_cube['darkref'].load().astype(np.uint16)
        whiteref = spectral_cube['whiteref'].load().astype(np.uint16)

        # Check if shapes are compatible
        if spec_img.shape != darkref.shape:
            darkref_median = np.median(darkref, axis=0)
            darkref = np.tile(darkref_median, (spec_img.shape[0], 1, 1))
        if spec_img.shape != whiteref.shape:
            whiteref_median = np.median(whiteref, axis=0)
            whiteref = np.tile(whiteref_median, (spec_img.shape[0], 1, 1))

        # Add a small epsilon to avoid division by zero
        epsilon = 1e-10
        normalized_data = (spec_img - darkref) / (whiteref - darkref + epsilon)
        return normalized_data
    else:
        raise KeyError("Missing 'darkref' or 'whiteref' in spectral_cube")





# def open_image_ref(image_folder: Path) -> tuple[Any, Any]:
#     png_path = list((Path(image_folder).glob('*.png')))
#     cap = Path(image_folder).joinpath('capture')
#     if len(list(cap.glob('*.raw'))) > 0:
#         raw_file = list(cap.glob('*.raw'))[0]
#         hdr_file = list(cap.glob('*.hdr'))[0]
#         spec_img = spy.io.envi.open(hdr_file.as_posix(), raw_file.as_posix())
#         print('spec image type: ', type(spec_img))
#         return str(png_path[0]), spec_img
#     return None, None






spectral_cube = open_image(Path(file_path))
# normed = normalize_image(spectral_cube)
# png_path, spec_img = open_image_ref(Path(file_path))
# print(type(spectral_cube['image'][1].bands.centers[0]))
# print('bands:', type(spectral_cube['image'][1].metadata.get['Wavelength'][0]))
# normed = (spectral_cube['image'][1] - spectral_cube['darkref']) / (spectral_cube['whiteref'] - spectral_cube['image'][1])
print((spectral_cube['darkref'][0]))
# print((spectral_cube['image'][0]))

# Print some information about the image
# print(f"Image shape: {spec_img[:, :, [1, 2]].transpose().shape}")
# print(f"Data type: {type(spec_img)}")
# print(f'Image data: {len(spec_img.bands.centers)}')
# print(type(normed))
# plt.imshow(normed[:, :, [100, 150, 220]]), plt.show()
'''



'''
total_channels = 840
channels = list(range(total_channels))

# Split channels into three groups
group1 = channels[::3]
group2 = channels[1::3]
group3 = channels[2::3]
print('channels', [[f'test{i}__{j}' for i in range(3)] for j in range(4)])




import numpy as np
from matplotlib.path import Path

image_path = '/Volumes/Extreme Pro/pair_new/pair_data/box_1_class_B_num_1/before.npy'



import numpy as np
from matplotlib.path import Path
import matplotlib.pyplot as plt

import numpy as np
from matplotlib.path import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

import numpy as np
from matplotlib.path import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def extract_pixels_in_polygon(image: np.ndarray, vertices: list) -> np.ndarray:
    """
    Extract pixels inside a polygon from a 2D or 3D NumPy array and display the polygon as a separate image.

    Parameters:
        image (np.ndarray): The input image as a 2D or 3D NumPy array.
        vertices (list): A list of (x, y) tuples representing the polygon vertices.

    Returns:
        np.ndarray: A 2D or 3D NumPy array containing the pixel values inside the polygon.
    """
    # Create a Path object for the polygon
    polygon_path = Path(vertices)

    # Get the dimensions of the image
    height, width = image.shape[:2]
    print('height, width', height, width)

    # Create a grid of coordinates for the image
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    coords = np.stack((x.ravel(), y.ravel()), axis=-1)
    print('coords', coords)

    # Create a mask for the pixels inside the polygon
    mask = polygon_path.contains_points(coords).reshape(height, width)

    # Create a new image to display the polygon
    polygon_image = np.zeros((height, width), dtype=np.uint8)
    polygon_image[mask] = 255  # Fill the polygon area with white (255)

    # Display the polygon as a separate image
    plt.imshow(polygon_image, cmap='gray')
    plt.title("Polygon as a Separate Image")
    plt.show()

    # Extract the pixels inside the polygon
    if image.ndim == 3:
        return image[mask]  # For 3D images (e.g., RGB or multi-channel)
    else:
        return image[mask]  # For 2D images (grayscale)

# Example usage
image_path = '/Volumes/Extreme Pro/pair_new/pair_data/box_1_class_B_num_1/before.npy'
image = np.load(image_path)  # Example 2D or 3D image
vertices = [(0, 0), (0, 300), (400, 200), (200, 0)]  # Define the polygon vertices
pixels_in_polygon = extract_pixels_in_polygon(image, vertices)

# Display the extracted pixels
plt.imshow(pixels_in_polygon, cmap='gray' if image.ndim == 2 else None)
plt.title("Extracted Pixels Inside Polygon")
plt.show()
'''




'''
from shapely.geometry import Polygon, Point
import numpy as np

import numpy as np
from shapely.geometry import Point, Polygon

def print_points_inside_polygon(points_array, polygon):
    """
    Prints all points from a NumPy array that are strictly inside the given polygon.

    Parameters:
    - points_array: NumPy array of shape (N, 2), where each row is a point (x, y).
    - polygon: Shapely Polygon object.
    """
    for point_coords in points_array:
        # Ensure point_coords is a 1D array with 2 elements
        # if point_coords.shape[0] != 2:
        #     raise ValueError(f"Each point must have 2 elements. Got shape {point_coords.shape}")
        point = Point(point_coords)

        if polygon.contains(point):
            print(f"Point inside polygon: ({point.x}, {point.y})")

# Example usage:
if __name__ == "__main__":
    # # Create a 20x20 array with random integers from 0 to 9
    # random_array = np.random.randint(0, 5, size=(7, 7))
    #
    # print(random_array)
    # Define polygon coordinates
    a = [[3, 0, 1, 3, 1, 1, 4],
         [1, 1, 4, 1, 4 ,2 ,4],
         [2, 2, 2, 2, 3, 0, 2],
         [0, 1, 3, 4, 0, 2, 3],
         [1, 4, 2, 1, 0, 0, 1],
         [4, 1, 4, 0, 4, 4, 0],
         [2, 4, 0, 2, 4, 0, 4]]

    sub_a = [
             [1, 4, 1, 4],
             [2, 2, 2, 3],
             [1, 3, 4, 0],
             [4, 2, 1, 0]
             ]

    # Extract all indices as (x, y) pairs



    # Define the polygon
    polygon_coords = [(1, 1), (4, 1), (4, 4), (1, 4)]
    polygon = Polygon(polygon_coords)

    # Print points inside the polygon
    print_points_inside_polygon(a, polygon)
'''

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from matplotlib.patches import Polygon as MplPolygon

def display_array_and_polygon_separately(array: np.ndarray, polygon_coords: list):
    """
    Displays a NumPy array and a polygon in separate subplots.

    Parameters:
        array (np.ndarray): The input 2D NumPy array to display.
        polygon_coords (list): A list of (x, y) tuples representing the polygon vertices.
    """
    # Create a Shapely Polygon object
    polygon = Polygon(polygon_coords)
    print('polygon', polygon)

    # Check if the polygon is valid
    if not polygon.is_valid:
        raise ValueError("The provided polygon coordinates do not form a valid polygon.")

    # Create the figure and subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Display the array in the first subplot
    axes[0].imshow(array, cmap='gray', origin='upper')
    axes[0].set_title("Array Display")
    axes[0].axis('off')

    # Display the polygon in the second subplot
    axes[1].set_xlim(0, array.shape[1])
    axes[1].set_ylim(array.shape[0], 0)
    mpl_polygon = MplPolygon(polygon_coords, closed=True, edgecolor='red', facecolor='none', linewidth=2)
    axes[1].add_patch(mpl_polygon)
    axes[1].set_title("Polygon Display")
    axes[1].axis('off')

    # Show the plots
    plt.tight_layout()
    plt.show()

from matplotlib.path import Path


def print_pixels_in_polygon(image: np.ndarray, vertices: list):
    """
    Prints all pixel values inside a polygon from a 2D or 3D NumPy array.

    Parameters:
        image (np.ndarray): The input image as a 2D or 3D NumPy array.
        vertices (list): A list of (x, y) tuples representing the polygon vertices.
    """
    # Create a Path object for the polygon
    polygon_path = Path(vertices)

    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Create a grid of coordinates for the image
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    coords = np.stack((x.ravel(), y.ravel()), axis=-1)

    # Create a mask for the pixels inside the polygon
    mask = polygon_path.contains_points(coords).reshape(height, width)
    plt.imshow(mask, cmap='gray', origin='upper')
    plt.title("Mask of Pixels bla bla"), plt.show()

    # Extract the pixel values inside the polygon
    if image.ndim == 3:  # For 3D images (e.g., multi-channel)
        # Apply the mask to each channel and reshape to maintain spatial structure
        mask = polygon_path.contains_points(coords).reshape(height, width, 840)
        pixels = image[mask, :].reshape(-1, mask.sum(), image.shape[2])
        print(f"Pixels shape: {pixels.shape}")
        # Display the mask
        plt.imshow(mask, cmap='gray', origin='upper')
        plt.title("Mask of Pixels Inside Polygon 3D")
        plt.show()
    else:  # For 2D images (grayscale)
        pixels = image[mask]
        print(f"Pixels inside the polygon:\n{pixels}")
        # Display the mask
        plt.imshow(mask, cmap='gray', origin='upper')
        plt.title("Mask of Pixels Inside Polygon")
        plt.show()

    return pixels

# Example usage
if __name__ == "__main__":
    a = [[3, 0, 1, 3, 1, 1, 4, 5],
         [1, 1, 4, 1, 4, 2, 4, 6],
         [2, 2, 2, 2, 3, 0, 2, 7],
         [0, 1, 3, 4, 0, 2, 3, 8],
         [1, 4, 2, 1, 0, 0, 1, 4],
         [4, 1, 4, 0, 4, 4, 0, 3],
         [2, 4, 0, 2, 4, 0, 4, 2],
         [5, 6, 2, 3, 1, 7, 8, 6]]

    sub_a = [
        [1, 4, 1, 4],
        [2, 2, 2, 3],
        [1, 3, 4, 0],
        [4, 2, 1, 0]
    ]



    # Create a sample 2D NumPy array
    array = np.array(a)
    print(array.shape)

    # Define polygon coordinates
    polygon_coords = [(0, 0), (2, 0), (4, 4), (0, 8)]

    # Display the polygon on the array
    # display_array_and_polygon_separately(array, polygon_coords)
    print_pixels_in_polygon(array, polygon_coords)
    image_path = '/Volumes/Extreme Pro/pair_new/pair_data/box_1_class_B_num_1/before.npy'
    image = np.load(image_path)  # Example 2D or 3D image

    polygon_coords = [(0, 0), (300, 0), (300, 300), (300, 0)]
    print_pixels_in_polygon(image, polygon_coords)

