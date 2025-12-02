import sys
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.path import Path
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QFileDialog, QLabel, QVBoxLayout,
                             QHBoxLayout, QMessageBox, QListWidget, QListWidgetItem, QRadioButton, QButtonGroup)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
from scipy.spatial.distance import euclidean
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import cv2


class NumpyViewerApp(QWidget):
    """
    A PyQt5-based GUI application to visualize and analyze NumPy image arrays.
    Supports loading multiple NumPy arrays from a folder, selecting channels for visualization,
    measuring distances, and extracting pixel statistics.
    """

    def __init__(self):
        """Initialize the GUI and application state variables."""
        super().__init__()
        self.initUI()
        self.images = []  # List to store NumPy arrays
        self.points = []  # Stores clicked points for distance calculation
        self.selected_channels = []  # Stores selected channels for visualization
        self.select_polygon = False  # Flag to indicate if polygon selection mode is on

    def initUI(self):
        """Set up the GUI layout and widgets."""
        self.setWindowTitle("NumPy Image Viewer")
        self.setWindowIcon(QIcon("icon.png"))
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        # Horizontal layout for buttons
        button_layout = QHBoxLayout()

        # Buttons for various actions
        self.load_button = QPushButton("Load NumPy Images from Folder")
        self.load_button.clicked.connect(self.load_folder)
        button_layout.addWidget(self.load_button)

        self.show_image_button = QPushButton("Show Selected Channels")
        self.show_image_button.clicked.connect(self.show_image)
        button_layout.addWidget(self.show_image_button)

        self.distance_button = QPushButton("Calculate Distance")
        self.distance_button.clicked.connect(self.calculate_distance)
        button_layout.addWidget(self.distance_button)

        self.pixel_stats_button = QPushButton("Calculate Pixel Statistics")
        self.pixel_stats_button.clicked.connect(self.calculate_pixel_statistics)
        button_layout.addWidget(self.pixel_stats_button)

        self.plot_pixel_button = QPushButton("Plot Pixel Values")
        self.plot_pixel_button.clicked.connect(self.plot_pixel_values)
        button_layout.addWidget(self.plot_pixel_button)


        self.toggle_polygon_button = QPushButton("Select Polygon")
        self.toggle_polygon_button.clicked.connect(self.toggle_polygon_mode)
        button_layout.addWidget(self.toggle_polygon_button)

        layout.addLayout(button_layout)

        # Labels to display image shapes
        self.shape_label = QLabel("Images Shape: Not Loaded")
        layout.addWidget(self.shape_label)

        # Horizontal layout for three channel selectors
        channel_layout = QHBoxLayout()

        self.channel_selector1 = QListWidget()
        self.channel_selector1.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        channel_layout.addWidget(self.channel_selector1)

        self.channel_selector2 = QListWidget()
        self.channel_selector2.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        channel_layout.addWidget(self.channel_selector2)

        self.channel_selector3 = QListWidget()
        self.channel_selector3.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        channel_layout.addWidget(self.channel_selector3)

        layout.addLayout(channel_layout)

        # Matplotlib canvas for image display
        self.canvas = FigureCanvas(Figure())
        layout.addWidget(self.canvas)
        self.ax1 = self.canvas.figure.add_subplot(1, 3, 1)  # Left subplot for Image 1
        self.ax2 = self.canvas.figure.add_subplot(1, 3, 2)  # Right subplot for Image 2
        self.ax3 = self.canvas.figure.add_subplot(1, 3, 3)  # Right subplot for statistics
        self.canvas.mpl_connect("button_press_event", self.on_click)

        self.setLayout(layout)

    def load_folder(self):
        """Load all NumPy image files from the selected folder."""
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.images = []  # Clear existing images
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".npy"):
                    file_path = os.path.join(folder_path, file_name)
                    array = np.load(file_path)
                    if array.ndim == 3:
                        self.images.append(array)
                    else:
                        QMessageBox.warning(self, "Error", f"File {file_name} is not a 3D array!")
            if self.images:
                self.shape_label.setText(f"Loaded {len(self.images)} images.")
                self.populate_channel_selectors()
            else:
                QMessageBox.warning(self, "Error", "No valid .npy files found in the folder.")

    def populate_channel_selectors(self):
        """Populate the three channel selectors with available channels."""
        if not self.images:
            return

        total_channels = self.images[0].shape[-1]
        channels = list(range(total_channels))

        self.channel_selector1.clear()
        self.channel_selector2.clear()
        self.channel_selector3.clear()

        for channel in channels:
            self.channel_selector1.addItem(QListWidgetItem(f"Channel {channel}"))
            self.channel_selector2.addItem(QListWidgetItem(f"Channel {channel}"))
            self.channel_selector3.addItem(QListWidgetItem(f"Channel {channel}"))

    def show_image(self):
        """Display selected image channels."""
        if not self.images:
            QMessageBox.warning(self, "Error", "No images loaded!")
            return

        # Flatten the selected items from all three channel selectors
        selected_items = (
                self.channel_selector1.selectedItems() +
                self.channel_selector2.selectedItems() +
                self.channel_selector3.selectedItems()
        )

        # Extract the channel numbers from the selected items
        self.selected_channels = [int(item.text().split()[-1]) for item in selected_items]

        # Limit to 3 channels for RGB display
        if len(self.selected_channels) > 3:
            QMessageBox.warning(self, "Warning",
                                f"Selected {len(self.selected_channels)} channels. Only first 3 will be displayed as RGB.")
            self.selected_channels = self.selected_channels[:3]

        # Clear and update the axes
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()

        # Show selected channels for the first two images if available
        if len(self.images) > 0:
            img_data = self.images[0][:, :, self.selected_channels].astype(np.float32)
            # Normalize to 0-1 range for display
            img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())
            self.ax1.imshow(img_data)
        if len(self.images) > 1:
            img_data = self.images[1][:, :, self.selected_channels].astype(np.float32)
            img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())
            self.ax2.imshow(img_data)

        self.ax1.set_title("Image 1")
        self.ax2.set_title("Image 2")
        self.ax3.set_title("Statistics")
        self.canvas.draw()

    def calculate_distance(self):
        """Calculate Euclidean distance between two selected points."""
        if len(self.points) < 2:
            QMessageBox.warning(self, "Error", "Select two points on the image!")
            return

        dist = euclidean(self.points[0], self.points[1])
        QMessageBox.information(self, "Distance", f"Euclidean Distance: {dist:.2f} pixels")

    @staticmethod
    def calculate_polygon_mean(image: np.ndarray, vertices: list) -> float:
        """
        Calculate the mean pixel value inside a polygon in a given image.

        Parameters:
            image (np.ndarray): The input image as a 2D or 3D NumPy array.
            vertices (list): A list of (x, y) tuples representing the polygon vertices.

        Returns:
            float: The mean pixel value inside the polygon.
        """
        vertices_array = np.array(vertices)
        height, width = image.shape[:2]
        y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        coords = np.stack((x.ravel(), y.ravel()), axis=-1)
        polygon_path = Path(vertices_array)
        mask = polygon_path.contains_points(coords).reshape(height, width)
        if image.ndim == 3:
            mean_value = np.mean(image[mask], axis=0)
        else:
            mean_value = np.mean(image[mask])
        return mean_value

    def calculate_pixel_statistics(self):
        """Calculate statistics for the selected pixel."""
        if not self.points:
            QMessageBox.warning(self, "Error", "Select a point on the image!")
            return

        x, y = self.points[-1]
        pixel_values = []
        for image in self.images:
            pixel_values.append(image[int(y), int(x), :])

        combined_pixel_values = np.concatenate(pixel_values)

        mean = np.mean(combined_pixel_values)
        median = np.median(combined_pixel_values)
        std_dev = np.std(combined_pixel_values)
        min_val = np.min(combined_pixel_values)
        max_val = np.max(combined_pixel_values)

        QMessageBox.information(self, "Pixel Statistics",
                                f"Mean: {mean:.2f}\nMedian: {median:.2f}\nStandard Deviation: {std_dev:.2f}\n"
                                f"Min: {min_val:.2f}\nMax: {max_val:.2f}")

    def plot_pixel_values(self):
        """Plot the pixel values across the selected channels."""
        if self.select_polygon:
            if not self.points:
                QMessageBox.warning(self, "Error", "No points have been selected!")
                return

            # Save the points for later use
            self.saved_points = self.points.copy()
            self.points = []  # Clear the current points list
            QMessageBox.information(self, "Points Saved", f"Points saved: {self.saved_points}")
            print(f"Saved points: {self.saved_points}")

        if not self.points:
            QMessageBox.warning(self, "Error", "Select a point on the image!")
            return

        x, y = self.points[-1]
        pixel_values = []
        for image in self.images:
            pixel_values.append(image[int(y), int(x), :])

        # plt.figure(figsize=(6, 4))
        # for i, values in enumerate(pixel_values):
        #     plt.plot(values, marker='o', label=f"Image {i + 1}")
        # plt.title(f"Pixel Value Plot at ({x}, {y})")
        # plt.xlabel("Channel Number")
        # plt.ylabel("Pixel Value")
        # plt.legend()
        # plt.grid(True)
        # plt.show()

    def on_click(self, event):
        """Handles the click event on the canvas to collect points."""
        if event.inaxes != self.ax1 and event.inaxes != self.ax2:
            return

        if not self.select_polygon:
            QMessageBox.warning(self, "Error", "Polygon selection mode is not active!")
            return

        # Add the clicked point to the list
        self.points.append((event.xdata, event.ydata))
        print(f"Point selected: {event.xdata}, {event.ydata}")

        # Plot the points on the canvas
        self.ax1.plot([point[0] for point in self.points],
                      [point[1] for point in self.points], 'r-')
        self.ax2.plot([point[0] for point in self.points],
                      [point[1] for point in self.points], 'r-')
        self.canvas.draw()

        # elif len(self.points) == 2:
        #     self.calculate_distance()


        # self.mean_0 = self.calculate_polygon_mean(self.images[0], self.points)
        # self.mean_1 = self.calculate_polygon_mean(self.images[1], self.points)

    def save_points(self):
        """Stops adding points and saves the list of points."""
        if not self.points:
            QMessageBox.warning(self, "Error", "No points have been selected!")
            return

        # Save the points for later use
        self.saved_points = self.points.copy()
        self.points = []  # Clear the current points list
        QMessageBox.information(self, "Points Saved", f"Points saved: {self.saved_points}")
        print(f"Saved points: {self.saved_points}")


    # def pixels_in_polygon(self, vertices: list) -> tuple[np.ndarray, np.ndarray]:
    #     """
    #     Extract pixels inside a polygon from two 2D or 3D NumPy arrays stored in self.image1 and self.image2.
    #
    #     Parameters:
    #         vertices (list): A list of (x, y) tuples representing the polygon vertices.
    #
    #     Returns:
    #         tuple[np.ndarray, np.ndarray]: Two 2D NumPy arrays containing the pixel values inside the polygon
    #                                        for self.image1 and self.image2.
    #     """
    #     # Create a Path object for the polygon
    #     polygon_path = Path(vertices)
    #
    #     # Get the dimensions of the images (assumes both images have the same dimensions)
    #     height, width = self.image1.shape[:2]
    #
    #     # Create a grid of coordinates for the images
    #     y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    #     coords = np.stack((x.ravel(), y.ravel()), axis=-1)
    #
    #     # Create a mask for the pixels inside the polygon
    #     mask = polygon_path.contains_points(coords).reshape(height, width)
    #
    #     # Extract the pixels inside the polygon for both images
    #     if self.image1.ndim == 3:
    #         pixels_image1 = self.image1[mask, :]  # For 3D images
    #         pixels_image2 = self.image2[mask, :]  # For 3D images
    #     else:
    #         pixels_image1 = self.image1[mask]  # For 2D images
    #         pixels_image2 = self.image2[mask]  # For 2D images
    #
    #     return pixels_image1, pixels_image2



    def toggle_polygon_mode(self):
        """Toggle between single point and polygon selection mode."""
        self.select_polygon = not self.select_polygon
        mode = "polygon" if self.select_polygon else "single point"
        QMessageBox.information(self, "Mode Changed", f"Switched to {mode} selection mode.")
    @staticmethod
    def polygon_mean(image: np.ndarray, vertices: list) -> list:
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
            # masked_image = mask * image
            poly_mean = np.mean(image[mask.squeeze() == 1], axis=0).tolist()
        return poly_mean

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = NumpyViewerApp()
    viewer.show()
    sys.exit(app.exec_())