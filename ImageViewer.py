import sys
import numpy as np
import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from matplotlib.path import Path
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QFileDialog, QLabel, QVBoxLayout,
                             QHBoxLayout, QComboBox, QMessageBox, QListWidget, QListWidgetItem, QRadioButton, QButtonGroup)
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
        self.poly = []
        self.mean_0 = []
        self.mean_1 = []
        self.selected_pixels = []  # Stores selected pixel values for statistics
        self.selected_channels = []  # Stores selected channels for visualization
        self.mean = {}
        self.median = {}
        self.std_dev = {}
        self.min_val = {}
        self.max_val = {}
        self.sigma = {}
        self.select_polygon = False  # Flag to indicate if polygon selection mode is on

    def initUI(self):
        """Set up the GUI layout and widgets."""
        self.setWindowTitle("NumPy Image Viewer")
        self.setWindowIcon(QIcon("icon.png"))
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()
        button_layout = QHBoxLayout()

        # Buttons for loading images
        self.load_button = QPushButton("Load NumPy Images from Folder")
        self.load_button.clicked.connect(self.load_folder)
        button_layout.addWidget(self.load_button)

        # Labels to display image shapes
        self.shape_label = QLabel("Images Shape: Not Loaded")
        layout.addWidget(self.shape_label)

        # Radio buttons for options (horizontal layout)
        radio_layout = QHBoxLayout()
        self.radio_group = QButtonGroup(self)
        self.radio_option1 = QRadioButton("Average")
        self.radio_option2 = QRadioButton("Median")
        self.radio_option3 = QRadioButton("Standard Deviation")
        self.radio_option4 = QRadioButton("Sigma")
        self.radio_option5 = QRadioButton("Variance")

        self.radio_group.addButton(self.radio_option1)
        self.radio_group.addButton(self.radio_option2)
        self.radio_group.addButton(self.radio_option3)
        self.radio_group.addButton(self.radio_option4)
        self.radio_group.addButton(self.radio_option5)

        self.radio_option1.setChecked(True)

        radio_layout.addWidget(self.radio_option1)
        radio_layout.addWidget(self.radio_option2)
        radio_layout.addWidget(self.radio_option3)
        radio_layout.addWidget(self.radio_option4)
        radio_layout.addWidget(self.radio_option5)
        layout.addLayout(radio_layout)

        self.radio_group.buttonClicked.connect(self.on_radio_button_clicked)

        # Channel selector for visualization
        self.channel_selector = QListWidget()
        self.channel_selector.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        layout.addWidget(self.channel_selector)

        # Button to display selected channels
        self.show_image_button = QPushButton("Show Selected Channels")
        self.show_image_button.clicked.connect(self.show_image)
        button_layout.addWidget(self.show_image_button)

        # Button to calculate Euclidean distance
        self.distance_button = QPushButton("Calculate Distance")
        self.distance_button.clicked.connect(self.calculate_distance)
        button_layout.addWidget(self.distance_button)

        # Button to calculate Euclidean distance
        self.plot_spectrum_button = QPushButton("Plot Spectrum")
        self.plot_spectrum_button.clicked.connect(self.plot_spectrum)
        button_layout.addWidget(self.plot_spectrum_button)

        # Button to calculate pixel statistics
        self.pixel_stats_button = QPushButton("Calculate Pixel Statistics")
        self.pixel_stats_button.clicked.connect(self.calculate_pixel_statistics)
        button_layout.addWidget(self.pixel_stats_button)

        # Button to plot pixel values across channels
        self.plot_pixel_button = QPushButton("Plot Pixel Values")
        self.plot_pixel_button.clicked.connect(self.plot_pixel_values)
        button_layout.addWidget(self.plot_pixel_button)

        # Button to toggle between selecting a polygon or a single point
        self.toggle_polygon_button = QPushButton("Select Polygon")
        self.toggle_polygon_button.clicked.connect(self.toggle_polygon_mode)
        button_layout.addWidget(self.toggle_polygon_button)

        layout.addLayout(button_layout)

        # Matplotlib canvas for image display
        self.canvas = FigureCanvas(Figure())
        layout.addWidget(self.canvas)
        self.ax1 = self.canvas.figure.add_subplot(1, 3, 1)  # Left subplot for Image 1
        self.ax2 = self.canvas.figure.add_subplot(1, 3, 2)  # middel subplot for Image 2
        self.ax3 = self.canvas.figure.add_subplot(1, 3, 3)  # Right subplot for statistics
        self.canvas.mpl_connect("button_press_event", self.on_click)

        self.setLayout(layout)

    def on_radio_button_clicked(self, button):
        """Handle radio button selection."""
        QMessageBox.information(self, "Radio Button Selected", f"You selected: {button.text()}")

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
                if self.images[0].shape[0] * self.images[0].shape[1] > self.images[1].shape[0] * self.images[1].shape[1]:
                    self.images[0] = cv2.resize(self.images[0], (self.images[1].shape[0], self.images[1].shape[1]
                                                                 ), interpolation=cv2.INTER_LINEAR)
                else:
                    self.images[1] = cv2.resize(self.images[1], (
                    self.images[0].shape[0], self.images[0].shape[1]),
                                                interpolation=cv2.INTER_LINEAR)

                print('image 0 shape', self.images[0].shape)
                print('images 1 shape', self.images[1].shape)

                self.shape_label.setText(f"Loaded {len(self.images)} images.")
                self.populate_channel_selector()
            else:
                QMessageBox.warning(self, "Error", "No valid .npy files found in the folder.")

    def populate_channel_selector(self):
        """Populate the channel selector with available channels."""
        self.channel_selector.clear()
        if self.images:
            for i in range(self.images[0].shape[-1]):
                self.channel_selector.addItem(QListWidgetItem(f"Channel {i}"))

    def show_image(self):
        """Display selected image channels."""
        if not self.images:
            QMessageBox.warning(self, "Error", "No images loaded!")
            return

        selected_items = self.channel_selector.selectedItems()
        self.selected_channels = [int(item.text().split()[-1]) for item in selected_items]

        # Clear and update the axes
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()

        # Show selected channels for the first two images if available
        if len(self.images) > 0:
            self.ax1.imshow(self.images[0][:, :, self.selected_channels].astype(np.uint16))
        if len(self.images) > 1:
            self.ax2.imshow(self.images[1][:, :, self.selected_channels].astype(np.uint16))

        self.ax1.set_title("Image 1")
        self.ax2.set_title("Image 2")
        self.ax3.set_title("Statistics")
        self.canvas.draw()

    def plot_spectrum(self):
        """Plot the spectrum of pixel values across all channels."""
        self.ax3.clear()

        # Check if a polygon is selected and means are calculated
        if self.select_polygon and self.mean_0.size > 0 and self.mean_1.size > 0:
            pixel_values_0 = self.mean_0.squeeze()
            pixel_values_1 = self.mean_1.squeeze()
        else:
            # Ensure there are points selected
            if not self.points:
                QMessageBox.warning(self, "Error", "Select a point or polygon on the image!")
                return
            # FIXME: check if the points are in the image and normalized
            # Use the last selected point
            x, y = (int(self.points[-1][0]) ,int(self.points[-1][1]))
            print('max image shape', np.max(self.images[0], axis=0).shape)
            # pixel_values_0 = ((self.images[0][y, x, :] - np.min(np.min(self.images[0], axis=0), axis=0)) /
            #                   (10e-6 + np.max(np.max(self.images[0], axis=0), axis=0)) - np.min(np.min(self.images[0], axis=0), axis=0))
            # pixel_values_1 = ((self.images[1][y, x, :] - np.min(np.min(self.images[1], axis=0), axis=0)) /
            #                   (10e-6 + np.max(np.max(self.images[1], axis=0), axis=0)) - np.min(np.min(self.images[1], axis=0), axis=0))

            pixel_values_0 = (self.images[0][y, x, :]) / 4096
            pixel_values_1 = (self.images[1][y, x, :]) / 4096

        # Plot the spectrum for both images
        channels = np.arange(pixel_values_0.shape[0])
        self.ax3.plot(channels, pixel_values_0, label="Image 1", color="blue")
        self.ax3.plot(channels, pixel_values_1, label="Image 2", color="green")

        self.ax3.set_title("Spectrum of Pixel Values")
        self.ax3.set_xlabel("Channel")
        self.ax3.set_ylabel("Pixel Value")
        self.ax3.legend()
        self.ax3.grid(True)

        self.canvas.draw()


    def calculate_distance(self):
        """Calculate Euclidean distance between two selected points."""
        if len(self.points) < 2:
            QMessageBox.warning(self, "Error", "Select two points on the image!")
            return

        dist = euclidean(self.points[0], self.points[1])
        QMessageBox.information(self, "Distance", f"Euclidean Distance: {dist:.2f} pixels")



    @staticmethod
    def calculate_polygon_mean(image: np.ndarray, vertices: list) -> np.ndarray:
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
            median = (np.median(image[mask.squeeze() == 1], axis=0).reshape(1, 1, -1) /
                      np.max(image[mask.squeeze() == 1], axis=0).reshape(1, 1, -1))
        return median

    def on_click(self, event):
        """Handles the click event on the canvas to select points."""
        if event.inaxes != self.ax1 and event.inaxes != self.ax2:
            return

        self.points.append((event.xdata, event.ydata))
        print(f"Point selected: {event.xdata}, {event.ydata}")

        if self.select_polygon:
            self.ax1.plot([point[0] for point in self.points],
                          [point[1] for point in self.points], 'r-')
            self.ax2.plot([point[0] for point in self.points],
                          [point[1] for point in self.points], 'r-')
            self.canvas.draw()

            self.mean_0 = self.calculate_polygon_mean(self.images[0], self.points)
            self.mean_1 = self.calculate_polygon_mean(self.images[1], self.points)

        elif len(self.points) == 2:
            self.calculate_distance()

    def toggle_polygon_mode(self):
        """Toggle between single point and polygon selection mode."""
        self.select_polygon = not self.select_polygon
        mode = "polygon" if self.select_polygon else "single point"
        self.points = []
        QMessageBox.information(self, "Mode Changed", f"Switched to {mode} selection mode.")
    # FIXME: plot the statistics of the selected pixels
    def calculate_pixel_statistics(self):
        """Calculate statistics for the selected pixel."""
        if not self.points:
            QMessageBox.warning(self, "Error", "Select a point on the image!")
            return

        x, y = self.points[-1]
        selected_items = self.channel_selector.selectedItems()
        selected_channels = [int(item.text().split()[-1]) for item in selected_items]
        if self.mean_0 and self.mean_1:
            pixel_values = (self.mean_0, self.mean_1)
        else:
            pixel_values = (self.image1[int(y), int(x), :], self.image1[int(y), int(x), :])


        # combined_pixel_values = np.concatenate(pixel_values)

        self.mean = (np.mean(pixel_values[0], axis=2), np.mean(pixel_values[1], axis=2))
        self.median = (np.median(pixel_values[0], axis=2), np.median(pixel_values[1], axis=2))
        self.std_dev = (np.std(pixel_values[0], axis=2), np.std(pixel_values[1], axis=2))
        self.min_val = (np.min(pixel_values[0], axis=2), np.min(pixel_values[1], axis=2))
        self.max_val = (np.max(pixel_values[0], axis=2), np.max(pixel_values[1], axis=2))

        QMessageBox.information(self, "Pixel Statistics",
                                f"Mean: {self.mean:.2f}\n Median: {self.median:.2f}\n Standard Deviation: {self.std_dev:.2f}"
                                f"\nMin: {self.min_val:.2f}\nMax: {self.max_val:.2f}")

    def plot_pixel_values(self):
        """Plot the pixel values across the selected channels."""
        if not self.points:
            QMessageBox.warning(self, "Error", "Select a point on the image!")
            return

        x, y = self.points[-1]
        selected_items = self.channel_selector.selectedItems()
        selected_channels = [int(item.text().split()[-1]) for item in selected_items]

        pixel_values = []
        for image in self.images:
            pixel_values.append(image[int(y), int(x), :])

        combined_channels = list(range(840))

        plt.figure(figsize=(6, 4))
        plt.plot(combined_channels, pixel_values[0], marker='o', color='b')
        plt.plot(combined_channels, pixel_values[1], marker='o', color='g')
        plt.title(f"Pixel Value Plot at ({x}, {y})")
        plt.xlabel("Channel Number")
        plt.ylabel("Pixel Value")
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = NumpyViewerApp()
    viewer.show()
    sys.exit(app.exec_())