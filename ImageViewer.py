import sys
import numpy as np
import os
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QFileDialog, QLabel, QVBoxLayout,
                             QHBoxLayout, QComboBox, QMessageBox, QListWidget, QListWidgetItem)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
from scipy.spatial.distance import euclidean
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


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

        # Buttons for loading images
        self.load_button = QPushButton("Load NumPy Images from Folder")
        self.load_button.clicked.connect(self.load_folder)
        layout.addWidget(self.load_button)

        # Labels to display image shapes
        self.shape_label = QLabel("Images Shape: Not Loaded")
        layout.addWidget(self.shape_label)

        # Channel selector for visualization
        self.channel_selector = QListWidget()
        self.channel_selector.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        layout.addWidget(self.channel_selector)

        # Button to display selected channels
        self.show_image_button = QPushButton("Show Selected Channels")
        self.show_image_button.clicked.connect(self.show_image)
        layout.addWidget(self.show_image_button)

        # Matplotlib canvas for image display
        self.canvas = FigureCanvas(Figure())
        layout.addWidget(self.canvas)
        self.ax1 = self.canvas.figure.add_subplot(1, 2, 1)  # Left subplot for Image 1
        self.ax2 = self.canvas.figure.add_subplot(1, 2, 2)  # Right subplot for Image 2
        self.canvas.mpl_connect("button_press_event", self.on_click)

        # Button to calculate Euclidean distance
        self.distance_button = QPushButton("Calculate Distance")
        self.distance_button.clicked.connect(self.calculate_distance)
        layout.addWidget(self.distance_button)

        # Button to calculate pixel statistics
        self.pixel_stats_button = QPushButton("Calculate Pixel Statistics")
        self.pixel_stats_button.clicked.connect(self.calculate_pixel_statistics)
        layout.addWidget(self.pixel_stats_button)

        # Button to plot pixel values across channels
        self.plot_pixel_button = QPushButton("Plot Pixel Values")
        self.plot_pixel_button.clicked.connect(self.plot_pixel_values)
        layout.addWidget(self.plot_pixel_button)

        # Button to toggle between selecting a polygon or a single point
        self.toggle_polygon_button = QPushButton("Select Polygon")
        self.toggle_polygon_button.clicked.connect(self.toggle_polygon_mode)
        layout.addWidget(self.toggle_polygon_button)

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

        # Show selected channels for the first two images if available
        if len(self.images) > 0:
            self.ax1.imshow(self.images[0][:, :, self.selected_channels].astype(np.uint16))
        if len(self.images) > 1:
            self.ax2.imshow(self.images[1][:, :, self.selected_channels].astype(np.uint16))

        self.ax1.set_title("Image 1")
        self.ax2.set_title("Image 2")
        self.canvas.draw()

    def calculate_distance(self):
        """Calculate Euclidean distance between two selected points."""
        if len(self.points) < 2:
            QMessageBox.warning(self, "Error", "Select two points on the image!")
            return

        dist = euclidean(self.points[0], self.points[1])
        QMessageBox.information(self, "Distance", f"Euclidean Distance: {dist:.2f} pixels")

    def on_click(self, event):
        """Handles the click event on the canvas to select points."""
        if event.inaxes != self.ax1 and event.inaxes != self.ax2:
            return

        # Record the coordinates of the click
        self.points.append((event.xdata, event.ydata))
        print(f"Point selected: {event.xdata}, {event.ydata}")

        # If in polygon mode, plot the polygon
        if self.select_polygon:
            self.ax1.plot([point[0] for point in self.points],
                          [point[1] for point in self.points], 'r-')
            self.ax2.plot([point[0] for point in self.points],
                          [point[1] for point in self.points], 'r-')
            self.canvas.draw()

        elif len(self.points) == 2:
            self.calculate_distance()  # Automatically calculate distance when two points are selected

    def toggle_polygon_mode(self):
        """Toggle between single point and polygon selection mode."""
        self.select_polygon = not self.select_polygon
        mode = "polygon" if self.select_polygon else "single point"
        QMessageBox.information(self, "Mode Changed", f"Switched to {mode} selection mode.")

    def calculate_pixel_statistics(self):
        """Calculate statistics for the selected pixel."""
        if not self.points:
            QMessageBox.warning(self, "Error", "Select a point on the image!")
            return

        # Get the coordinates of the selected pixel
        x, y = self.points[-1]

        # Get the selected channels for all images
        selected_items = self.channel_selector.selectedItems()
        selected_channels = [int(item.text().split()[-1]) for item in selected_items]

        # Calculate statistics for the selected pixel across all loaded images
        pixel_values = []
        for image in self.images:
            pixel_values.append(image[int(y), int(x), selected_channels])

        combined_pixel_values = np.concatenate(pixel_values)

        # Calculate statistics (mean, median, std_dev, etc.)
        self.mean = np.mean(combined_pixel_values)
        self.median = np.median(combined_pixel_values)
        self.std_dev = np.std(combined_pixel_values)
        self.min_val = np.min(combined_pixel_values)
        self.max_val = np.max(combined_pixel_values)

        # Display the statistics
        QMessageBox.information(self, "Pixel Statistics",
                                f"Mean: {self.mean:.2f}\n Median: {self.median:.2f}\n Standard Deviation: {self.std_dev:.2f}"
                                f"\nMin: {self.min_val:.2f}\nMax: {self.max_val:.2f}")

    def plot_pixel_values(self):
        """Plot the pixel values across the selected channels."""
        if not self.points:
            QMessageBox.warning(self, "Error", "Select a point on the image!")
            return

        # Get the coordinates of the selected pixel
        x, y = self.points[-1]

        # Get the selected channels for all images
        selected_items = self.channel_selector.selectedItems()
        selected_channels = [int(item.text().split()[-1]) for item in selected_items]

        # Get pixel values for all images
        pixel_values = []
        for image in self.images:
            pixel_values.append(image[int(y), int(x), :])

        # combined_pixel_values = np.concatenate(pixel_values)
        combined_channels = list(range(840))

        # Plot the pixel values
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
