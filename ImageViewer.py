import sys
import numpy as np
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
    Supports loading two NumPy arrays, selecting channels for visualization, measuring distances,
    and extracting pixel statistics.
    """

    def __init__(self):
        """Initialize the GUI and application state variables."""
        super().__init__()
        self.initUI()
        self.array1 = None  # First NumPy array
        self.array2 = None  # Second NumPy array
        self.points = []  # Stores clicked points for distance calculation
        self.selected_pixels = []  # Stores selected pixel values for statistics
        self.selected_channels = []  # Stores selected channels for visualization

    def initUI(self):
        """Set up the GUI layout and widgets."""
        self.setWindowTitle("NumPy Image Viewer")
        self.setWindowIcon(QIcon("icon.png"))
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        # Buttons for loading images
        self.load_button1 = QPushButton("Load NumPy Image File 1")
        self.load_button1.clicked.connect(lambda: self.load_file(1))
        layout.addWidget(self.load_button1)

        self.load_button2 = QPushButton("Load NumPy Image File 2")
        self.load_button2.clicked.connect(lambda: self.load_file(2))
        layout.addWidget(self.load_button2)

        # Labels to display image shapes
        self.shape_label1 = QLabel("Image 1 Shape: Not Loaded")
        layout.addWidget(self.shape_label1)

        self.shape_label2 = QLabel("Image 2 Shape: Not Loaded")
        layout.addWidget(self.shape_label2)

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

        self.setLayout(layout)

    def load_file(self, image_number):
        """Load a NumPy image file for the specified image number (1 or 2)."""
        file_path, _ = QFileDialog.getOpenFileName(self, f"Open NumPy File {image_number}", "", "NumPy Files (*.npy)")
        if file_path:
            array = np.load(file_path)
            if array.ndim == 3:
                if image_number == 1:
                    self.array1 = array
                    self.shape_label1.setText(f"Image 1 Shape: {array.shape}")
                else:
                    self.array2 = array
                    self.shape_label2.setText(f"Image 2 Shape: {array.shape}")
                self.populate_channel_selector()
            else:
                QMessageBox.warning(self, "Error", "Unsupported image format! Must be a 3D array.")

    def populate_channel_selector(self):
        """Populate the channel selector with available channels."""
        self.channel_selector.clear()
        if self.array1 is not None:
            # Add channels for image 1
            for i in range(self.array1.shape[-1]):
                self.channel_selector.addItem(QListWidgetItem(f"Image 1 - Channel {i}"))
        # if self.array2 is not None:
        #     # Add channels for image 2
        #     for i in range(self.array2.shape[-1]):
        #         self.channel_selector.addItem(QListWidgetItem(f"Image 2 - Channel {i}"))

    def show_image(self):
        """Display selected image channels."""
        if self.array1 is None or self.array2 is None:
            QMessageBox.warning(self, "Error", "Both images must be loaded!")
            return

        selected_items = self.channel_selector.selectedItems()
        selected_channels_image1 = []
        selected_channels_image2 = selected_channels_image1

        # Separate the selected channels for Image 1 and Image 2
        for item in selected_items:
            if "Image 1" in item.text():
                selected_channels_image1.append(int(item.text().split()[-1]))
                # selected_channels_image2.append(int(item.text().split()[-1]))
            # if "Image 2" in item.text():


        # Clear and update the axes
        self.ax1.clear()
        if selected_channels_image1:
            # Show selected channels for Image 1
            self.ax1.imshow(self.array1[:, :, selected_channels_image1].astype(np.uint16))
        else:
            self.ax1.imshow(self.array1[:, :, :3].astype(np.uint16))  # Show the first 3 channels if none selected
        self.ax1.set_title("Image 1")

        self.ax2.clear()
        if selected_channels_image2:
            # Show selected channels for Image 2
            self.ax2.imshow(self.array2[:, :, selected_channels_image2].astype(np.uint16))
        else:
            self.ax2.imshow(self.array2[:, :, :3].astype(np.uint16))  # Show the first 3 channels if none selected
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

        if len(self.points) == 2:
            self.calculate_distance()  # Automatically calculate distance when two points are selected

    def calculate_pixel_statistics(self):
        """Calculate statistics for the selected pixel."""
        if not self.points:
            QMessageBox.warning(self, "Error", "Select a point on the image!")
            return

        # Get the coordinates of the selected pixel
        x, y = self.points[-1]

        # Get the selected channels for image 1 and image 2
        selected_items = self.channel_selector.selectedItems()
        selected_channels_image1 = []
        selected_channels_image2 = []

        for item in selected_items:
            if "Image 1" in item.text():
                selected_channels_image1.append(int(item.text().split()[-1]))
            elif "Image 2" in item.text():
                selected_channels_image2.append(int(item.text().split()[-1]))

        # Calculate statistics for Image 1 at the selected pixel
        if selected_channels_image1:
            pixel_values_image1 = self.array1[int(y), int(x), selected_channels_image1]
        else:
            pixel_values_image1 = self.array1[int(y), int(x), :3]  # Default to first 3 channels

        # Calculate statistics for Image 2 at the selected pixel
        if selected_channels_image2:
            pixel_values_image2 = self.array2[int(y), int(x), selected_channels_image2]
        else:
            pixel_values_image2 = self.array2[int(y), int(x), :3]  # Default to first 3 channels

        # Combine pixel values from both images
        combined_pixel_values = np.concatenate([pixel_values_image1, pixel_values_image2])

        # Calculate statistics
        mean = np.mean(combined_pixel_values)
        median = np.median(combined_pixel_values)
        std_dev = np.std(combined_pixel_values)
        min_val = np.min(combined_pixel_values)
        max_val = np.max(combined_pixel_values)

        # Display the statistics
        QMessageBox.information(self, "Pixel Statistics",
                                f"Mean: {mean:.2f}\nMedian: {median:.2f}\nStandard Deviation: {std_dev:.2f}\n"
                                f"Min: {min_val:.2f}\nMax: {max_val:.2f}")

    def plot_pixel_values(self):
        """Plot the pixel values across the selected channels."""
        if not self.points:
            QMessageBox.warning(self, "Error", "Select a point on the image!")
            return

        # Get the coordinates of the selected pixel
        x, y = self.points[-1]

        # Get the selected channels for image 1 and image 2
        selected_items = self.channel_selector.selectedItems()
        selected_channels_image1 = []
        selected_channels_image2 = []

        for item in selected_items:
            if "Image 1" in item.text():
                selected_channels_image1.append(int(item.text().split()[-1]))
            elif "Image 2" in item.text():
                selected_channels_image2.append(int(item.text().split()[-1]))

        # Get pixel values
        pixel_values_image1 = self.array1[int(y), int(x), selected_channels_image1] if selected_channels_image1 else []
        pixel_values_image2 = self.array2[int(y), int(x), selected_channels_image2] if selected_channels_image2 else []

        # Combine pixel values for plotting
        combined_pixel_values = np.concatenate([pixel_values_image1, pixel_values_image2])
        combined_channels = list(range(1, len(combined_pixel_values) + 1))

        # Plot the pixel values
        plt.figure(figsize=(6, 4))
        plt.plot(combined_channels, combined_pixel_values, marker='o')
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
