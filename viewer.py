import sys
import numpy as np
import os
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QFileDialog, QLabel, QVBoxLayout,
                             QMessageBox)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from pathlib import Path
import spectral as spy
from typing import Any


class NumpyViewerApp(QWidget):
    """
    A PyQt5-based GUI application to visualize and analyze NumPy image arrays.
    Supports loading multiple NumPy arrays from two files, selecting channels for visualization,
    and plotting pixel values.
    """

    def __init__(self):
        """Initialize the GUI and application state variables."""
        super().__init__()
        self.initUI()
        self.image1 = None  # Store tuple of (PNG image, RAW image)
        self.image2 = None  # Store NumPy array from file 2
        self.points = []  # Stores clicked points for distance calculation
        self.selected_channels = []  # Stores selected channels for visualization
        self.select_polygon = False  # Flag to indicate if polygon selection mode is on

    def initUI(self):
        """Set up the GUI layout and widgets."""
        self.setWindowTitle("NumPy Image Viewer")
        self.setWindowIcon(QIcon("icon.png"))
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        # Buttons for loading images from two files
        self.load_button1 = QPushButton("Load specim Image from Folder 1")
        self.load_button1.clicked.connect(lambda: self.load_file(1))
        layout.addWidget(self.load_button1)

        self.load_button2 = QPushButton("Load Hyper Spectral Image from File 2")
        self.load_button2.clicked.connect(lambda: self.load_file(2))
        layout.addWidget(self.load_button2)

        # Labels to display image shapes
        self.shape_label1 = QLabel("Image Shape File 1: Not Loaded")
        layout.addWidget(self.shape_label1)

        self.shape_label2 = QLabel("Image Shape File 2: Not Loaded")
        layout.addWidget(self.shape_label2)

        # Button to display selected channels
        self.show_image_button = QPushButton("Sow RGB Image")
        self.show_image_button.clicked.connect(self.show_image)
        layout.addWidget(self.show_image_button)

        # Matplotlib canvas for image display
        self.canvas = FigureCanvas(Figure())
        layout.addWidget(self.canvas)
        self.ax1 = self.canvas.figure.add_subplot(1, 2, 1)  # Left subplot for Image 1
        self.ax2 = self.canvas.figure.add_subplot(1, 2, 2)  # Right subplot for Image 2
        self.canvas.mpl_connect("button_press_event", self.on_click)

        # Button to plot pixel values across channels
        self.plot_pixel_button = QPushButton("Plot Pixel Values")
        self.plot_pixel_button.clicked.connect(self.plot_pixel_values)
        layout.addWidget(self.plot_pixel_button)

        # Button to toggle between selecting a polygon or a single point
        self.toggle_polygon_button = QPushButton("Select Polygon")
        self.toggle_polygon_button.clicked.connect(self.toggle_polygon_mode)
        layout.addWidget(self.toggle_polygon_button)

        self.setLayout(layout)

    def load_file(self, file_number):
        """Load a NumPy image file."""
        if file_number == 1:
            folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
            if folder_path:
                png_path, raw_image, hdr_file = self.open_image(Path(folder_path))
                if png_path and raw_image is not None:
                    print(f"PNG path: {png_path}, RAW image shape: {raw_image.shape}")
                    png_image = plt.imread(str(png_path))
                    self.image1 = (png_image, raw_image)
                    self.shape_label1.setText(f"Loaded PNG and RAW images from Folder 1 with shape {png_image.shape}.")
                else:
                    QMessageBox.warning(self, "Error", f"PNG or RAW file not found in {folder_path}!")
        else:
            file_path = QFileDialog.getOpenFileName(self, "Select File", "", "NumPy Files (*.npy)")[0]
            if file_path:
                array = np.load(file_path)
                if array.ndim == 3:
                    self.image2 = array
                    self.shape_label2.setText(f"Loaded image from File 2 with shape {self.image2.shape}.")
                else:
                    QMessageBox.warning(self, "Error", f"File {os.path.basename(file_path)} is not a 3D array!")

    @staticmethod
    def open_image(image_folder: Path) -> tuple[Any, Any, Any]:
        png_path = list((Path(image_folder).glob('*.png')))
        cap = Path(image_folder).joinpath('capture')
        if len(list(cap.glob('*.raw'))) > 0:
            raw_file = list(cap.glob('*.raw'))[0]
            hdr_file = list(cap.glob('*.hdr'))[0]
            spec_img = spy.io.envi.open(hdr_file.as_posix(), raw_file.as_posix())
            spec_img = spec_img[:, :, :]
            return str(png_path[0]), spec_img, hdr_file
        return None, None, None

    def show_image(self):
        """Display selected image channels."""
        if self.image1 is None and self.image2 is None:
            QMessageBox.warning(self, "Error", "No images loaded!")
            return

        # Clear and update the axes
        self.ax1.clear()
        self.ax2.clear()

        # Show the PNG image for the first image if available
        if self.image1 is not None:
            self.ax1.imshow(self.image1[0])
            self.ax1.set_title("Image 1 (PNG)")
        else:
            self.ax1.set_title("No Image 1")

        # Show the second image if available
        if self.image2 is not None:
            self.ax2.imshow(self.image2[:, :, :].astype(np.uint16))
            self.ax2.set_title("Image 2")
        else:
            self.ax2.set_title("No Image 2")

        self.canvas.draw()

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
                          [point[1] for point in self.points], 'b-')
        else:
            # Mark the selected point
            self.ax1.plot(event.xdata, event.ydata, 'ro')
            self.ax2.plot(event.xdata, event.ydata, 'bo')

        self.canvas.draw()

    def toggle_polygon_mode(self):
        """Toggle between single point and polygon selection mode."""
        self.select_polygon = not self.select_polygon
        self.points = []  # Reset points when switching modes
        mode = "polygon" if self.select_polygon else "single point"
        QMessageBox.information(self, "Mode Changed", f"Switched to {mode} selection mode.")

    def plot_pixel_values(self):
        """Plot the pixel values across the selected channels."""
        if not self.points:
            QMessageBox.warning(self, "Error", "Select a point on the image!")
            return

        # Get the coordinates of the selected pixel
        x, y = self.points[-1]

        # Get pixel values for both images
        pixel_values = []
        if self.image1 is not None:
            pixel_values.append(self.image1[1][int(y), int(x), :])
        if self.image2 is not None:
            pixel_values.append(self.image2[int(y), int(x), :])

        # Plot the pixel values
        plt.figure(figsize=(6, 4))
        for i, values in enumerate(pixel_values):
            plt.plot(values, marker='o', label=f'Image {i + 1}')
        plt.title(f"Pixel Value Plot at ({x}, {y})")
        plt.xlabel("Channel Number")
        plt.ylabel("Pixel Value")
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = NumpyViewerApp()
    viewer.show()
    sys.exit(app.exec_())