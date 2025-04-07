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
from typing import Any, Dict


class NumpyViewerApp(QWidget):
    """
    A PyQt5-based GUI application to visualize and analyze NumPy image arrays.
    Supports loading a NumPy array from a file, selecting channels for visualization,
    and plotting pixel values.
    """

    def __init__(self):
        """Initialize the GUI and application state variables."""
        super().__init__()
        self.initUI()
        self.image1 = None  # Store tuple of (PNG image, RAW image)
        self.bands = None  # Store bands from the image
        self.points = []  # Stores clicked points for distance calculation
        self.selected_channels = []  # Stores selected channels for visualization
        self.select_polygon = False  # Flag to indicate if polygon selection mode is on

    def initUI(self):
        """Set up the GUI layout and widgets."""
        self.setWindowTitle("NumPy Image Viewer")
        self.setWindowIcon(QIcon("icon.png"))
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        # Button for loading image from a folder
        self.load_button1 = QPushButton("Load specim Image from Folder 1")
        self.load_button1.clicked.connect(lambda: self.load_file(1))
        layout.addWidget(self.load_button1)

        # Label to display image shape
        self.shape_label1 = QLabel("Image Shape File 1: Not Loaded")
        layout.addWidget(self.shape_label1)

        # Button to display selected channels
        self.show_image_button = QPushButton("Show RGB Image")
        self.show_image_button.clicked.connect(self.show_image)
        layout.addWidget(self.show_image_button)

        # Matplotlib canvas for image display
        self.canvas = FigureCanvas(Figure())
        layout.addWidget(self.canvas)
        self.ax1 = self.canvas.figure.add_subplot(1, 2, 1)  # Left subplot for Image 1
        self.ax2 = self.canvas.figure.add_subplot(1, 2, 2)  # Right subplot for pixel values
        self.canvas.mpl_connect("button_press_event", self.on_click)

        # Button to plot pixel values across channels
        self.plot_pixel_button = QPushButton("Plot Pixel Values")
        self.plot_pixel_button.clicked.connect(self.show_image)
        layout.addWidget(self.plot_pixel_button)

        # Button to toggle between selecting a polygon or a single point
        self.toggle_polygon_button = QPushButton("Select Polygon")
        self.toggle_polygon_button.clicked.connect(self.toggle_polygon_mode)
        layout.addWidget(self.toggle_polygon_button)

        self.setLayout(layout)

    def load_file(self, file_number):
        """Load a NumPy image file."""
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            images = self.open_image(Path(folder_path))
            if images is not None:
                print(f"PNG path: {images['image'][0]}, RAW image shape: {images['image'][1].shape}")
                png_image = plt.imread(str(images['image'][0]))
                self.image1 = (png_image, images['image'][1])
                self.bands = images['image'][1].bands.centers
                self.shape_label1.setText(f"Loaded PNG and RAW images from Folder 1 with shape {png_image.shape}.")
            else:
                QMessageBox.warning(self, "Error", f"PNG or RAW file not found in {folder_path}!")

    @staticmethod
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
                image_files = next(
                    (hdr, raw) for hdr, raw in files if 'DARKREF' not in hdr.name and 'WHITEREF' not in hdr.name)

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

    def show_image(self):
        """Display the PNG image and plot pixel values if available."""
        if self.image1 is None:
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

        # Plot pixel values if a point is selected
        if self.points:
            x, y = self.points[-1]
            pixel_values = self.image1[1][int(y), int(x), :].flatten()
            self.ax2.plot(self.bands, pixel_values, marker='o', markersize=2, label='Pixel Values')
            self.ax2.set_title(f"Pixel Values at ({x:.1f}, {y:.1f})")
            self.ax2.set_xlabel("Wavelength (nm)")
            self.ax2.set_ylabel("Pixel Value")
            self.ax2.legend()
        else:
            self.ax2.set_title("No Pixel Values")

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
        else:
            # Mark the selected point and add coordinates text
            self.ax1.plot(event.xdata, event.ydata, 'ro')
            self.ax1.text(event.xdata, event.ydata, f'({event.xdata:.1f}, {event.ydata:.1f})', color='white')

        self.canvas.draw()

    def toggle_polygon_mode(self):
        """Toggle between single point and polygon selection mode."""
        self.select_polygon = not self.select_polygon
        self.points = []  # Reset points when switching modes
        mode = "polygon" if self.select_polygon else "single point"
        QMessageBox.information(self, "Mode Changed", f"Switched to {mode} selection mode.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = NumpyViewerApp()
    viewer.show()
    sys.exit(app.exec_())