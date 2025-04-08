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
spy.settings.envi_support_nonlowercase_params = True


class NumpyViewerApp(QWidget):
    """
    A PyQt5-based GUI application to visualize and analyze NumPy image arrays.
    Supports loading a NumPy array from a file, selecting channels for visualization,
    and plotting pixel values.
    """

    def __init__(self):
        """Initialize the GUI and application state variables."""
        super().__init__()
        self.images = None  # Store loaded images
        self.image1 = None  # Store tuple of (PNG image, RAW image)
        self.original_image1 = None  # Store the original image for reverting
        self.whiteref = None  # Store reference images
        self.darkref = None  # Store reference images
        self.bands = None  # Store bands from the image
        self.points = []  # Stores clicked points for distance calculation
        self.selected_channels = []  # Stores selected channels for visualization
        self.select_polygon = False  # Flag to indicate if polygon selection mode is on
        self.is_normalized = False  # Flag to track if the image is normalized
        self.initUI()

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

        # Combined button for normalizing and reverting the image
        self.norm_revert_button = QPushButton("Normalize Image")
        self.norm_revert_button.clicked.connect(self.toggle_norm_revert)
        layout.addWidget(self.norm_revert_button)

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
            self.images = self.open_image(Path(folder_path))
            if self.images is not None:
                print(f"PNG path: {self.images['image'][0]}, RAW image shape: {self.images['image'][1].shape}")
                png_image = plt.imread(str(self.images['image'][0]))
                self.image1 = (png_image, self.images['image'][1])
                self.original_image1 = (png_image, self.images['image'][1])  # Save the original image
                self.bands = [float(band) for band in self.images['image'][1].metadata['Wavelength']]
                self.shape_label1.setText(f"Loaded PNG and RAW images from Folder 1 with shape {png_image.shape}.")
            else:
                QMessageBox.warning(self, "Error", f"PNG or RAW file not found in {folder_path}!")
            if self.images['whiteref'] is not None and self.images['darkref'] is not None:
                self.whiteref = self.images['whiteref']
                self.darkref = self.images['darkref']

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

        # Plot pixel values for all selected points
        if self.points and self.bands is not None:
            for i, (x, y) in enumerate(self.points):
                pixel_values = self.image1[1][int(y), int(x), :].flatten()
                self.ax2.plot(self.bands, pixel_values, marker='o', markersize=2, label=f'Point {i+1} ({x:.1f}, {y:.1f})')

            self.ax2.set_title("Pixel Values for Selected Points")
            self.ax2.set_xlabel("Wavelength (nm)")
            self.ax2.set_ylabel("Pixel Value")
            self.ax2.legend(prop={'size': 5})
        else:
            self.ax2.set_title("No Pixel Values")

        self.canvas.draw()

    def toggle_norm_revert(self):
        """Toggle between normalizing the image and reverting to the original."""
        if self.image1 is None:
            QMessageBox.warning(self, "Error", "No images loaded!")
            return

        if self.is_normalized:
            # Revert to the original image
            self.image1 = self.original_image1
            self.norm_revert_button.setText("Normalize Image")
            self.is_normalized = False
        else:
            # Normalize the image
            if self.darkref is None or self.whiteref is None:
                QMessageBox.warning(self, "Error", "Missing 'darkref' or 'whiteref' in metadata")
                return

            spec_img = self.image1[1].load().astype(np.float16)
            darkref = self.darkref.load().astype(np.float16)
            whiteref = self.whiteref.load().astype(np.float16)

            # Check if shapes are compatible
            if spec_img.shape != darkref.shape:
                darkref_median = np.median(darkref, axis=0)
                darkref = np.tile(darkref_median, (spec_img.shape[0], 1, 1))
            if spec_img.shape != whiteref.shape:
                whiteref_median = np.median(whiteref, axis=0)
                whiteref = np.tile(whiteref_median, (spec_img.shape[0], 1, 1))

            # Add a small epsilon to avoid division by zero
            epsilon = 1e-6
            normalized_data = (spec_img - darkref) / (whiteref - darkref + epsilon)

            # Update the image with normalized data
            self.image1 = (self.image1[0], normalized_data.astype(np.float16))
            self.norm_revert_button.setText("Revert to Original Image")
            self.is_normalized = True

        self.show_image()

    def on_click(self, event):
        """Handles the click event on the canvas to select points."""
        if event.inaxes != self.ax1:
            return

        # Record the coordinates of the click
        self.points.append((event.xdata, event.ydata))

        print(f"Point selected: {event.xdata}, {event.ydata}")

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