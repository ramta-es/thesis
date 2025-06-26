import sys
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QFileDialog, QLabel, QVBoxLayout,
                             QHBoxLayout, QComboBox, QMessageBox, QListWidget, QListWidgetItem, QRadioButton,
                             QButtonGroup)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from pathlib import Path

from typing import Any, Dict

from pathlib import Path
from typing import Union, Any
import time
import os

try:
    import cupy as cp

    CUPY_AVAILABLE = True
    GPU_AVAILABLE = cp.cuda.is_available()
except ImportError:
    CUPY_AVAILABLE = False
    GPU_AVAILABLE = False

print("CuPy installed:", CUPY_AVAILABLE)
print("CUDA GPU available:", GPU_AVAILABLE)

if GPU_AVAILABLE:
    num_cuda = cp.cuda.runtime.getDeviceCount()
    print("Found CUDA devices:", [f"cuda:{i}" for i in range(num_cuda)])
    xp = cp
else:
    xp = np  # fallback to NumPy


class SpectralWorker(QObject):
    """Worker to perform spectral calculations in a separate thread."""
    finished = pyqtSignal(object)  # Signal emitting the result
    progress = pyqtSignal(int)  # Signal for progress updates
    error = pyqtSignal(str)  # Signal for errors

    def __init__(self, image_path, percentage_mat):
        super().__init__()
        self.image_path = image_path
        self.percentage_mat = percentage_mat

    def run(self):
        """Run the hyperspectral calculation."""
        try:
            # Perform calculation in the worker thread
            result = hyper_spec_calc(self.image_path, self.percentage_mat)

            # Move data back to CPU if using GPU
            if GPU_AVAILABLE:
                result = cp.asnumpy(result)
                # Free GPU memory
                mempool = cp.get_default_memory_pool()
                pinned_mempool = cp.get_default_pinned_memory_pool()
                mempool.free_all_blocks()
                pinned_mempool.free_all_blocks()

            self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))


def read_and_filter_csv(csv_path):
    """
    Read a CSV file containing spectral data, filter wavelengths between 400-750nm,
    and calculate percentage matrix.

    Parameters:
        csv_path (str): Path to the CSV file containing spectral data

    Returns:
        dict: Dictionary with wavelengths and normalized spectrometer matrix
    """
    # Check if file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # First read the headers to get column names
    col_names = pd.read_csv(csv_path, nrows=0).columns

    # Filter column names
    v_columns = [col for col in col_names if "[V]" in col]
    nm_columns = [col for col in col_names if "[nm]" in col]

    # Validate columns exist
    if not v_columns or not nm_columns:
        raise ValueError("CSV does not contain expected columns with '[V]' or '[nm]'")

    # Read the CSV only once with just the columns we need
    df_selected = pd.read_csv(csv_path, usecols=v_columns + nm_columns).to_numpy()

    # Filter rows where wavelength is between 400-750nm
    filtered = df_selected[(df_selected[:, 0] > 400) & (df_selected[:, 0] < 750), :]

    # Extract spectrometer matrix (all columns except the first wavelength column)
    spectrometer_mat = filtered[:, 1:]

    # Calculate column sums for normalization (avoid division by zero)
    col_sums = np.sum(spectrometer_mat, axis=0)
    col_sums[col_sums == 0] = 1.0

    return {
        'wave_lengths': filtered[:, 0],
        'percentage_mat': spectrometer_mat / col_sums
    }


def hyper_spec_calc(image_path: Union[str, Path],
                    percentage_mat: Union[np.ndarray, Any]) -> np.ndarray:
    if not isinstance(image_path, str):
        image_path = str(image_path)

    # Load image with NumPy first to check dimensions
    image_np = np.load(image_path)

    # Transpose if needed
    if image_np.ndim == 3 and image_np.shape[0] < 100:
        image_np = np.transpose(image_np, (1, 2, 0))
        print(f"Image shape after transpose: {image_np.shape}")

    # Get dimensions
    height, width, channels = image_np.shape

    # Move percentage matrix to GPU once
    percentage_mat_gpu = cp.asarray(percentage_mat)
    num_outputs = percentage_mat_gpu.shape[0]

    # Create result matrix in CPU memory
    result_matrix = np.zeros((height, width, num_outputs), dtype=np.float32)

    start_loop = time.time()
    for i in range(height):
        # Process one row at a time to save GPU memory
        row_gpu = cp.asarray(image_np[i])  # Shape: (width, channels)

        # Set up broadcasting dimensions
        expanded_row = row_gpu[:, None, :]  # Shape: (width, 1, channels)
        expanded_pmat = percentage_mat_gpu[None, :, :]  # Shape: (1, num_outputs, channels)

        # Multiply and sum
        weighted = expanded_row * expanded_pmat
        row_result = xp.sum(weighted, axis=2)

        # Move result back to CPU
        result_matrix[i] = xp.asnumpy(row_result)

        # Free memory periodically
        if i % 100 == 0:
            cp.get_default_memory_pool().free_all_blocks()
            print(f"Processing row {i}/{height} ({i / height * 100:.1f}%)")

    print(f"Total processing time: {time.time() - start_loop:.2f} seconds")
    return result_matrix.transpose((2, 0, 1))


class NumpyViewerApp(QWidget):
    """
    A PyQt5-based GUI application to visualize and analyze NumPy image arrays.
    Supports loading a NumPy array from a file, selecting channels for visualization,
    and plotting pixel values.
    """

    def __init__(self):
        """Initialize the GUI and application state variables."""
        super().__init__()
        self.percentage_mat = None
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
        self.wavelengths = None  # For storing wavelength data from CSV
        self.save_path = None  # Path to save calculation results

    def initUI(self):
        """Set up the GUI layout and widgets."""
        self.setWindowTitle("NumPy Image Viewer")
        self.setWindowIcon(QIcon("icon.png"))
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()
        button_layout = QHBoxLayout()
        load_layout = QHBoxLayout()

        # Buttons for loading images
        self.load_file_button = QPushButton("Load Single NumPy File")
        self.load_file_button.clicked.connect(self.load_single_file)
        load_layout.addWidget(self.load_file_button)

        # Add CSV loading button
        self.load_csv_button = QPushButton("Load Spectral CSV")
        self.load_csv_button.clicked.connect(self.load_csv_file)
        load_layout.addWidget(self.load_csv_button)

        layout.addLayout(load_layout)

        # Labels to display image shapes
        self.shape_label = QLabel("Images Shape: Not Loaded")
        layout.addWidget(self.shape_label)

        # Channel selector with FIXED HEIGHT
        self.channel_selector = QListWidget()
        self.channel_selector.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.channel_selector.setFixedHeight(80)  # Set fixed height
        layout.addWidget(self.channel_selector)

        # Button layout
        self.show_image_button = QPushButton("Show Selected Channels")
        self.show_image_button.clicked.connect(self.show_image)
        button_layout.addWidget(self.show_image_button)

        self.plot_pixel_button = QPushButton("Plot Pixel Values")
        self.plot_pixel_button.clicked.connect(self.plot_pixel_values)
        button_layout.addWidget(self.plot_pixel_button)

        # Add clear button
        self.clear_button = QPushButton("Clear Points")
        self.clear_button.clicked.connect(self.clear)
        button_layout.addWidget(self.clear_button)

        self.calc_button = QPushButton("Run Spectral Calculation")
        self.calc_button.clicked.connect(self.run_spectral_calculation)
        button_layout.addWidget(self.calc_button)

        layout.addLayout(button_layout)

        # Matplotlib canvas configured to expand
        from PyQt5.QtWidgets import QSizePolicy
        self.canvas = FigureCanvas(Figure())
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Create matplotlib toolbar for navigation (zoom, pan, etc.)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Add canvas and toolbar to layout
        layout.addWidget(self.toolbar)  # Add the toolbar
        layout.addWidget(self.canvas, 1)  # The '1' is a stretch factor

        self.ax1 = self.canvas.figure.add_subplot(1, 2, 1)
        self.ax3 = self.canvas.figure.add_subplot(1, 2, 2)
        self.canvas.mpl_connect("button_press_event", self.on_click)

        self.setLayout(layout)

    def load_csv_file(self):
        self.clear()
        """Load spectral data from a CSV file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select CSV File", "", "CSV Files (*.csv)"
        )

        if file_path:
            try:
                result = read_and_filter_csv(file_path)

                # Extract wavelengths and percentage matrix
                self.wavelengths = result['wave_lengths']
                self.percentage_mat = result['percentage_mat']

                # Update UI with information about loaded data
                QMessageBox.information(self, "Success",
                                        f"Loaded spectral data with {self.wavelengths.shape[0]} wavelength bands")

                # Populate wavelength selector if we have valid data
                self.populate_wavelength_selector(self.wavelengths)

            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load CSV file: {str(e)}")

    def run_spectral_calculation(self):
        """Run hyperspectral calculation using loaded image and spectral data."""
        # Check if we have necessary data
        if self.wavelengths is None:
            QMessageBox.warning(self, "Error", "No spectral CSV data loaded!")
            return

        # Get file path for saving the result
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Calculation Result", "", "NumPy Files (*.npy)"
        )

        if not save_path:
            return  # User canceled

        try:
            # First, select the input image file
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select Image File for Calculation", "", "NumPy Files (*.npy)"
            )

            if not file_path:
                return  # User canceled

            # Extract the needed data from what we loaded previously
            if self.percentage_mat is None:
                QMessageBox.warning(self, "Error", "Spectrometer matrix not available!")
                return

            # Store save path for later use
            self.save_path = save_path

            # Create and set up the worker thread
            self.thread = QThread()
            self.worker = SpectralWorker(file_path, self.percentage_mat)
            self.worker.moveToThread(self.thread)

            # Connect signals and slots
            self.thread.started.connect(self.worker.run)
            self.worker.finished.connect(self.handle_calculation_result)
            self.worker.finished.connect(self.thread.quit)
            self.worker.error.connect(self.handle_calculation_error)

            # Clean up when thread finishes
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)

            # Update UI and disable button during calculation
            self.shape_label.setText("Calculation running... Please wait")
            self.calc_button.setEnabled(False)

            # Store start time and start the thread
            self.thread.startTime = time.time()
            self.thread.start()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start calculation: {str(e)}")

    def handle_calculation_result(self, result):
        """Handle the completed calculation result."""
        try:
            # Save the result
            np.save(self.save_path, result)

            # Calculate and display timing info
            elapsed = time.time() - self.thread.startTime if hasattr(self.thread, 'startTime') else 0
            QMessageBox.information(self, "Calculation Complete",
                                    f"Calculation finished in {elapsed:.2f} seconds.\n"
                                    f"Result saved to: {os.path.basename(self.save_path)}")

            # Re-enable button
            self.calc_button.setEnabled(True)
            self.shape_label.setText(f"Calculation complete. Shape: {result.shape}")

            # Optionally load the result for viewing
            if QMessageBox.question(self, "Load Result",
                                    "Do you want to load the result for viewing?",
                                    QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
                self.images = [result.transpose((1, 2, 0))]  # Replace with new result
                self.populate_channel_selector()
                self.show_image()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save result: {str(e)}")

    def handle_calculation_error(self, error_msg):
        """Handle errors from the calculation thread."""
        QMessageBox.critical(self, "Calculation Error", f"Error during calculation: {error_msg}")
        self.calc_button.setEnabled(True)
        self.shape_label.setText("Calculation failed")

    def populate_wavelength_selector(self, wavelengths):
        """Populate the channel selector with wavelength values."""
        self.channel_selector.clear()
        for i, wl in enumerate(wavelengths):
            self.channel_selector.addItem(QListWidgetItem(f"Wave {wl:.1f}nm"))

    def on_radio_button_clicked(self, button):
        """Handle radio button selection."""
        QMessageBox.information(self, "Radio Button Selected", f"You selected: {button.text()}")

    def load_single_file(self):
        """Load a single NumPy image file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select NumPy File", "", "NumPy Files (*.npy)"
        )

        if file_path:
            self.images = []  # Clear existing images
            try:
                array = np.load(file_path).astype(np.float16)
                if array.ndim == 3:
                    self.images.append(np.transpose(array, (1, 2, 0)))
                    print(f'Image shape: {array.shape}')

                    self.shape_label.setText(f"Loaded image. Shape: {array.shape}")
                    self.populate_channel_selector()
                else:
                    QMessageBox.warning(self, "Error", f"File {os.path.basename(file_path)} is not a 3D array!")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load file: {str(e)}")

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
        self.selected_channels = [int(self.channel_selector.row(item)) for item in selected_items]

        # Clear only the image axis, not the plot axis
        self.ax1.clear()

        # Show selected channels for the image
        if self.images:
            # If no channels selected, use first channel
            if not self.selected_channels:
                self.selected_channels = [0]

            if len(self.selected_channels) == 1:
                # Display single channel as grayscale
                self.ax1.imshow(self.images[0][:, :, self.selected_channels[0]], cmap='gray')
            else:
                # For multiple channels, create a composite (use first 3 channels at most)
                rgb_channels = self.selected_channels[:3]
                while len(rgb_channels) < 3:
                    rgb_channels.append(0)  # Pad with zeros if needed

                rgb_img = np.zeros((self.images[0].shape[0], self.images[0].shape[1], 3))
                for i, chan in enumerate(rgb_channels[:3]):
                    rgb_img[:, :, i] = self.images[0][:, :, chan]

                # Normalize for display
                if rgb_img.max() > 0:
                    rgb_img = rgb_img / rgb_img.max()
                self.ax1.imshow(rgb_img)

            # Redraw all previously selected points
            for x, y in self.points:
                self.ax1.plot(x, y, 'go')  # Green circle
                self.ax1.text(x, y, f'({x:.1f}, {y:.1f})', color='white')

        self.ax1.set_title("Image")
        self.canvas.draw()

    def clear(self):
        """Clear the plot and the chosen pixels."""
        self.points = []  # Reset the list of selected points
        self.show_image()  # Refresh display
        self.canvas.draw()  # Redraw the canvas

    def on_click(self, event):
        """Handles the click event on the canvas to select points."""
        if event.inaxes != self.ax1:
            return

        # Check if the toolbar is in navigation mode (zoom, pan, etc.)
        if self.toolbar.mode != "":
            return  # Skip point selection when using toolbar navigation tools

        # Record the coordinates of the click
        self.points.append((event.xdata, event.ydata))
        print(f"Point selected: {event.xdata}, {event.ydata}")

        # Use 'go' for green circle marker
        self.ax1.plot(event.xdata, event.ydata, 'go')
        self.ax1.text(event.xdata, event.ydata, f'({event.xdata:.1f}, {event.ydata:.1f})', color='white')

        self.canvas.draw()

        # Clear any existing points on the plots
        self.ax1.clear()
        self.show_image()  # Refresh the display

    def plot_pixel_values(self):
        """Plot the pixel values across all channels for all selected points."""
        if not self.points:
            QMessageBox.warning(self, "Error", "Select at least one point on the image!")
            return

        if not self.images:
            QMessageBox.warning(self, "Error", "No images loaded!")
            return

        # Clear the plot
        self.ax3.clear()

        # Get image dimensions
        img_height, img_width = self.images[0].shape[:2]
        image = self.images[0]

        # Define a list of colors for multiple plots
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']

        # Plot data for each selected point
        for idx, (x, y) in enumerate(self.points):
            # Convert to integers for indexing
            x_int, y_int = int(x), int(y)

            # Check if the point is within image boundaries
            if x_int < 0 or x_int >= img_width or y_int < 0 or y_int >= img_height:
                print(f"Point ({x_int}, {y_int}) is outside image boundaries!")
                continue

            try:
                # Get pixel values for this point
                pixel_values = image[y_int, x_int, :]

                # Select color (cycle through colors if more points than colors)
                color = colors[idx % len(colors)]

                # Plot with the point coordinates in the label
                if self.wavelengths is not None:
                    self.ax3.plot(self.wavelengths, pixel_values, marker='o',
                                  label=f"Point {idx + 1} ({x_int}, {y_int})",
                                  color=color, markersize=3)
                else:
                    channel_indices = np.arange(pixel_values.shape[0])
                    self.ax3.plot(channel_indices, pixel_values, marker='o',
                                  label=f"Point {idx + 1} ({x_int}, {y_int})",
                                  color=color, markersize=3)

            except IndexError as e:
                print(f"IndexError at point ({x_int}, {y_int}): {e}")
                continue

        # Set labels and title
        if self.wavelengths is not None:
            self.ax3.set_xlabel("Wavelength (nm)")
        else:
            self.ax3.set_xlabel("Channel")

        self.ax3.set_ylabel("Pixel Value")
        self.ax3.set_title(f"Spectral Profiles for {len(self.points)} Points")
        self.ax3.legend(loc='best', prop={'size': 8})
        self.ax3.grid(True)
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = NumpyViewerApp()
    viewer.show()
    sys.exit(app.exec_())