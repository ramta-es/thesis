import sys
import numpy as np
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QProgressBar, QTextEdit, QMessageBox,
                             QGroupBox, QSpinBox, QListWidget, QRadioButton,
                             QSizePolicy)

from PyQt5.QtCore import QThread, pyqtSignal


class ImageCanvas(FigureCanvasQTAgg):
    """Canvas for displaying image with region/pixel selection"""

    selection_changed = pyqtSignal()  # Signal emitted when selection changes

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)

        self.image_data = None
        self.rect_selector = None
        self.selected_region = None
        self.selected_pixels = []
        self.mode = None

    def display_image(self, image_data, channels):
        """Display RGB visualization using specified channels"""
        self.image_data = image_data
        self.ax.clear()

        if len(channels) >= 3:
            rgb = np.stack([
                image_data[:, :, channels[0]],
                image_data[:, :, channels[1]],
                image_data[:, :, channels[2]]
            ], axis=2)
        else:
            rgb = np.mean(image_data, axis=2)

        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())

        self.ax.imshow(rgb)
        self.ax.set_title(f"Channels: R={channels[0]}, G={channels[1]}, B={channels[2]}")
        self.draw()

    def enable_region_selection(self):
        """Enable rectangle selection for region"""
        self.mode = 'region'
        self.selected_pixels = []

        from matplotlib.widgets import RectangleSelector

        def on_select(eclick, erelease):
            x1, y1 = int(eclick.xdata), int(eclick.ydata)
            x2, y2 = int(erelease.xdata), int(erelease.ydata)

            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            self.selected_region = (y1, y2, x1, x2)
            self.selection_changed.emit()  # Notify parent

        self.rect_selector = RectangleSelector(
            self.ax, on_select,
            useblit=True,
            button=[1],
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=True
        )

    def enable_pixel_selection(self):
        """Enable individual pixel selection"""
        self.mode = 'pixel'
        self.cid = self.mpl_connect('button_press_event', self.on_pixel_click)

    def on_pixel_click(self, event):
        """Handle pixel click event"""
        if event.inaxes != self.ax or self.image_data is None:
            return

        x, y = int(event.xdata), int(event.ydata)

        if 0 <= y < self.image_data.shape[0] and 0 <= x < self.image_data.shape[1]:
            self.selected_pixels.append((y, x))
            self.ax.plot(x, y, 'r+', markersize=10, markeredgewidth=2)
            self.draw()
            self.selection_changed.emit()  # Notify parent

    def get_region_pixels(self):
        """Extract pixels from selected region"""
        if self.selected_region is None or self.image_data is None:
            return None

        y1, y2, x1, x2 = self.selected_region
        selected = self.image_data[y1:y2, x1:x2, :]
        return selected.reshape(-1, self.image_data.shape[2])

    def get_individual_pixels(self):
        """Extract individual selected pixels"""
        if not self.selected_pixels or self.image_data is None:
            return None

        pixels = np.array([self.image_data[y, x, :] for y, x in self.selected_pixels])
        return pixels

    def clear_selections(self):
        """Clear all selections"""
        self.selected_region = None
        self.selected_pixels = []
        if hasattr(self, 'cid'):
            self.mpl_disconnect(self.cid)


class CalculationThread(QThread):
    """Thread for running Mahalanobis calculations"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(np.ndarray)
    error = pyqtSignal(str)

    def __init__(self, data_cloud, measured_pixels):
        super().__init__()
        self.data_cloud = data_cloud
        self.measured_pixels = measured_pixels

    def run(self):
        try:
            num_channels = self.data_cloud.shape[1]
            distances = np.zeros(num_channels)

            for excluded_channel in range(num_channels):
                channel_mask = np.ones(num_channels, dtype=bool)
                channel_mask[excluded_channel] = False

                selected_data_cloud = self.data_cloud[:, channel_mask]
                selected_measured_pixels = self.measured_pixels[:, channel_mask]

                pixel_distances = self.calculate_mahalanobis_distance(
                    selected_data_cloud, selected_measured_pixels
                )
                distances[excluded_channel] = np.mean(pixel_distances)

                progress = int(((excluded_channel + 1) / num_channels) * 100)
                self.progress.emit(progress)

            self.finished.emit(distances)

        except Exception as e:
            self.error.emit(str(e))

    def calculate_mahalanobis_distance(self, data_cloud, measured_dist):
        """Calculate Mahalanobis distance between data cloud and measured pixels"""
        mean1 = np.mean(data_cloud, axis=0)
        cov_matrix = np.cov(data_cloud.T)

        try:
            inv_cov = np.linalg.inv(cov_matrix)
            distances = [mahalanobis(pixel, mean1, inv_cov) for pixel in measured_dist]
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(cov_matrix)
            diff = measured_dist - mean1
            distances = np.sqrt(np.sum((diff @ inv_cov) * diff, axis=1)).tolist()

        return distances


class MahalanobisGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mahalanobis Distance - Custom Pixel Selection")
        self.setGeometry(100, 100, 1200, 900)

        self.image = None
        self.display_channels = [500, 300, 150]
        self.data_cloud = None
        self.measured_pixels = None
        self.distances = None

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Image loading
        load_group = QGroupBox("Image Loading")
        load_layout = QHBoxLayout()
        self.load_btn = QPushButton("Load Image")
        self.load_btn.clicked.connect(self.load_image)
        self.image_label = QLabel("No image loaded")
        load_layout.addWidget(self.load_btn)
        load_layout.addWidget(self.image_label)
        load_layout.addStretch()
        load_group.setLayout(load_layout)
        main_layout.addWidget(load_group)

        # Channel selection
        channel_group = QGroupBox("Display Channels (RGB)")
        channel_layout = QHBoxLayout()

        labels = ["R:", "G:", "B:"]
        defaults = [500, 300, 150]
        spin_attr_names = ["r_spin", "g_spin", "b_spin"]

        for label, default, attr in zip(labels, defaults, spin_attr_names):
            channel_layout.addWidget(QLabel(label))
            spin = QSpinBox()
            spin.setMinimum(0)
            spin.setMaximum(839)
            spin.setValue(default)
            spin.valueChanged.connect(self.update_display)
            channel_layout.addWidget(spin)
            setattr(self, attr, spin)



        self.update_display_btn = QPushButton("Update Display")
        self.update_display_btn.clicked.connect(self.update_display)
        self.update_display_btn.setEnabled(False)
        channel_layout.addWidget(self.update_display_btn)
        channel_layout.addStretch()

        channel_group.setLayout(channel_layout)
        main_layout.addWidget(channel_group)

        # Selection controls
        selection_layout = QHBoxLayout()

        # Set 1 (Region)
        set1_group = QGroupBox("Set 1: Region Selection")
        set1_layout = QVBoxLayout()
        self.select_region_btn = QPushButton("Select Region (Draw Rectangle)")
        self.select_region_btn.clicked.connect(self.select_region)
        self.select_region_btn.setEnabled(False)
        self.region_label = QLabel("No region selected")
        set1_layout.addWidget(self.select_region_btn)
        set1_layout.addWidget(self.region_label)
        set1_group.setLayout(set1_layout)

        # Set 2 (Individual pixels)
        set2_group = QGroupBox("Set 2: Individual Pixels")
        set2_layout = QVBoxLayout()
        self.select_pixels_btn = QPushButton("Select Pixels (Click on Image)")
        self.select_pixels_btn.clicked.connect(self.select_pixels)
        self.select_pixels_btn.setEnabled(False)
        self.clear_pixels_btn = QPushButton("Clear Pixel Selection")
        self.clear_pixels_btn.clicked.connect(self.clear_pixel_selection)
        self.clear_pixels_btn.setEnabled(False)
        self.pixels_list = QListWidget()
        self.pixels_list.setMaximumHeight(80)
        set2_layout.addWidget(self.select_pixels_btn)
        set2_layout.addWidget(self.clear_pixels_btn)
        set2_layout.addWidget(QLabel("Selected pixels:"))
        set2_layout.addWidget(self.pixels_list)
        set2_group.setLayout(set2_layout)

        selection_layout.addWidget(set1_group)
        selection_layout.addWidget(set2_group)
        main_layout.addLayout(selection_layout)

        # Image canvas
        self.canvas = ImageCanvas()
        self.canvas.setMinimumHeight(300)  # ensure some space for controls
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.selection_changed.connect(self.on_selection_changed)
        main_layout.addWidget(self.canvas, 1)

        # Calculate button
        self.calc_btn = QPushButton("Calculate Mahalanobis Distances")
        self.calc_btn.clicked.connect(self.calculate_distances)
        self.calc_btn.setEnabled(False)
        main_layout.addWidget(self.calc_btn)

        # Progress bar
        self.progress_bar = QProgressBar()
        main_layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("Ready")
        main_layout.addWidget(self.status_label)

        # Results
        results_layout = QHBoxLayout()
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text, 3)

        # Buttons
        buttons_layout = QVBoxLayout()
        self.plot_btn = QPushButton("Show Plots")
        self.plot_btn.clicked.connect(self.show_plots)
        self.plot_btn.setEnabled(False)
        self.save_btn = QPushButton("Save Results")
        self.save_btn.clicked.connect(self.save_results)
        self.save_btn.setEnabled(False)
        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.clicked.connect(self.clear_all)
        buttons_layout.addWidget(self.plot_btn)
        buttons_layout.addWidget(self.save_btn)
        buttons_layout.addWidget(self.clear_btn)
        buttons_layout.addStretch()

        results_layout.addLayout(buttons_layout, 1)
        main_layout.addLayout(results_layout)

    def load_image(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "NumPy files (*.npy);;All files (*.*)"
        )
        if filepath:
            try:
                self.image = np.load(filepath)
                self.image_label.setText(f"Shape: {self.image.shape}")

                max_channel = self.image.shape[2] - 1
                self.r_spin.setMaximum(max_channel)
                self.g_spin.setMaximum(max_channel)
                self.b_spin.setMaximum(max_channel)

                self.update_display()
                self.select_region_btn.setEnabled(True)
                self.select_pixels_btn.setEnabled(True)
                self.update_display_btn.setEnabled(True)

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image:\n{str(e)}")

    def update_display(self):
        if self.image is None:
            return

        self.display_channels = [
            self.r_spin.value(),
            self.g_spin.value(),
            self.b_spin.value()
        ]
        self.canvas.display_image(self.image, self.display_channels)

    def select_region(self):
        self.canvas.clear_selections()
        self.canvas.enable_region_selection()
        self.status_label.setText("Draw rectangle to select region for Set 1")

    def select_pixels(self):
        self.canvas.clear_selections()
        self.canvas.enable_pixel_selection()
        self.clear_pixels_btn.setEnabled(True)
        self.status_label.setText("Click on image to select individual pixels for Set 2")

    def clear_pixel_selection(self):
        self.canvas.clear_selections()
        self.pixels_list.clear()
        self.status_label.setText("Pixel selection cleared")
        self.calc_btn.setEnabled(False)
        self.region_label.setText("No region selected")

    def on_selection_changed(self):
        """Called when canvas selection changes"""
        # Update region label
        if self.canvas.selected_region is not None:
            pixels = self.canvas.get_region_pixels()
            if pixels is not None:
                self.region_label.setText(f"Region: {pixels.shape[0]} pixels selected")

        # Update pixel list
        if len(self.canvas.selected_pixels) > 0:
            self.pixels_list.clear()
            for y, x in self.canvas.selected_pixels:
                self.pixels_list.addItem(f"Pixel ({y}, {x})")

        # Enable calculate button if both selections are made
        self.check_calc_ready()

    def check_calc_ready(self):
        """Enable calculate button when both selections are made"""
        if (self.canvas.selected_region is not None and
                len(self.canvas.selected_pixels) > 0):
            self.calc_btn.setEnabled(True)
        else:
            self.calc_btn.setEnabled(False)

    def calculate_distances(self):
        self.data_cloud = self.canvas.get_region_pixels()
        if self.data_cloud is None:
            QMessageBox.warning(self, "Warning", "Please select a region first")
            return

        self.measured_pixels = self.canvas.get_individual_pixels()
        if self.measured_pixels is None:
            QMessageBox.warning(self, "Warning", "Please select at least one pixel")
            return

        self.calc_btn.setEnabled(False)
        self.status_label.setText("Calculating...")
        self.progress_bar.setValue(0)

        self.calc_thread = CalculationThread(self.data_cloud, self.measured_pixels)
        self.calc_thread.progress.connect(self.update_progress)
        self.calc_thread.finished.connect(self.calculation_finished)
        self.calc_thread.error.connect(self.calculation_error)
        self.calc_thread.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def calculation_finished(self, distances):
        self.distances = distances
        self.display_results()
        self.calc_btn.setEnabled(True)
        self.plot_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.status_label.setText("Calculation complete!")

    def calculation_error(self, error_msg):
        QMessageBox.critical(self, "Error", f"Calculation failed:\n{error_msg}")
        self.calc_btn.setEnabled(True)
        self.status_label.setText("Error")

    def display_results(self):
        text = "=" * 60 + "\n"
        text += "MAHALANOBIS DISTANCE RESULTS\n"
        text += "=" * 60 + "\n\n"
        text += f"Set 1 (Region): {self.data_cloud.shape[0]} pixels\n"
        text += f"Set 2 (Individual): {self.measured_pixels.shape[0]} pixels\n"
        text += f"Total channels analyzed: {len(self.distances)}\n"
        text += f"Mean distance: {np.mean(self.distances):.4f}\n"
        text += f"Std deviation: {np.std(self.distances):.4f}\n"
        text += f"Min distance: {np.min(self.distances):.4f} (Channel {np.argmin(self.distances)})\n"
        text += f"Max distance: {np.max(self.distances):.4f} (Channel {np.argmax(self.distances)})\n"
        text += "=" * 60 + "\n\n"

        sorted_indices = np.argsort(self.distances)
        text += "Top 10 channels with HIGHEST distance:\n"
        for i in range(min(10, len(self.distances))):
            idx = sorted_indices[-(i + 1)]
            text += f"  Channel {idx}: {self.distances[idx]:.4f}\n"

        text += "\nTop 10 channels with LOWEST distance:\n"
        for i in range(min(10, len(self.distances))):
            idx = sorted_indices[i]
            text += f"  Channel {idx}: {self.distances[idx]:.4f}\n"

        self.results_text.setText(text)

    def show_plots(self):
        if self.distances is None:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(self.distances, linewidth=1, alpha=0.7)
        ax1.set_xlabel('Excluded Channel Index')
        ax1.set_ylabel('Mahalanobis Distance')
        ax1.set_title('Mahalanobis Distance vs Excluded Channel')
        ax1.grid(True, alpha=0.3)

        ax2.hist(self.distances, bins=50, edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Mahalanobis Distance')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Distances')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def save_results(self):
        if self.distances is None:
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Results", "", "NumPy files (*.npy);;All files (*.*)"
        )

        if filepath:
            try:
                np.save(filepath, self.distances)
                QMessageBox.information(self, "Success", f"Results saved to:\n{filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save:\n{str(e)}")

    def clear_all(self):
        self.image = None
        self.data_cloud = None
        self.measured_pixels = None
        self.distances = None

        self.image_label.setText("No image loaded")
        self.region_label.setText("No region selected")
        self.pixels_list.clear()
        self.results_text.clear()
        self.progress_bar.setValue(0)
        self.status_label.setText("Ready")

        self.canvas.ax.clear()
        self.canvas.draw()
        self.canvas.clear_selections()

        self.select_region_btn.setEnabled(False)
        self.select_pixels_btn.setEnabled(False)
        self.clear_pixels_btn.setEnabled(False)
        self.update_display_btn.setEnabled(False)
        self.calc_btn.setEnabled(False)
        self.plot_btn.setEnabled(False)
        self.save_btn.setEnabled(False)


def main():
    app = QApplication(sys.argv)
    window = MahalanobisGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()