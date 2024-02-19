import sys
import numpy as np
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QPushButton,
    QFileDialog,
    QVBoxLayout,
    QLabel,
    QHBoxLayout,
    QDialog,
    QSpinBox,
    QComboBox,
)
from PyQt6.QtGui import QPixmap, QImage, QAction
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from libs.video_utils import generate_vr_video
from libs.settings import settings_manager


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray, np.ndarray)
    finished_signal = pyqtSignal()

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.running = True  # Control the thread running

    def run(self):
        generate_vr_video(self.video_path, self.emit_frame)
        self.finished_signal.emit()

    def stop(self):
        self.running = False

    def emit_frame(self, frame: np.ndarray, new_frame: np.ndarray):
        if self.running:
            self.change_pixmap_signal.emit(frame, new_frame)


class ConverterWidget(QWidget):
    def __init__(self):
        super().__init__()

        # Layout
        self.layout = QVBoxLayout()

        # Horizontal layout for the QLabels
        self.h_layout = QHBoxLayout()  # Create a QHBoxLayout

        # Original Image label
        self.original_image_label = QLabel()
        self.original_image_label.setMaximumSize(1280, 720)
        self.h_layout.addWidget(self.original_image_label)

        # Depth Map label
        self.new_frame_label = QLabel()
        self.new_frame_label.setMaximumSize(1280, 720)
        self.h_layout.addWidget(self.new_frame_label)

        # Add the QHBoxLayout to the main QVBoxLayout
        self.layout.addLayout(self.h_layout)

        # Start button
        self.start_button = QPushButton("Select Video and Start")
        self.start_button.clicked.connect(self.open_file_dialog)
        self.layout.addWidget(self.start_button)

        # Set the layout to the QWidget
        self.setLayout(self.layout)

        # Video thread
        self.video_thread = None

    def open_file_dialog(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", "", "Video Files (*.mp4 *.flv *.ts *.mts *.avi)"
        )
        if filename:
            self.start_button.setEnabled(False)
            if (
                self.video_thread is not None
            ):  # Stop existing video thread if it's running
                self.video_thread.stop()
                self.video_thread.wait()
            self.video_thread = VideoThread(filename)
            self.video_thread.change_pixmap_signal.connect(self.update_frame_labels)
            self.video_thread.finished_signal.connect(self.unlock_start_button)
            self.video_thread.start()

    def unlock_start_button(self):
        self.start_button.setEnabled(True)

    def update_frame_labels(self, frame: np.ndarray, new_frame: np.ndarray):
        video_pixmap = self.convert_cv_to_pixmap(frame, self.original_image_label)
        new_frame_pixmap = self.convert_cv_to_pixmap(new_frame, self.new_frame_label)
        self.original_image_label.setPixmap(video_pixmap)
        self.new_frame_label.setPixmap(new_frame_pixmap)

    def convert_cv_to_pixmap(self, cv_img: np.ndarray, label: QImage):
        """Converts an opencv image to QPixmap"""
        if len(cv_img.shape) == 2:  # Grayscale image
            height, width = cv_img.shape
            q_img = QImage(
                cv_img.data, width, height, width, QImage.Format.Format_Grayscale8
            )
        else:  # Assume BGR color image
            height, width, channels = cv_img.shape
            bytes_per_line = channels * width
            q_img = QImage(
                cv_img.data, width, height, bytes_per_line, QImage.Format.Format_RGB888
            ).rgbSwapped()
        pixmap = QPixmap.fromImage(q_img)
        return pixmap.scaled(
            label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    def closeEvent(self, event):
        if self.video_thread is not None:
            self.video_thread.stop()
            self.video_thread.wait()
        super().closeEvent(event)


class SettingsDialog(QDialog):
    """
    A simple settings dialog.
    """

    def __init__(self, parent=None):
        super(SettingsDialog, self).__init__(parent)

        self.available_ai_models = [
            "Intel/dpt-swinv2-large-384",
            "Intel/dpt-beit-large-512",
            "LiheYoung/depth-anything-large-hf",
            "LiheYoung/depth-anything-base-hf",
            "LiheYoung/depth-anything-small-hf",
            "LiheYoung/depth_anything_vitl14",
            "LiheYoung/depth_anything_vitb14",
            "LiheYoung/depth_anything_vits14",
        ]
        settings_manager.load_settings()
        print(settings_manager.settings)
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Settings")
        layout = QVBoxLayout(self)

        # Batch Size Setting
        batch_size_layout = QHBoxLayout()
        self.batch_size_label = QLabel("Batch Size:")
        self.batch_size_input = QSpinBox()
        self.batch_size_input.setMinimum(1)
        self.batch_size_input.setMaximum(10000)
        self.batch_size_input.setValue(settings_manager.settings.batch_size)
        batch_size_layout.addWidget(self.batch_size_label)
        batch_size_layout.addWidget(self.batch_size_input)
        layout.addLayout(batch_size_layout)

        # AI Model Setting
        ai_model_layout = QHBoxLayout()
        self.ai_model_label = QLabel("AI Model:")
        self.ai_model_input = QComboBox()
        self.ai_model_input.addItems(self.available_ai_models)
        # Select the current ai_model in the dropdown
        current_ai_model_index = self.ai_model_input.findText(
            settings_manager.settings.ai_model
        )
        if current_ai_model_index >= 0:
            self.ai_model_input.setCurrentIndex(current_ai_model_index)
        ai_model_layout.addWidget(self.ai_model_label)
        ai_model_layout.addWidget(self.ai_model_input)
        layout.addLayout(ai_model_layout)

        # Depth Offset Setting
        depth_offset_layout = QHBoxLayout()
        self.depth_offset_label = QLabel("Depth Offset:")
        self.depth_offset_input = QSpinBox()
        self.depth_offset_input.setMinimum(0)
        self.depth_offset_input.setMaximum(1000)
        self.depth_offset_input.setValue(settings_manager.settings.depth_offset)
        depth_offset_layout.addWidget(self.depth_offset_label)
        depth_offset_layout.addWidget(self.depth_offset_input)
        layout.addLayout(depth_offset_layout)

        # Save Button
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_settings)
        layout.addWidget(self.save_button)

        self.setLayout(layout)

    def save_settings(self):
        # Retrieve values from input widgets and save them using the settings manager
        settings_manager.set("batch_size", self.batch_size_input.value())
        selected_model = self.ai_model_input.currentText()
        settings_manager.set("ai_model", selected_model)
        settings_manager.set("depth_offset", self.depth_offset_input.value())

        # Close the dialog after saving
        self.accept()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Window Configurations
        self.setWindowTitle("Video To VR (V1.0)")
        self.setGeometry(200, 100, 1280, 720)

        # Main Widget
        self.player = ConverterWidget()
        self.setCentralWidget(self.player)

        # Status bar
        self.statusBar().showMessage("Ready")

        # Menu bar
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        # Add File Menu Actions
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)

        # Create the settings action.
        settings_action = QAction("Settings", self)
        settings_action.setStatusTip("Open settings")
        settings_action.setShortcut("Ctrl+Alt+S")
        settings_action.triggered.connect(self.openSettings)

        file_menu.addAction(settings_action)
        file_menu.addAction(exit_action)

    def openSettings(self):
        """
        Opens the settings dialog.
        """
        settings_dialog = SettingsDialog(self)
        settings_dialog.exec()  # Show the dialog as a modal window.


def main():
    app = QApplication(sys.argv)
    ex = MainWindow()
    ex.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
