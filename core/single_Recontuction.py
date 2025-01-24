import sys
import os
import numpy as np
import open3d as o3d  # For point cloud and mesh processing
import torch

from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton,
                             QFileDialog, QVBoxLayout, QProgressBar, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# Import your model and other necessary modules
from models.encoder.encoder import Encoder
from models.decoder.decoder import Decoder
import utils.data_transforms
import utils.helpers  # Make sure this includes your binvox_rw module
import mcubes  # For marching cubes


class ReconstructionWorker(QThread):
    progress_updated = pyqtSignal(int)
    reconstruction_finished = pyqtSignal(np.ndarray)
    error_occurred = pyqtSignal(str)

    def __init__(self, image_path, model_path, cfg, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.model_path = model_path
        self.cfg = cfg

    def run(self):
        try:
            # 1. Model Loading
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            encoder = Encoder(self.cfg).to(device)
            decoder = Decoder(self.cfg).to(device)

            checkpoint = torch.load(self.model_path, map_location=device)
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            decoder.load_state_dict(checkpoint['decoder_state_dict'])
            encoder.eval()
            decoder.eval()

            # 2. Image Loading and Preprocessing
            IMG_SIZE = self.cfg.CONST.IMG_H, self.cfg.CONST.IMG_W
            CROP_SIZE = self.cfg.CONST.CROP_IMG_H, self.cfg.CONST.CROP_IMG_W
            transform = utils.data_transforms.Compose([
                utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
                utils.data_transforms.ToTensor(),  # Assuming your model takes tensors
                utils.data_transforms.normalize
            ])

            image = QImage(self.image_path)
            if image.isNull():
                raise IOError(f"Failed to load image: {self.image_path}")
            image = image.convertToFormat(QImage.Format_RGB888)  # Ensure correct format
            image_np = np.frombuffer(image.constBits(), np.uint8).reshape(image.height(), image.width(), 3)
            rendering_image = transform(image_np).unsqueeze(0).to(device)


            # 3. Inference
            with torch.no_grad():
                image_features = encoder(rendering_image)
                generated_volume = decoder(image_features).squeeze(dim=0).squeeze(dim=0)

            # 4. Postprocessing (Thresholding, Meshing)
            threshold = 0.4  # Adjust as needed
            pred_volume = (generated_volume > threshold).cpu().numpy()

            # Example meshing (using marching cubes):
            vertices, triangles = mcubes.marching_cubes(pred_volume, 0)
            # ... further mesh processing if needed ...


            self.reconstruction_finished.emit(pred_volume)

        except Exception as e:
            self.error_occurred.emit(str(e))


class SingleViewReconstructionApp(QWidget):
    def __init__(self):
        super().__init__()

        # Replace with your actual config
        class CFG:
            CONST = type('obj', (object,), {'IMG_H': 224, 'IMG_W': 224, 'CROP_IMG_H': 224, 'CROP_IMG_W':224})
        self.cfg = CFG
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Single View 3D Reconstruction")

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)

        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.loadImage)

        self.reconstruct_button = QPushButton("Reconstruct")
        self.reconstruct_button.clicked.connect(self.reconstruct)
        self.reconstruct_button.setEnabled(False)  # Disable until image is loaded

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.load_button)
        layout.addWidget(self.reconstruct_button)
        layout.addWidget(self.progress_bar)

        self.setLayout(layout)

    def loadImage(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.bmp);;All Files (*)", options=options)
        if filename:
            self.image_path = filename  # Store the image path
            pixmap = QPixmap(filename)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio))
            self.reconstruct_button.setEnabled(True)

    def reconstruct(self):

        model_path = "path/to/your/model.pth"  # Replace with your model path

        self.worker = ReconstructionWorker(self.image_path, model_path, self.cfg)
        self.worker.progress_updated.connect(self.updateProgress)
        self.worker.reconstruction_finished.connect(self.reconstructionFinished)
        self.worker.error_occurred.connect(self.showError)
        self.worker.start()
        self.reconstruct_button.setEnabled(False)


    def updateProgress(self, value):
        self.progress_bar.setValue(value)

    def reconstructionFinished(self, volume):
        # Process the reconstructed volume (e.g., display, save)
        print("Reconstruction finished!")
        # Example: Save as binvox
        output_path = "output.binvox"  # Choose your output path
        with open(output_path, 'wb') as f:
             vox = utils.binvox_rw.Voxels(volume, (32,) * 3, (0,) * 3, 1, 'xzy')
             vox.write(f)
        self.reconstruct_button.setEnabled(True) # Enable the button again

        QMessageBox.information(self, "Success", "Reconstruction complete! Model saved to output.binvox")

    def showError(self, message):
        QMessageBox.critical(self, "Error", f"An error occurred during reconstruction:\n{message}")
        self.reconstruct_button.setEnabled(True)




if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SingleViewReconstructionApp()
    window.show()
    sys.exit(app.exec_())
