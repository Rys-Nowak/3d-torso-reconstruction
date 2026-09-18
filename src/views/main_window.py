import gc
from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSlot
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from src.utils.data_preparation import (
    crop_trimesh,
)
from src.workers import (
    FitWorker,
    FullBodyVisualizationWorker,
    PreprocessWorker,
    ProcessedMeshSaveWorker,
    ScanLoadWorker,
    VisualizationWorker,
)

from .config_view import ConfigView
from .mesh_view import MeshView
from .plot_view import PlotView
from .vedo_view import VedoView


class MainWindow(QWidget):
    def __init__(self, scan_path, output_path=None):
        super().__init__()
        path_text = str(scan_path).strip() if scan_path is not None else ""
        self.scan_path = Path(path_text).expanduser() if path_text else None
        self.uses_default_output_path = not output_path
        self.output_path = (
            self._default_output_path(self.scan_path)
            if self.uses_default_output_path
            else Path(output_path).expanduser()
        )
        self.raw_mesh = None
        self.comparison_mesh = None
        self.preprocessed_mesh = None
        self.preprocessed_points = None
        self.original_to_processed_transform = None
        self.processed_to_output_transform = None
        self.fit_result = None
        self.fit_cropped_points = None
        self.fit_cropped_mesh = None
        self.ground_truth_mesh = None
        self.ground_truth_path = None
        self.worker_thread = None
        self.worker = None
        self.worker_success_slot = None
        self.progress = None
        self.cancel_button = None
        self.pending_scan_path = None
        self.pending_ground_truth_path = None
        self.orientation_dirty = False

        self.setWindowTitle("Single Scan STAR Fitting")
        self.resize(1280, 850)
        layout = QVBoxLayout(self)
        self.stack = QStackedWidget()
        self.mesh_view = MeshView(self.raw_mesh, self.scan_path)
        self.config_view = ConfigView()
        self.plot_view = PlotView()
        self.vedo_view = VedoView()
        for view in (self.mesh_view, self.config_view, self.plot_view, self.vedo_view):
            self.stack.addWidget(view)
        layout.addWidget(self.stack, 1)

        navigation = QHBoxLayout()
        self.previous_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")
        navigation.addWidget(self.previous_button)
        navigation.addStretch()
        navigation.addWidget(self.next_button)
        layout.addLayout(navigation)
        self._connect_signals()
        self._update_navigation()
        if self.scan_path is not None and self.scan_path.is_file():
            QTimer.singleShot(0, self.start_scan_loading)

    @staticmethod
    def _default_output_path(scan_path):
        scan_name = scan_path.stem if scan_path and scan_path.stem else "scan"
        project_root = Path(__file__).resolve().parents[2]
        return project_root / "out" / f"{scan_name}.ply"

    @staticmethod
    def _processed_output_path(output_path):
        suffix = output_path.suffix or ".ply"
        return output_path.with_name(f"{output_path.stem}_processed{suffix}")

    def _connect_signals(self):
        self.previous_button.clicked.connect(self.previous_page)
        self.next_button.clicked.connect(self.next_page)
        self.mesh_view.load_button.clicked.connect(self.start_scan_loading)
        self.mesh_view.path_input.returnPressed.connect(self.start_scan_loading)
        self.plot_view.apply_orientation_button.clicked.connect(
            self.start_preprocessing
        )
        self.plot_view.update_crop_button.clicked.connect(
            self.start_processed_mesh_saving
        )
        self.config_view.auto_orient.stateChanged.connect(self.orientation_changed)
        for control in self.plot_view.rotation_inputs:
            control.valueChanged.connect(self.orientation_changed)
        for control in self.plot_view.flip_inputs:
            control.stateChanged.connect(self.orientation_changed)
        self.plot_view.fit_button.clicked.connect(self.start_fitting)
        self.plot_view.visualize_button.clicked.connect(self.start_visualization)
        self.plot_view.ground_truth_load_button.clicked.connect(
            self.start_ground_truth_loading
        )
        self.plot_view.ground_truth_path_input.textChanged.connect(
            self.ground_truth_path_changed
        )
        self.plot_view.full_body_visualize_button.clicked.connect(
            self.start_full_body_visualization
        )
        for control in self.plot_view.min_inputs + self.plot_view.max_inputs:
            control.valueChanged.connect(self.invalidate_fit)
        self.config_view.gender.currentTextChanged.connect(self.invalidate_fit)

    def next_page(self):
        index = self.stack.currentIndex()
        if index == 1:
            self.start_preprocessing()
        elif index == 0:
            self.stack.setCurrentIndex(index + 1)
            self._update_navigation()

    def previous_page(self):
        index = self.stack.currentIndex()
        if index > 0:
            self.stack.setCurrentIndex(index - 1)
        self._update_navigation()

    def start_scan_loading(self):
        path = Path(self.mesh_view.path_input.text()).expanduser()
        if not path.is_file():
            QMessageBox.critical(self, "Load error", f"Scan file not found: {path}")
            return
        self.pending_scan_path = path
        self._start_worker(
            ScanLoadWorker(str(path)), "Loading scan...", self._scan_loading_finished
        )

    @pyqtSlot(object)
    def _scan_loading_finished(self, mesh):
        self.scan_path = self.pending_scan_path
        if self.uses_default_output_path:
            self.output_path = self._default_output_path(self.scan_path)
        self.raw_mesh = mesh
        self.comparison_mesh = None
        self.preprocessed_mesh = None
        self.preprocessed_points = None
        self.original_to_processed_transform = None
        self.processed_to_output_transform = None
        self.orientation_dirty = False
        self.ground_truth_mesh = None
        self.ground_truth_path = None
        self.pending_ground_truth_path = None
        self.plot_view.ground_truth_path_input.clear()
        self.plot_view.clear_points()
        self.mesh_view.set_mesh(mesh, self.scan_path)
        self.invalidate_fit()

    def start_preprocessing(self):
        self.comparison_mesh = None
        self.preprocessed_mesh = None
        self.preprocessed_points = None
        self.original_to_processed_transform = None
        self.processed_to_output_transform = None
        self.plot_view.clear_points()
        self.invalidate_fit()
        settings = {
            **self.config_view.preprocessing_settings(),
            **self.plot_view.orientation_settings(),
        }
        self._start_worker(
            PreprocessWorker(
                self.raw_mesh,
                settings,
                self._processed_output_path(self.output_path),
            ),
            "Preprocessing scan and preparing crop view...",
            self._preprocessing_finished,
        )

    @pyqtSlot(object)
    def _preprocessing_finished(self, result):
        self.comparison_mesh = result["comparison_mesh"]
        self.preprocessed_mesh = result["processed_mesh"]
        self.preprocessed_points = result["points"]
        self.original_to_processed_transform = result[
            "original_to_processed_transform"
        ]
        self.processed_to_output_transform = result[
            "processed_to_output_transform"
        ]
        self.orientation_dirty = False
        self.plot_view.set_points(self.preprocessed_points)
        self.invalidate_fit()
        self.plot_view.summary.setText(
            f"Preprocessing complete; mesh saved to {result['output_path']}"
        )
        self.stack.setCurrentWidget(self.plot_view)

    def start_processed_mesh_saving(self):
        if self.preprocessed_mesh is None:
            return
        try:
            self.plot_view.cropped_points()
            minimum, maximum = self.plot_view.bounds()
        except ValueError as error:
            QMessageBox.critical(self, "Crop error", str(error))
            return

        self._start_worker(
            ProcessedMeshSaveWorker(
                self.preprocessed_mesh,
                minimum,
                maximum,
                self.processed_to_output_transform,
                self._processed_output_path(self.output_path),
            ),
            "Saving cropped processed scan...",
            self._processed_mesh_saving_finished,
        )

    @pyqtSlot(object)
    def _processed_mesh_saving_finished(self, result):
        self.plot_view.summary.setText(
            "Processed crop saved; "
            f"{result['face_count']:,} faces written to {result['output_path']}"
        )

    def invalidate_fit(self, *_):
        self.fit_result = None
        self.fit_cropped_points = None
        self.fit_cropped_mesh = None
        self.vedo_view.clear_result()
        self.plot_view.fit_button.setEnabled(self.preprocessed_points is not None)
        self.plot_view.visualize_button.setEnabled(False)
        self.plot_view.full_body_visualize_button.setEnabled(False)

    def orientation_changed(self, *_):
        self.orientation_dirty = True
        self.invalidate_fit()
        self.plot_view.fit_button.setEnabled(False)
        self.plot_view.update_crop_button.setEnabled(False)
        if self.preprocessed_points is not None:
            self.plot_view.summary.setText("Apply orientation before fitting.")

    def start_fitting(self):
        try:
            cropped_points = self.plot_view.cropped_points()
            minimum, maximum = self.plot_view.bounds()
            cropped_mesh = crop_trimesh(self.preprocessed_mesh, minimum, maximum)
            if cropped_mesh.is_empty:
                raise ValueError(
                    "The selected crop does not contain any complete mesh faces."
                )
        except Exception as error:
            QMessageBox.critical(self, "Fitting error", str(error))
            return

        self.invalidate_fit()
        self.fit_cropped_points = cropped_points
        self.fit_cropped_mesh = cropped_mesh
        settings = self.config_view.fitting_settings()
        settings["output_path"] = str(self.output_path)
        settings["output_transform"] = np.linalg.inv(
            self.original_to_processed_transform
        )
        self._start_worker(
            FitWorker(cropped_points.unsqueeze(0), settings),
            "Fitting STAR model...",
            self._fitting_finished,
        )

    @pyqtSlot(object)
    def _fitting_finished(self, result):
        result["comparison_mesh"] = self.comparison_mesh
        result["cropped_mesh"] = self.fit_cropped_mesh
        result["cropped_points"] = self.fit_cropped_points
        self.fit_result = result
        self.plot_view.summary.setText(
            "Fitting complete; full mesh saved to "
            f"{result['output_path']} and cropped torso saved to "
            f"{result['cropped_output_path']}"
        )

    def ground_truth_path_changed(self, value):
        path_text = value.strip()
        selected_path = Path(path_text).expanduser() if path_text else None
        if selected_path == self.ground_truth_path:
            return
        self.ground_truth_mesh = None
        self.ground_truth_path = None
        self.plot_view.full_body_visualize_button.setEnabled(False)

    def start_ground_truth_loading(self):
        path = Path(
            self.plot_view.ground_truth_path_input.text()
        ).expanduser()
        if not path.is_file():
            QMessageBox.critical(
                self,
                "Load error",
                f"Ground-truth scan file not found: {path}",
            )
            return
        self.pending_ground_truth_path = path
        self._start_worker(
            ScanLoadWorker(str(path)),
            "Loading full-body ground truth...",
            self._ground_truth_loading_finished,
        )

    @pyqtSlot(object)
    def _ground_truth_loading_finished(self, mesh):
        self.ground_truth_mesh = mesh
        self.ground_truth_path = self.pending_ground_truth_path
        self.plot_view.summary.setText(
            f"Full-body ground truth loaded: {self.ground_truth_path}"
        )

    def start_visualization(self):
        if self.fit_result is None:
            QMessageBox.information(self, "Fit required", "Fit the model first.")
            return
        self.vedo_view.clear_result()
        self._start_worker(
            VisualizationWorker(
                self.fit_result,
                self.fit_result["cropped_mesh"],
                self.config_view.visualization_resolution.value(),
            ),
            "Preparing visualization...",
            self._visualization_finished,
        )

    @pyqtSlot(object)
    def _visualization_finished(self, result):
        self.vedo_view.show_result(
            self.fit_result["comparison_mesh"],
            result["prediction_mesh"],
            self.fit_result["cropped_points"],
            result["losses"],
            self.fit_result["joints"],
            result["score"],
            full_prediction_mesh=result["full_prediction_mesh"],
            crop_bounds=self.fit_result["input_bounds"],
        )
        self.stack.setCurrentWidget(self.vedo_view)

    def start_full_body_visualization(self):
        if self.fit_result is None:
            QMessageBox.information(self, "Fit required", "Fit the model first.")
            return
        if self.ground_truth_mesh is None:
            QMessageBox.information(
                self,
                "Ground truth required",
                "Load a full-body ground-truth scan first.",
            )
            return
        self.vedo_view.clear_result()
        use_icp = self.plot_view.ground_truth_icp.isChecked()
        self._start_worker(
            FullBodyVisualizationWorker(
                self.fit_result,
                self.ground_truth_mesh,
                self.fit_result["cropped_mesh"],
                self.config_view.visualization_resolution.value(),
                use_icp=use_icp,
            ),
            (
                "Aligning and evaluating full-body ground truth..."
                if use_icp
                else "Evaluating full-body ground truth..."
            ),
            self._full_body_visualization_finished,
        )

    @pyqtSlot(object)
    def _full_body_visualization_finished(self, result):
        self.vedo_view.show_result(
            self.fit_result["comparison_mesh"],
            result["prediction_mesh"],
            result["target_points"],
            result["losses"],
            self.fit_result["joints"],
            result["score"],
            full_body_mesh=result["full_body_mesh"],
        )
        self.stack.setCurrentWidget(self.vedo_view)

    def _start_worker(self, worker, message, success_slot):
        if self.worker_thread is not None and self.worker_thread.isRunning():
            return
        self._set_busy(message)
        self.worker_thread = QThread(self)
        self.worker = worker
        self.worker_success_slot = success_slot
        worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(worker.run)
        worker.finished.connect(self._worker_succeeded)
        worker.failed.connect(self._worker_failed)
        worker.cancelled.connect(self._worker_cancelled)
        worker.finished.connect(self.worker_thread.quit)
        worker.failed.connect(self.worker_thread.quit)
        worker.cancelled.connect(self.worker_thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        worker.cancelled.connect(worker.deleteLater)
        self.worker_thread.finished.connect(self._worker_stopped)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.start()

    @pyqtSlot(object)
    def _worker_succeeded(self, result):
        try:
            self.worker_success_slot(result)
        except Exception as error:
            QMessageBox.critical(self, "Workflow error", str(error))
        finally:
            self._clear_busy()

    @pyqtSlot(str)
    def _worker_failed(self, message):
        self._clear_busy()
        QMessageBox.critical(self, "Workflow error", message)

    @pyqtSlot()
    def _worker_cancelled(self):
        self._clear_busy()

    @pyqtSlot()
    def cancel_worker(self):
        if self.worker is None:
            return
        self.worker.cancel()
        if self.progress is not None:
            self.progress.setLabelText("Cancelling...")
            self.progress.show()
        if self.cancel_button is not None:
            self.cancel_button.setEnabled(False)

    def _worker_stopped(self):
        self.worker = None
        self.worker_thread = None
        self.worker_success_slot = None
        gc.collect()

    def _set_busy(self, message):
        self.plot_view.fit_button.setEnabled(False)
        self.plot_view.visualize_button.setEnabled(False)
        self.plot_view.update_crop_button.setEnabled(False)
        self.plot_view.ground_truth_load_button.setEnabled(False)
        self.plot_view.full_body_visualize_button.setEnabled(False)
        self.previous_button.setEnabled(False)
        self.next_button.setEnabled(False)
        self.progress = QProgressDialog(message, None, 0, 0, self)
        self.cancel_button = QPushButton("Cancel", self.progress)
        self.progress.setCancelButton(self.cancel_button)
        self.progress.canceled.connect(self.cancel_worker)
        self.progress.setWindowTitle("Please wait")
        self.progress.setWindowModality(Qt.ApplicationModal)
        self.progress.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.progress.setMinimumDuration(0)
        self.progress.setAutoClose(False)
        self.progress.setAutoReset(False)
        self.progress.show()

    def _clear_busy(self):
        if self.progress is not None:
            self.progress.close()
            self.progress.deleteLater()
            self.progress = None
            self.cancel_button = None
        can_use_processed_scan = (
            self.preprocessed_points is not None and not self.orientation_dirty
        )
        self.plot_view.fit_button.setEnabled(can_use_processed_scan)
        self.plot_view.visualize_button.setEnabled(self.fit_result is not None)
        self.plot_view.update_crop_button.setEnabled(can_use_processed_scan)
        self.plot_view.ground_truth_load_button.setEnabled(
            bool(self.plot_view.ground_truth_path_input.text().strip())
        )
        self.plot_view.full_body_visualize_button.setEnabled(
            self.fit_result is not None and self.ground_truth_mesh is not None
        )
        self._update_navigation()

    def _update_navigation(self):
        index = self.stack.currentIndex()
        self.previous_button.setEnabled(index > 0)
        self.next_button.setEnabled(index < 2 and self.raw_mesh is not None)

    def closeEvent(self, event):
        if self.worker_thread is not None and self.worker_thread.isRunning():
            event.ignore()
            if self.progress is not None:
                self.progress.raise_()
                self.progress.activateWindow()
            return
        if self.mesh_view.plotter is not None:
            self.mesh_view.plotter.close()
        self.vedo_view.plotter.close()
        super().closeEvent(event)
