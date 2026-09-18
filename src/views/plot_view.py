import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.utils.data_preparation import crop_pcd
from src.utils.visualization import plot_crop_comparison


class PlotView(QWidget):
    AXES = ("X", "Y", "Z")
    AXIS_LABELS = (
        "X (left-right)",
        "Y (legs-head)",
        "Z (back-front)",
    )

    def __init__(self):
        super().__init__()
        self.points = None
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("3. Cropped point-cloud comparison"))
        layout.addWidget(QLabel(", ".join(self.AXIS_LABELS)))

        controls = QHBoxLayout()
        controls.addWidget(self._create_orientation_controls())
        controls.addWidget(self._create_crop_controls(), 1)
        layout.addLayout(controls)

        self.summary = QLabel()
        layout.addWidget(self.summary)
        self.figure = Figure(constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas, 1)

        ground_truth = QHBoxLayout()
        ground_truth.addWidget(QLabel("Full-body ground truth"))
        self.ground_truth_path_input = QLineEdit()
        self.ground_truth_path_input.setPlaceholderText("Select a mesh file")
        self.ground_truth_browse_button = QPushButton("Browse")
        self.ground_truth_load_button = QPushButton("Load")
        self.ground_truth_load_button.setEnabled(False)
        self.ground_truth_icp = QCheckBox("Align with ICP")
        self.ground_truth_icp.setChecked(False)
        ground_truth.addWidget(self.ground_truth_path_input, 1)
        ground_truth.addWidget(self.ground_truth_browse_button)
        ground_truth.addWidget(self.ground_truth_load_button)
        ground_truth.addWidget(self.ground_truth_icp)
        layout.addLayout(ground_truth)

        actions = QHBoxLayout()
        actions.addStretch()
        self.fit_button = QPushButton("Fit model")
        self.fit_button.setEnabled(False)
        self.visualize_button = QPushButton("Show fitting result")
        self.visualize_button.setEnabled(False)
        self.full_body_visualize_button = QPushButton(
            "Show gorund-truth evaluation"
        )
        self.full_body_visualize_button.setEnabled(False)
        actions.addWidget(self.fit_button)
        actions.addWidget(self.visualize_button)
        actions.addWidget(self.full_body_visualize_button)
        layout.addLayout(actions)

        self.ground_truth_browse_button.clicked.connect(
            self.select_ground_truth_path
        )
        self.ground_truth_path_input.textChanged.connect(
            lambda value: self.ground_truth_load_button.setEnabled(
                bool(value.strip())
            )
        )

    def _create_orientation_controls(self):
        group = QGroupBox("Orientation")
        layout = QGridLayout(group)
        layout.addWidget(QLabel(", ".join(self.AXIS_LABELS)), 0, 0, 1, 3)
        layout.addWidget(QLabel("Axis"), 1, 0)
        layout.addWidget(QLabel("Rotation"), 1, 1)
        layout.addWidget(QLabel("Flip"), 1, 2)
        self.rotation_inputs = []
        self.flip_inputs = []
        for row, axis_label in enumerate(self.AXIS_LABELS, start=2):
            rotation = QDoubleSpinBox()
            rotation.setDecimals(1)
            rotation.setRange(-360.0, 360.0)
            rotation.setSingleStep(90.0)
            rotation.setSuffix(" deg")
            flip = QCheckBox()
            self.rotation_inputs.append(rotation)
            self.flip_inputs.append(flip)
            layout.addWidget(QLabel(axis_label), row, 0)
            layout.addWidget(rotation, row, 1)
            layout.addWidget(flip, row, 2)
        self.reset_orientation_button = QPushButton("Reset")
        self.apply_orientation_button = QPushButton("Apply orientation")
        self.reset_orientation_button.clicked.connect(self.reset_orientation)
        layout.addWidget(self.reset_orientation_button, 5, 0)
        layout.addWidget(self.apply_orientation_button, 5, 1, 1, 2)
        return group

    def _create_crop_controls(self):
        group = QGroupBox("Manual crop")
        layout = QGridLayout(group)
        layout.addWidget(QLabel("Axis"), 0, 0)
        layout.addWidget(QLabel("Minimum"), 0, 1)
        layout.addWidget(QLabel("Maximum"), 0, 2)
        self.min_inputs = []
        self.max_inputs = []
        for row, axis_label in enumerate(self.AXIS_LABELS, start=1):
            minimum = self._bound_input()
            maximum = self._bound_input()
            self.min_inputs.append(minimum)
            self.max_inputs.append(maximum)
            layout.addWidget(QLabel(axis_label), row, 0)
            layout.addWidget(minimum, row, 1)
            layout.addWidget(maximum, row, 2)
        self.update_crop_button = QPushButton("Update crop")
        self.update_crop_button.setEnabled(False)
        self.update_crop_button.clicked.connect(self.update_plot)
        layout.addWidget(self.update_crop_button, 1, 3, 3, 1)
        layout.setColumnStretch(4, 1)
        return group

    @staticmethod
    def _bound_input():
        control = QDoubleSpinBox()
        control.setDecimals(4)
        control.setRange(-10000.0, 10000.0)
        control.setSingleStep(0.01)
        return control

    def orientation_settings(self):
        return {
            "rotations": tuple(control.value() for control in self.rotation_inputs),
            "flips": tuple(control.isChecked() for control in self.flip_inputs),
        }

    def reset_orientation(self):
        for control in self.rotation_inputs:
            control.setValue(0.0)
        for control in self.flip_inputs:
            control.setChecked(False)

    def select_ground_truth_path(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select full-body ground truth",
            self.ground_truth_path_input.text(),
            "3D meshes (*.obj *.ply *.stl);;All files (*)",
        )
        if path:
            self.ground_truth_path_input.setText(path)

    def clear_points(self):
        self.points = None
        self.update_crop_button.setEnabled(False)
        self.figure.clear()
        self.canvas.draw_idle()
        self.summary.clear()

    def set_points(self, points):
        self.points = points.detach().cpu()
        self.update_crop_button.setEnabled(True)
        minimum = self.points.min(dim=0).values.numpy()
        maximum = self.points.max(dim=0).values.numpy()
        padding = np.maximum((maximum - minimum) * 0.01, 1e-4)
        for axis in range(3):
            self.min_inputs[axis].setValue(float(minimum[axis] - padding[axis]))
            self.max_inputs[axis].setValue(float(maximum[axis] + padding[axis]))
        self.update_plot()

    def bounds(self):
        minimum = np.array([control.value() for control in self.min_inputs])
        maximum = np.array([control.value() for control in self.max_inputs])
        if np.any(minimum >= maximum):
            raise ValueError("Every minimum crop bound must be below its maximum.")
        return minimum, maximum

    def cropped_points(self):
        if self.points is None:
            raise ValueError("No preprocessed point cloud is available.")
        minimum, maximum = self.bounds()
        cropped, _ = crop_pcd(self.points.unsqueeze(0), minimum, maximum)
        if cropped.shape[1] == 0:
            raise ValueError("The selected crop does not contain any points.")
        return cropped.squeeze(0)

    def update_plot(self):
        if self.points is None:
            return
        try:
            cropped = self.cropped_points().numpy()
            minimum, maximum = self.bounds()
        except ValueError as error:
            self.summary.setText(str(error))
            return
        original = self.points.numpy()
        plot_crop_comparison(self.figure, original, cropped, minimum, maximum)
        self.summary.setText(f"{len(cropped):,} of {len(original):,} points selected")
        self.canvas.draw_idle()
