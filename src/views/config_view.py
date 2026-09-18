from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class ConfigView(QWidget):
    EARLY_STOPPING_MAX_ITERATIONS = 500

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("2. Preprocessing and model configuration"))

        self.auto_orient = QCheckBox("Automatically orient scan")
        self.auto_orient.setChecked(True)
        self.auto_bed = QCheckBox("Automatically detect and remove bed")
        self.auto_bed.setChecked(True)
        self.auto_head = QCheckBox("Automatically detect and remove head")
        self.auto_head.setChecked(True)
        layout.addWidget(self.auto_orient)
        layout.addWidget(self.auto_bed)
        layout.addWidget(self.auto_head)

        parameters = QGroupBox("Model and fitting parameters")
        parameter_layout = QGridLayout(parameters)
        self.gender = QComboBox()
        self.gender.addItems(["neutral", "male", "female"])
        self.sampling_distance = self._double_control(0.01, 1000.0, 5.0, 2)
        self.sampling_distance.setSuffix(" mm")
        self.num_betas = self._integer_control(1, 300, 128)
        self.learning_rate = self._double_control(0.00001, 1.0, 0.01, 5)
        self.learning_rate.setSingleStep(0.001)

        fields = (
            ("Gender", self.gender),
            ("Surface sampling distance", self.sampling_distance),
            ("Shape parameters (betas)", self.num_betas),
            ("Learning rate", self.learning_rate),
        )
        for row, (label, control) in enumerate(fields):
            parameter_layout.addWidget(QLabel(label), row, 0)
            parameter_layout.addWidget(control, row, 1)

        self.early_stopping = QCheckBox("Enable early stopping")
        parameter_layout.addWidget(self.early_stopping, 4, 0, 1, 2)

        self.iteration_inputs = []
        self.manual_iteration_widgets = []
        iteration_defaults = (
            ("Translation iterations", 5),
            ("Pose iterations", 50),
            ("Shape iterations", 70),
            ("Shape tuning iterations", 20),
        )
        for row, (label, default) in enumerate(iteration_defaults, start=5):
            control = self._integer_control(1, 10000, default)
            label_widget = QLabel(label)
            self.iteration_inputs.append(control)
            self.manual_iteration_widgets.extend((label_widget, control))
            parameter_layout.addWidget(label_widget, row, 0)
            parameter_layout.addWidget(control, row, 1)

        self.early_stopping_epsilon = self._double_control(
            0.00000001, 1.0, 0.00001, 8
        )
        self.early_stopping_patience = self._integer_control(1, 10000, 10)
        self.early_stopping_widgets = (
            QLabel("Early stopping epsilon"),
            self.early_stopping_epsilon,
            QLabel("Iterations without improvement"),
            self.early_stopping_patience,
        )
        parameter_layout.addWidget(self.early_stopping_widgets[0], 5, 0)
        parameter_layout.addWidget(self.early_stopping_widgets[1], 5, 1)
        parameter_layout.addWidget(self.early_stopping_widgets[2], 6, 0)
        parameter_layout.addWidget(self.early_stopping_widgets[3], 6, 1)

        self.visualization_resolution = self._integer_control(1, 5, 3)
        self.visualization_resolution.setToolTip(
            "1 uses no subdivisions; 5 uses four subdivisions"
        )
        resolution_row = 9
        parameter_layout.addWidget(QLabel("Visualization resolution"), resolution_row, 0)
        parameter_layout.addWidget(self.visualization_resolution, resolution_row, 1)
        parameter_layout.setColumnStretch(2, 1)
        layout.addWidget(parameters)
        layout.addStretch()
        self.early_stopping.toggled.connect(self._set_early_stopping_enabled)
        self._set_early_stopping_enabled(False)

    @staticmethod
    def _integer_control(minimum, maximum, value):
        control = QSpinBox()
        control.setRange(minimum, maximum)
        control.setValue(value)
        return control

    @staticmethod
    def _double_control(minimum, maximum, value, decimals):
        control = QDoubleSpinBox()
        control.setDecimals(decimals)
        control.setRange(minimum, maximum)
        control.setValue(value)
        return control

    def preprocessing_settings(self):
        return {
            "auto_orient": self.auto_orient.isChecked(),
            "auto_bed": self.auto_bed.isChecked(),
            "auto_head": self.auto_head.isChecked(),
            "sampling_distance": self.sampling_distance.value() / 1000.0,
        }

    def _set_early_stopping_enabled(self, enabled):
        for widget in self.manual_iteration_widgets:
            widget.setVisible(not enabled)
        for widget in self.early_stopping_widgets:
            widget.setVisible(enabled)

    def fitting_settings(self):
        early_stopping = self.early_stopping.isChecked()
        iterations = (
            (self.EARLY_STOPPING_MAX_ITERATIONS,) * len(self.iteration_inputs)
            if early_stopping
            else tuple(control.value() for control in self.iteration_inputs)
        )
        return {
            "gender": self.gender.currentText(),
            "num_betas": self.num_betas.value(),
            "learning_rate": self.learning_rate.value(),
            "iterations": iterations,
            "early_stopping": early_stopping,
            "epsilon": self.early_stopping_epsilon.value(),
            "patience": self.early_stopping_patience.value(),
        }
