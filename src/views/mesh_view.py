from PyQt5.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
import vedo
from vedo import Mesh
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


class MeshView(QWidget):
    def __init__(self, mesh, scan_path):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.heading = QLabel()
        self.layout.addWidget(self.heading)

        path_row = QHBoxLayout()
        self.path_input = QLineEdit()
        self.browse_button = QPushButton("Browse")
        self.load_button = QPushButton("Load scan")
        self.browse_button.clicked.connect(self.select_path)
        path_row.addWidget(self.path_input, 1)
        path_row.addWidget(self.browse_button)
        path_row.addWidget(self.load_button)
        self.layout.addLayout(path_row)

        self.vtk_widget = None
        self.plotter = None
        if mesh is None:
            self.show_path_selection(scan_path)
        else:
            self.set_mesh(mesh, scan_path)

    def select_path(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select 3D scan",
            self.path_input.text(),
            "3D meshes (*.obj *.ply *.stl);;All files (*)",
        )
        if path:
            self.path_input.setText(path)

    def set_mesh(self, mesh, scan_path):
        if self.plotter is None:
            self.vtk_widget = QVTKRenderWindowInteractor(self)
            self.layout.addWidget(self.vtk_widget, 1)
            self.plotter = vedo.Plotter(qt_widget=self.vtk_widget, bg="white")
        self.heading.setText(f"1. Loaded scan: {scan_path}")
        self.path_input.setText(str(scan_path))
        actor = Mesh([mesh.vertices, mesh.faces]).c("dodgerblue")
        self.plotter.clear()
        self.plotter.show(actor, axes=1, resetcam=True)

    def show_path_selection(self, scan_path=""):
        self.heading.setText("1. Select scan")
        self.path_input.setText(str(scan_path or ""))
