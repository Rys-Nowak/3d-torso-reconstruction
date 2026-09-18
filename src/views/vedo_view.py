from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget
import vedo
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from src.utils.visualization import show_meshes_vedo


class VedoView(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        self.heading = QLabel("4. Final STAR fit")
        layout.addWidget(self.heading)
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.vtk_widget, 1)
        self.plotter = vedo.Plotter(qt_widget=self.vtk_widget, bg="white")

    def clear_result(self):
        self.plotter.clear(deep=True)

    def show_result(
        self,
        raw_mesh,
        prediction_mesh,
        cropped_points,
        losses,
        joints,
        score,
        full_body_mesh=None,
        full_prediction_mesh=None,
        crop_bounds=None,
    ):
        title = (
            "Full-body evaluation"
            if full_body_mesh is not None
            else "Final STAR fit"
        )
        self.heading.setText(
            f"4. {title} ({len(cropped_points):,} target points)"
        )
        show_meshes_vedo(
            raw_mesh,
            prediction_mesh,
            full_prediction_mesh=full_prediction_mesh,
            full_body_mesh=full_body_mesh,
            cropped_points=cropped_points.unsqueeze(0),
            crop_bounds=crop_bounds,
            losses=losses,
            joints=joints,
            plotter=self.plotter,
            score=score,
        )
