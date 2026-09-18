import torch
import trimesh

from src.body_reconstruction.fit import evaluate_dist
from src.utils.data_preparation import crop_trimesh

from .base import Worker


def build_prediction_mesh(fit_result, resolution, cancel_check):
    cancel_check()
    prediction_mesh = trimesh.Trimesh(
        fit_result["vertices"],
        fit_result["faces"],
        process=False,
    )
    for _ in range(resolution - 1):
        cancel_check()
        prediction_mesh = prediction_mesh.subdivide()
    cancel_check()
    return prediction_mesh


class VisualizationWorker(Worker):
    def __init__(self, fit_result, cropped_mesh, resolution):
        super().__init__()
        self.fit_result = fit_result
        self.cropped_mesh = cropped_mesh
        self.resolution = resolution

    def execute(self):
        prediction_mesh = build_prediction_mesh(
            self.fit_result,
            self.resolution,
            self.check_cancelled,
        )
        cropped_prediction_mesh = crop_trimesh(
            prediction_mesh, *self.fit_result["input_bounds"]
        )
        losses, score = evaluate_dist(
            cropped_prediction_mesh,
            self.cropped_mesh,
            camera_location=self.fit_result["camera_location"],
        )
        self.check_cancelled()
        return {
            "prediction_mesh": cropped_prediction_mesh,
            "full_prediction_mesh": prediction_mesh,
            "losses": losses,
            "score": score,
        }
