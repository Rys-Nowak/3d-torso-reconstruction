import numpy as np
import torch

from src.body_reconstruction.fit import evaluate_dist_full
from src.utils.data_preparation import icp_translate, transform_mesh

from .base import Worker
from .visualization_worker import build_prediction_mesh


class FullBodyVisualizationWorker(Worker):
    def __init__(
        self,
        fit_result,
        ground_truth_mesh,
        cropped_mesh,
        resolution,
        use_icp=False,
    ):
        super().__init__()
        self.fit_result = fit_result
        self.ground_truth_mesh = ground_truth_mesh
        self.cropped_mesh = cropped_mesh
        self.resolution = resolution
        self.use_icp = use_icp

    def execute(self):
        prediction_mesh = build_prediction_mesh(
            self.fit_result,
            self.resolution,
            self.check_cancelled,
        )
        self.check_cancelled()
        ground_truth_mesh = transform_mesh(
            self.ground_truth_mesh,
            np.linalg.inv(np.asarray(self.fit_result["output_transform"])),
        )
        if self.use_icp:
            full_body_mesh = icp_translate(
                ground_truth_mesh,
                prediction_mesh,
                [0, 0, 0],
            )
        else:
            full_body_mesh = ground_truth_mesh
        self.check_cancelled()
        evaluation_bounds = (
            self.cropped_mesh.vertices.min(axis=0),
            self.cropped_mesh.vertices.max(axis=0),
        )
        losses, _, eval_score = evaluate_dist_full(
            prediction_mesh,
            full_body_mesh,
            evaluation_bounds,
        )
        eval_target_points = torch.as_tensor(
            self.cropped_mesh.vertices, dtype=torch.float32
        )
        self.check_cancelled()
        return {
            "prediction_mesh": prediction_mesh,
            "full_body_mesh": full_body_mesh,
            "target_points": eval_target_points,
            "losses": losses,
            "score": eval_score,
        }
