from pathlib import Path

import numpy as np
import trimesh

from src.utils.data_preparation import preprocess_scan, transform_mesh

from .base import Worker


class PreprocessWorker(Worker):
    def __init__(self, raw_mesh, settings, output_path):
        super().__init__()
        self.raw_mesh = raw_mesh
        self.settings = settings
        self.output_path = Path(output_path)

    def execute(self):
        comparison_mesh, processed_mesh, points, transform = preprocess_scan(
            self.raw_mesh,
            cancel_check=self.check_cancelled,
            return_transform=True,
            **self.settings,
        )
        self.check_cancelled()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        manual_rotation = trimesh.transformations.euler_matrix(
            *np.radians(self.settings.get("rotations", (0.0, 0.0, 0.0))),
            axes="sxyz",
        )
        processed_to_output_transform = (
            manual_rotation @ np.linalg.inv(transform)
        )
        output_mesh = transform_mesh(
            processed_mesh,
            processed_to_output_transform,
        )
        output_mesh.export(self.output_path)
        self.check_cancelled()
        return {
            "comparison_mesh": comparison_mesh,
            "processed_mesh": processed_mesh,
            "points": points,
            "original_to_processed_transform": transform,
            "processed_to_output_transform": processed_to_output_transform,
            "output_path": self.output_path,
        }
