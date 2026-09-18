import gc
from pathlib import Path

import numpy as np
import torch
import trimesh

from STAR.star.pytorch.star import STAR
from src.body_reconstruction.fit import fit_star_to_partial_scan
from src.utils.data_preparation import crop_trimesh, star_crop_bounds

from .base import Worker


class FitWorker(Worker):
    def __init__(self, target, settings):
        super().__init__()
        self.target = target
        self.settings = settings

    def execute(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = None
        parameters = None
        prediction = None
        output_mesh = None
        cropped_output_mesh = None
        try:
            model = STAR(
                gender=self.settings["gender"],
                num_betas=self.settings["num_betas"],
            ).to(device)
            parameters = fit_star_to_partial_scan(
                self.target.to(device),
                model,
                self.settings["iterations"],
                num_betas=self.settings["num_betas"],
                lr=self.settings["learning_rate"],
                early_stopping=self.settings["early_stopping"],
                epsilon=self.settings["epsilon"],
                patience=self.settings["patience"],
                cancel_check=self.check_cancelled,
            )
            self.check_cancelled()
            prediction = model(*parameters).squeeze(0)
            canonical_vertices = prediction.detach().cpu().numpy()
            canonical_joints = (
                model.J_regressor @ prediction
            ).detach().cpu().numpy()
            output_path = Path(self.settings["output_path"])
            output_transform = np.asarray(self.settings["output_transform"])
            output_mesh = trimesh.Trimesh(
                canonical_vertices.copy(), np.asarray(model.f), process=False
            )
            input_points = self.target.detach().cpu().squeeze(0).numpy()
            input_bounds = star_crop_bounds(input_points, output_mesh, canonical_joints)
            cropped_output_mesh = crop_trimesh(output_mesh, *input_bounds)
            output_mesh.apply_transform(output_transform)
            cropped_output_mesh.apply_transform(output_transform)
            output_mesh.fix_normals()
            cropped_output_mesh.fix_normals()
            vertices = np.asarray(output_mesh.vertices).copy()
            faces = np.asarray(output_mesh.faces).copy()
            joints = trimesh.transform_points(
                canonical_joints,
                output_transform,
            )
            camera_location = trimesh.transform_points(
                np.array([[0.0, 0.0, 1.0]]),
                output_transform,
            )[0]
            if cropped_output_mesh.is_empty:
                raise ValueError("The fitting bounds produce an empty model crop.")
            suffix = output_path.suffix
            cropped_output_path = output_path.with_name(
                f"{output_path.stem}_cropped{suffix}"
            )
            self.check_cancelled()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_mesh.export(output_path)
            self.check_cancelled()
            cropped_output_mesh.export(cropped_output_path)
            self.check_cancelled()
            return {
                "vertices": canonical_vertices,
                "faces": faces,
                "joints": canonical_joints,
                "camera_location": np.array([0.0, 0.0, 1.0]),
                "output_vertices": vertices,
                "output_joints": joints,
                "output_camera_location": camera_location,
                "output_transform": output_transform,
                "input_bounds": input_bounds,
                "output_path": output_path,
                "cropped_output_path": cropped_output_path,
            }
        finally:
            del cropped_output_mesh, output_mesh, prediction, parameters, model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
