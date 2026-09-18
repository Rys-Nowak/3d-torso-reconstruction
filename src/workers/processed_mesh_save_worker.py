from pathlib import Path

from src.utils.data_preparation import crop_trimesh, transform_mesh

from .base import Worker


class ProcessedMeshSaveWorker(Worker):
    def __init__(self, mesh, minimum, maximum, output_transform, output_path):
        super().__init__()
        self.mesh = mesh
        self.minimum = minimum
        self.maximum = maximum
        self.output_transform = output_transform
        self.output_path = Path(output_path)

    def execute(self):
        self.check_cancelled()
        cropped_mesh = crop_trimesh(
            self.mesh,
            self.minimum,
            self.maximum,
        )
        if cropped_mesh.is_empty:
            raise ValueError(
                "The selected crop does not contain any complete mesh faces."
            )

        self.check_cancelled()
        output_mesh = transform_mesh(cropped_mesh, self.output_transform)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        output_mesh.export(self.output_path)
        self.check_cancelled()
        return {
            "output_path": self.output_path,
            "face_count": len(cropped_mesh.faces),
        }
