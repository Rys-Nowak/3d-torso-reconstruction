from src.utils.data_preparation import ensure_trimesh, load_scan

from .base import Worker


class ScanLoadWorker(Worker):
    def __init__(self, path):
        super().__init__()
        self.path = path

    def execute(self):
        self.check_cancelled()
        mesh = ensure_trimesh(load_scan(self.path))
        self.check_cancelled()
        return mesh
