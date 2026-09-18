import torch


def chamfer_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    distances = torch.cdist(x, y).square()
    x_to_y = distances.min(dim=2).values.mean(dim=1)
    y_to_x = distances.min(dim=1).values.mean(dim=1)
    return (x_to_y + y_to_x).mean()
