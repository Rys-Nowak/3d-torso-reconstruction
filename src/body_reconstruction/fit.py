import torch
import torch.optim as optim
import open3d as o3d
import numpy as np

from ..utils.data_preparation import crop_pcd, process_pcd
from ..utils.metrics import chamfer_distance


def fit_star_to_partial_scan(
    target_points,
    star_model,
    num_iters,
    lr=0.01,
    num_betas=10,
    l2=(1, 0.05, 0.1),
    early_stopping=False,
    epsilon=0.000001,
    patience=10,
    cancel_check=None,
    loss_history=None,
):
    if early_stopping and epsilon < 0:
        raise ValueError("Early stopping epsilon must be non-negative.")
    if early_stopping and patience < 1:
        raise ValueError("Early stopping patience must be at least 1.")
    device = target_points.device
    pose = torch.zeros((1, 72), requires_grad=False, device=device)
    betas = torch.zeros((1, num_betas), requires_grad=False, device=device)
    trans = torch.zeros((1, 3), requires_grad=True, device=device)

    pose[:, 17*3:18*3] = torch.tensor([0.5, 0.5, 1.2])
    pose[:, 16*3:17*3] = torch.tensor([0.5, -0.5, -1.2])

    optimizer = optim.Adam([trans], lr=lr)
    print("Starting Translation Optimization...")
    stopping = {
        "early_stopping": early_stopping,
        "epsilon": epsilon,
        "patience": patience,
        "cancel_check": cancel_check,
    }
    if cancel_check:
        cancel_check()
    train_star_to_partial_scan(
        target_points, star_model, num_iters[0], pose, betas, trans, optimizer,
        stage="translation", loss_history=loss_history, **stopping,
    )
    print(f"Translation optimization complete. Found: {trans}")

    if cancel_check:
        cancel_check()
    optimizer = optim.Adam([pose, trans], lr=lr)
    pose = pose.requires_grad_()
    print("Starting Pose Optimization...")
    train_star_to_partial_scan(
        target_points, star_model, num_iters[1], pose, betas, trans, optimizer,
        l2=l2, stage="pose", loss_history=loss_history, **stopping,
    )
    print(f"Pose optimization complete. Found: {pose}")

    if cancel_check:
        cancel_check()
    optimizer = optim.Adam([pose, betas, trans], lr=lr)
    betas = betas.requires_grad_()
    print("Starting Shape Optimization...")
    train_star_to_partial_scan(
        target_points, star_model, num_iters[2], pose, betas, trans, optimizer,
        l2=l2, stage="shape", loss_history=loss_history, **stopping,
    )

    trans = trans.detach()
    pose = pose.detach()
    if len(num_iters) > 3:
        if cancel_check:
            cancel_check()
        betas = betas.requires_grad_()
        optimizer = optim.Adam([betas], lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_iters[3], lr/10)
        print("Tuning Shape...")
        train_star_to_partial_scan(
            target_points, star_model, num_iters[3], pose, betas, trans, optimizer,
            scheduler=scheduler, l2=l2, stage="shape_tuning",
            loss_history=loss_history, **stopping,
        )
    
    print(f"Shape optimization complete. Found: {betas}")

    print("Optimization Complete")
    return pose, betas.detach(), trans


def train_star_to_partial_scan(
    target_partial_points,
    star_model,
    num_iters,
    pose,
    betas,
    trans,
    optimizer,
    scheduler=None,
    l2=(1, 0.05, 0.1),
    early_stopping=False,
    epsilon=0.000001,
    patience=10,
    cancel_check=None,
    stage=None,
    loss_history=None,
):
    best_loss = float("inf")
    iterations_without_improvement = 0
    for i in range(num_iters):
        if cancel_check:
            cancel_check()
        optimizer.zero_grad()
        predicted_vertices = star_model(pose=pose, betas=betas, trans=trans)
        predicted_partial, _ = process_pcd(predicted_vertices,
                                        target_partial_points.squeeze(0).min(dim=0).values,
                                        target_partial_points.squeeze(0).max(dim=0).values)
        loss_chamfer = chamfer_distance(predicted_partial, target_partial_points)
        l2_legs = torch.mean(pose[:, 3*1:3*3]) ** 2 +\
            torch.mean(pose[:, 3*4:3*6]) ** 2 +\
            torch.mean(pose[:, 3*7:3*9]) ** 2 +\
            torch.mean(pose[:, 3*10:3*12]) ** 2
        l2_arms = torch.mean(pose[:, 3*18:3*24]) ** 2 + torch.mean(pose[:, 3*15:3*16]) ** 2
        l2_head = torch.mean(pose[:, 3*12:3*13]) ** 2
        total_loss = loss_chamfer + l2[0] * l2_legs + l2[1] * l2_arms + l2[2] * l2_head
        total_loss.backward()
        optimizer.step()
        if cancel_check:
            cancel_check()
        if scheduler:
            scheduler.step()
        
        loss_value = total_loss.detach().item()
        if loss_history is not None:
            loss_history.append({
                "stage": stage,
                "iteration": i + 1,
                "chamfer_loss": loss_chamfer.detach().item(),
                "total_loss": loss_value,
            })
        print(f"Iteration [{i+1}/{num_iters}] - Chamfer Loss: {loss_chamfer.item():.6f}")

        if not early_stopping:
            continue
        if best_loss - loss_value > epsilon:
            best_loss = loss_value
            iterations_without_improvement = 0
        else:
            iterations_without_improvement += 1
        if iterations_without_improvement >= patience:
            print(
                "Early stopping after "
                f"{i + 1} iterations (improvement <= {epsilon:.8g} for "
                f"{patience} iterations)."
            )
            break


def evaluate_dist_full(pred_mesh, target_mesh, eval_bounds):
    _, mask = crop_pcd(torch.FloatTensor(pred_mesh.vertices).unsqueeze(0),
        target_mesh.bounds[0], target_mesh.bounds[1])
    _, eval_mask = crop_pcd(torch.FloatTensor(pred_mesh.vertices).unsqueeze(0), *eval_bounds)
    mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(target_mesh.vertices, dtype=o3d.core.Dtype.Float32),
        o3d.core.Tensor(target_mesh.faces, dtype=o3d.core.Dtype.UInt32),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    distances = scene.compute_distance(
        o3d.core.Tensor(pred_mesh.vertices, dtype=o3d.core.Dtype.Float32)
    ).numpy()
    distances[np.logical_not(mask)] = 0.0
    return distances, np.mean(distances[mask]), np.mean(distances[eval_mask])


def evaluate_dist(pred_mesh, target_mesh, camera_location=(0, 0, 1)):
    _, mask = process_pcd(torch.FloatTensor(pred_mesh.vertices).unsqueeze(0),
        target_mesh.bounds[0], target_mesh.bounds[1],
        camera_location=camera_location)
    mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(target_mesh.vertices, dtype=o3d.core.Dtype.Float32),
        o3d.core.Tensor(target_mesh.faces, dtype=o3d.core.Dtype.UInt32),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    distances = scene.compute_distance(
        o3d.core.Tensor(pred_mesh.vertices, dtype=o3d.core.Dtype.Float32)
    ).numpy()
    distances[np.logical_not(mask)] = 0.0
    return distances, np.mean(distances[mask])
