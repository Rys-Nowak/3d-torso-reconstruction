import numpy as np
import open3d as o3d
import trimesh
import torch


BOUNDING_BOX_MAX_Z_PADDING = 0.05


def pad_bounding_box_max_z(max_bound):
    if torch.is_tensor(max_bound):
        padded = max_bound.clone()
        if not padded.is_floating_point():
            padded = padded.float()
    else:
        padded = np.array(max_bound, dtype=float, copy=True)
    padded[2] += BOUNDING_BOX_MAX_Z_PADDING
    return padded


def remove_hidden_points(points, camera_location=[0, 0, 1], radius=500):
    o3dPcd = o3d.geometry.PointCloud()
    o3dPcd.points = o3d.utility.Vector3dVector(points)
    _, pt_map = o3dPcd.hidden_point_removal(camera_location, radius)
    return pt_map


def crop_pcd(
    points, min_bound=np.array([-0.4, -0.4, -0.5]), max_bound=np.array([0.5, 0.6, 0.5])
):
    max_bound = pad_bounding_box_max_z(max_bound)
    mask = (
        (points[:, :, 0] >= min_bound[0]) & (points[:, :, 0] <= max_bound[0]) &
        (points[:, :, 1] >= min_bound[1]) & (points[:, :, 1] <= max_bound[1]) &
        (points[:, :, 2] >= min_bound[2]) & (points[:, :, 2] <= max_bound[2])
    )
    return points[mask].unsqueeze(0), mask.squeeze(0).numpy().astype(np.bool8)


def process_pcd(
    points,
    min_bound,
    max_bound,
    camera_location=(0, 0, 1),
):
    visible_indices = remove_hidden_points(
        points.detach().squeeze(0).numpy(),
        camera_location=camera_location,
    )
    visible_mask = np.zeros(points.shape[1], dtype=bool)
    visible_mask[visible_indices] = True
    _, crop_mask = crop_pcd(points, min_bound, max_bound)
    mask = np.logical_and(visible_mask, crop_mask)
    return points[torch.tensor(mask[np.newaxis])].unsqueeze(0), mask


def load_scan(path):
    return trimesh.load(path)


def ensure_trimesh(loaded):
    if isinstance(loaded, trimesh.Scene):
        meshes = list(loaded.geometry.values())
        if not meshes:
            raise ValueError("The scan does not contain any mesh geometry.")
        return trimesh.util.concatenate(meshes)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError("The selected file is not a triangle mesh.")
    return loaded


def transform_mesh(mesh, transform):
    transformed = mesh.copy()
    transformed.apply_transform(transform)
    transformed.fix_normals()
    return transformed


def transform_points(points, transform):
    is_tensor = torch.is_tensor(points)
    source = points.detach().cpu().numpy() if is_tensor else np.asarray(points)
    transformed = trimesh.transform_points(source, transform)
    if not is_tensor:
        return transformed
    return torch.as_tensor(
        transformed,
        dtype=points.dtype,
        device=points.device,
    )


def apply_orientation(
    mesh,
    auto_orient=True,
    rotations=(0, 0, 0),
    flips=(False,) * 3,
    return_transform=False,
):
    transform = np.eye(4)
    if auto_orient:
        transform = automatic_orientation_transform(mesh) @ transform
    rotation_matrix = trimesh.transformations.euler_matrix(
        *np.radians(rotations), axes="sxyz"
    )
    transform = rotation_matrix @ transform
    flip_matrix = np.diag(
        [-1.0 if flip else 1.0 for flip in flips] + [1.0]
    )
    transform = flip_matrix @ transform

    transformed_vertices = trimesh.transform_points(mesh.vertices, transform)
    centering = trimesh.transformations.translation_matrix(
        -transformed_vertices.mean(axis=0)
    )
    transform = centering @ transform

    oriented_mesh = mesh.copy()
    oriented_mesh.apply_transform(transform)
    oriented_mesh.fix_normals()
    if return_transform:
        return oriented_mesh, transform
    return oriented_mesh


def preprocess_scan(
    raw_mesh,
    auto_orient=True,
    auto_bed=True,
    auto_head=True,
    rotations=(0, 0, 0),
    flips=(False,) * 3,
    sampling_distance=0.005,
    sample_count=50_000,
    cancel_check=None,
    return_transform=False,
):
    if cancel_check:
        cancel_check()
    mesh, original_to_processed = apply_orientation(
        raw_mesh,
        auto_orient,
        rotations,
        flips,
        return_transform=True,
    )
    comparison_mesh = mesh.copy()
    if cancel_check:
        cancel_check()
    sampled, _ = trimesh.sample.sample_surface_even(
        mesh, sample_count, sampling_distance
    )
    points = torch.as_tensor(sampled, dtype=torch.float32)
    if cancel_check:
        cancel_check()

    bed_plane = None
    if auto_bed:
        points, bed_plane, bed_flip = crop_bed(points, return_transform=True)
        if bed_flip is not None:
            mesh = transform_mesh(mesh, bed_flip)
            comparison_mesh = transform_mesh(comparison_mesh, bed_flip)
            original_to_processed = bed_flip @ original_to_processed
        if cancel_check:
            cancel_check()
    neck_position = None
    if auto_head:
        points, neck_position = decapitate(points)
        points = flip_body_pcd(points, neck_position)
        if neck_position < 0:
            comparison_mesh = flip_mesh(comparison_mesh)
            body_flip = np.diag([-1.0, -1.0, 1.0, 1.0])
            original_to_processed = body_flip @ original_to_processed
        if cancel_check:
            cancel_check()

    processed_mesh = crop_mesh_by_plane(mesh, bed_plane)
    if neck_position is not None:
        if neck_position < 0:
            processed_mesh = flip_mesh(processed_mesh)
            neck_position = -neck_position
        processed_mesh = decapitate_mesh(processed_mesh, neck_position)
    if cancel_check:
        cancel_check()

    result = (comparison_mesh, processed_mesh, points)
    if return_transform:
        return result + (original_to_processed,)
    return result


def crop_trimesh(mesh, bbox_min, bbox_max):
    bbox_max = pad_bounding_box_max_z(bbox_max)
    inside = np.all(
        (mesh.vertices >= bbox_min) &
        (mesh.vertices <= bbox_max),
        axis=1
    )
    face_mask = inside[mesh.faces].all(axis=1)
    return mesh.submesh([face_mask], append=True)


def star_crop_bounds(points, mesh, joints):
    bounds = [points.min(axis=0), points.max(axis=0)]
    bounds[0][2] = mesh.bounds[0][2]
    bounds[0][0] = max(bounds[0][0], joints[[16, 17], 0].min() - 0.05)
    bounds[1][0] = min(bounds[1][0], joints[[16, 17], 0].max() + 0.05)
    bounds[0][1] = max(bounds[0][1], joints[[1, 2], 1].min() - 0.05)
    bounds[1][1] = min(bounds[1][1], joints[[12], 1].max() + 0.05)
    return bounds


def icp_translate(mesh_to_translate, reference_mesh, crop_rates=[0,0,0]):
    mesh = mesh_to_translate.copy()

    components = mesh.split(only_watertight=False)
    mesh = max(components, key=lambda m: len(m.faces))

    mesh = mesh.apply_translation(-mesh.vertices.mean(axis=0))

    source_points = mesh.sample(5000)
    reference_points = reference_mesh.sample(5000)
    T, _, _ = trimesh.registration.icp(
        source_points,
        reference_points,
        max_iterations=200,
    )
    mesh.apply_transform(T)

    min_values = mesh.vertices.min(axis=0)
    min_values -= min_values * crop_rates
    max_values = mesh.vertices.max(axis=0)
    max_values -= max_values * crop_rates
    mesh = crop_trimesh(mesh, min_values, max_values)
    
    return mesh


def robust_orientation_vertices(vertices, lower=2.0, upper=98.0, min_count=100):
    vertices = np.asarray(vertices, dtype=float)
    finite_vertices = vertices[np.isfinite(vertices).all(axis=1)]
    if len(finite_vertices) < min_count:
        return finite_vertices

    low = np.percentile(finite_vertices, lower, axis=0)
    high = np.percentile(finite_vertices, upper, axis=0)
    mask = np.all(
        (finite_vertices >= low) & (finite_vertices <= high),
        axis=1,
    )
    trimmed = finite_vertices[mask]
    if len(trimmed) < min_count:
        return finite_vertices
    return trimmed


def orient_axis_sign(axis):
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return axis

    axis = axis / norm
    dominant_coordinate = np.argmax(np.abs(axis))
    if axis[dominant_coordinate] < 0:
        axis = -axis
    return axis


def automatic_orientation_transform(mesh):
    vertices = robust_orientation_vertices(mesh.vertices)
    transform = np.eye(4)
    if len(vertices) < 3:
        return transform

    centered = vertices - np.median(vertices, axis=0)
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    if not np.isfinite(covariance).all():
        return transform

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]

    body_axis = orient_axis_sign(eigenvectors[:, order[0]])
    width_axis = orient_axis_sign(eigenvectors[:, order[1]])
    depth_axis = np.cross(width_axis, body_axis)
    depth_axis_norm = np.linalg.norm(depth_axis)
    if depth_axis_norm < 1e-12:
        return transform
    depth_axis = depth_axis / depth_axis_norm
    width_axis = np.cross(body_axis, depth_axis)
    width_axis = width_axis / np.linalg.norm(width_axis)

    source_basis = np.column_stack((width_axis, body_axis, depth_axis))
    if np.linalg.det(source_basis) < 0:
        depth_axis = -depth_axis
        source_basis = np.column_stack((width_axis, body_axis, depth_axis))

    transform[:3, :3] = source_basis.T
    return transform


def crop_bed(points, return_transform=False):
    o3d_pcd = o3d.geometry.PointCloud(
        points=o3d.utility.Vector3dVector(points),
    )
    plane_model, _ = o3d_pcd.segment_plane(
        distance_threshold=0.007,
        ransac_n=3,
        num_iterations=20000,
    )
    [a, b, c, d] = plane_model
    if c < 0.98:
        result = (points, None)
        return result + (None,) if return_transform else result

    dist = points @ np.array([a, b, c]) + d
    positive_mask = dist > 0.02
    negative_mask = dist < -0.02
    keep_negative_side = negative_mask.sum() > positive_mask.sum()
    mask = negative_mask if keep_negative_side else positive_mask
    points = points[mask]

    bed_flip = None
    if keep_negative_side:
        bed_flip = np.diag([-1.0, 1.0, -1.0, 1.0])
        points = transform_points(points, bed_flip)
        plane_model = -(bed_flip @ np.asarray(plane_model))

    result = (points, plane_model)
    return result + (bed_flip,) if return_transform else result


def trim_outliers(tensor, axis=0, lower=5, upper=95):
    lo = np.percentile(tensor, lower, axis=axis, keepdims=True)
    hi = np.percentile(tensor, upper, axis=axis, keepdims=True)
    mask = (tensor.numpy() >= lo) & (tensor.numpy() <= hi)
    return tensor[mask]


def decapitate(points, n_slices=100, body_axis=np.array([0, 1, 0]), head_offset=0.03):
    proj = points @ body_axis
    bins = np.linspace(proj.min(), proj.max(), n_slices + 1)
    widths = []
    centers = []
    for i in range(n_slices):
        lo = bins[i]
        hi = bins[i + 1]
        mask = (proj >= lo) & (proj < hi)
        pts = points[mask]
        if len(pts) < 20:
            widths.append(np.nan)
            centers.append((lo + hi) / 2)
            continue

        centers.append((lo + hi) / 2)
        x = trim_outliers(pts[:, 0])
        width = x.abs().max()
        widths.append(width)

    widths = np.array(widths)
    centers = np.array(centers)

    search_region = slice(int(0.1*n_slices), int(0.9*n_slices))
    head_offset = int(n_slices*head_offset)
    neck_idx = search_region.start + np.argmin(
        widths[search_region]
    ) + head_offset
    neck_position = centers[neck_idx]
    if neck_position >= 0:
        neck_mask = points[:, 1] < neck_position
    else:
        neck_mask = points[:, 1] > neck_position

    return points[neck_mask], neck_position


def flip_body_pcd(points, neck_pos):
    if neck_pos < 0:
        points[:, 0] = -points[:, 0]
        points[:, 1] = -points[:, 1]
    
    return points


def crop_mesh_by_plane(mesh, plane):
    if plane is None:
        return mesh

    [a, b, c, d] = plane
    dist = mesh.vertices @ np.array([a, b, c]) + d
    mask = max(dist > 0.02, dist < -0.02, key = lambda x: sum(x))
    face_mask = mask[mesh.faces].all(axis=1)
    mesh_cropped = mesh.submesh([face_mask], append=True)
    mesh_cropped.remove_unreferenced_vertices()
    return mesh_cropped


def decapitate_mesh(mesh_cropped, neck_pos):
    if neck_pos >= 0:
        mask = mesh_cropped.vertices[:, 1] <= neck_pos
    else:
        mask = mesh_cropped.vertices[:, 1] >= neck_pos

    face_mask = mask[mesh_cropped.faces].all(axis=1)
    mesh_cropped = mesh_cropped.submesh([face_mask], append=True)
    mesh_cropped.remove_unreferenced_vertices()
    return mesh_cropped


def flip_mesh(mesh):
    return mesh.apply_scale([-1, -1, 1])
