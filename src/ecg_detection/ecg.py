"""Geometry + colour detection of ECG electrodes in textured torso scans."""

import csv
from dataclasses import dataclass
from itertools import permutations, product
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import trimesh
import vedo
from matplotlib.colors import rgb_to_hsv
from PIL import Image
from scipy.ndimage import (distance_transform_edt, gaussian_filter, grey_opening,
                           label as connected_components, maximum_filter)
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from src.utils.data_preparation import ensure_trimesh


LEAD_COLOURS = {
    "V1": "red",
    "V2": "yellow",
    "V3": "green",
    "V4": "darkblue",
    "V5": "orange",
    "V6": "purple",
}
LEADS = tuple(LEAD_COLOURS)


@dataclass
class ElectrodeTemplate:
    cloud: o3d.geometry.PointCloud
    marker: np.ndarray
    contact: np.ndarray


@dataclass
class ElectrodeDetection:
    label: Optional[str]
    position: np.ndarray
    marker_position: np.ndarray
    transform: np.ndarray
    score: float
    colour_position: Optional[np.ndarray] = None
    geometry_position: Optional[np.ndarray] = None
    geometry_score: Optional[float] = None


def load_coloured_mesh(path):
    """Load a mesh and repair the stale texture names used by the ECG OBJ files."""
    path = Path(path)
    mesh = ensure_trimesh(trimesh.load(path, process=False, skip_materials=True))
    if hasattr(mesh.visual, "uv"):
        images = [path.with_suffix(".jpg")] + sorted(path.parent.glob("*.jpg"))
        image = next((p for p in images if p.exists()), None)
        if image:
            mesh.visual = trimesh.visual.TextureVisuals(
                uv=np.asarray(mesh.visual.uv), image=Image.open(image).convert("RGB")
            )
    return mesh


def vertex_colours(mesh, required=True):
    """Return per-vertex RGB values in [0, 1]."""
    if getattr(mesh.visual, "kind", None) == "texture":
        image = np.asarray(mesh.visual.material.image.convert("RGB"))
        uv = np.asarray(mesh.visual.uv)
        h, w = image.shape[:2]
        x = np.clip(np.rint(uv[:, 0] * (w - 1)).astype(int), 0, w - 1)
        y = np.clip(np.rint((1 - uv[:, 1]) * (h - 1)).astype(int), 0, h - 1)
        return image[y, x] / 255.0
    if getattr(mesh.visual, "kind", None) in ("vertex", "face"):
        if mesh.visual.kind == "vertex":
            colours = mesh.visual.vertex_colors
        else:
            colours = mesh.visual.face_colors[mesh.vertex_faces[:, 0]]
        return np.asarray(colours)[:, :3] / 255.0
    if required:
        raise ValueError("The scan has no RGB/texture layer; colour labels cannot be read.")
    return None


def _cloud(points, voxel=0.0):
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    return cloud.voxel_down_sample(voxel) if voxel else cloud


def prepare_template(path, voxel=0.003, output=None):
    """Crop the template; optionally export its original triangles and RGB to PLY."""
    o3d.utility.random.seed(0)
    mesh = load_coloured_mesh(path)
    points, colours = np.asarray(mesh.vertices), vertex_colours(mesh)
    hsv = rgb_to_hsv(colours)
    plane, _ = _cloud(points).segment_plane(0.001, 3, 5000)
    height = np.abs(points @ np.asarray(plane[:3]) + plane[3])
    red = ((hsv[:, 0] < 0.035) | (hsv[:, 0] > 0.965)) & (hsv[:, 1] > 0.45)
    yellow = (hsv[:, 0] > 0.10) & (hsv[:, 0] < 0.19) & (hsv[:, 1] > 0.45)
    marker_mask = max((red, yellow), key=np.count_nonzero)
    marker = np.median(points[marker_mask], axis=0)
    mask = (height > 0.001) & (np.linalg.norm(points - marker, axis=1) < 0.08)
    electrode = points[mask]

    # The adhesive/contact is the wide end of the otherwise narrow cable.
    _, axes = np.linalg.eigh(np.cov((electrode - electrode.mean(0)).T))
    axis = axes[:, -1]
    projection = (electrode - marker) @ axis
    ends = [electrode[projection < np.percentile(projection, 25)],
            electrode[projection > np.percentile(projection, 75)]]
    spread = [np.linalg.norm(np.ptp(end - np.outer((end - marker) @ axis, axis), axis=0))
              for end in ends]
    contact = np.median(ends[int(spread[1] > spread[0])], axis=0)
    # The physical pad lies 10 mm beyond the visible holder end.
    direction = axis * np.sign((contact - marker) @ axis or 1)
    contact += 0.010 * direction
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        cropped = trimesh.Trimesh(
            vertices=points, faces=mesh.faces[mask[mesh.faces].all(axis=1)],
            vertex_colors=np.uint8(np.clip(colours, 0, 1) * 255), process=False,
        )
        cropped.remove_unreferenced_vertices()
        cropped.export(output, file_type="ply")
    return ElectrodeTemplate(_cloud(electrode, voxel), marker, contact)


def load_templates(directory, voxel=0.003):
    paths = [path for path in sorted(Path(directory).rglob("*.obj"))
             if path.with_suffix(".jpg").exists() or list(path.parent.glob("*.jpg"))]
    if not paths:
        raise FileNotFoundError("No electrode OBJ templates found in %s" % directory)
    return [prepare_template(path, voxel) for path in paths]


def _components(points, mask, confidence):
    selected = points[mask]
    if not len(selected):
        return []
    labels = np.asarray(
        _cloud(selected).cluster_dbscan(eps=0.012, min_points=1, print_progress=False)
    )
    result = []
    for label in np.unique(labels):
        group = selected[labels == label]
        extent = np.linalg.norm(np.ptp(group, axis=0))
        if extent < 0.065:
            score = float(np.mean(confidence[mask][labels == label]) + 0.03 * min(len(group), 20))
            result.append((np.median(group, axis=0), score))
    return result


def _colour_seeds(mesh):
    points = np.asarray(mesh.vertices)
    hsv = rgb_to_hsv(vertex_colours(mesh))
    h, s, v = hsv.T
    hue0 = np.minimum(h, 1 - h)
    definitions = {
        "red": ((hue0 < 0.035) & (s > 0.35) & (v > 0.18), s - 5 * hue0),
        "yellow": ((h > 0.10) & (h < 0.19) & (s > 0.35) & (v > 0.30), s - 5 * abs(h - 0.145)),
        "green": ((h > 0.29) & (h < 0.43) & (s > 0.32) & (v > 0.10),
                  s - 5 * abs(h - 0.35) - 0.5 * abs(v - 0.55)),
        "blue": ((h > 0.50) & (h < 0.72) & (s > 0.25) & (v > 0.10), s - 3 * abs(h - 0.62)),
        "brown": ((h > 0.035) & (h < 0.10) & (s > 0.40) & (v < 0.55), s + (0.55 - v)),
        "orange": ((h > 0.025) & (h < 0.09) & (s > 0.35) & (v > 0.20), s - 4 * abs(h - 0.055)),
        "black": ((v < 0.22), 0.22 - v),
        "purple": ((h > 0.70) & (h < 0.94) & (s > 0.25) & (v > 0.10), s - 3 * abs(h - 0.82)),
    }
    colours = {name: _components(points, mask, confidence)
               for name, (mask, confidence) in definitions.items()}
    candidates = {
        "V1": colours["red"],
        "V2": colours["yellow"],
        "V3": colours["green"],
        "V4": colours["blue"] + colours["brown"],
        "V5": colours["orange"] + colours["black"],
        "V6": colours["purple"],
    }
    chosen = {}

    def best(lead):
        pool = [(p, q) for p, q in candidates[lead]
                if all(np.linalg.norm(p - other) > 0.025
                       for other in chosen.values())]
        if pool:
            chosen[lead] = max(pool, key=lambda item: item[1])[0]
            return chosen[lead]
        return None

    v2 = best("V2")
    if v2 is not None and candidates["V3"]:
        v3 = max(candidates["V3"], key=lambda item:
                 item[1] - 20 * abs(np.linalg.norm(item[0] - v2) - 0.055))[0]
        if np.linalg.norm(v3 - v2) > 0.025:
            chosen["V3"] = v3
        else:
            v3 = best("V3")
    else:
        v3 = best("V3")
    step = None if v2 is None or v3 is None else v3 - v2
    direction = None if step is None else step / np.linalg.norm(step)

    def choose(lead, origin, expected, forward=False, radius=0.13, weight=18):
        if origin is None or expected is None:
            return best(lead)
        pool = [(p, q) for p, q in candidates[lead]
                if 0.008 < np.linalg.norm(p - origin) < radius
                and all(np.linalg.norm(p - other) > 0.025
                        for other in chosen.values())
                and (not forward or direction is None or
                     np.dot(p - origin, direction) > 0.01)]
        if not pool:
            return best(lead)
        chosen[lead] = max(
            pool, key=lambda item: item[1] - weight * np.linalg.norm(item[0] - expected)
        )[0]
        return chosen[lead]

    v1 = choose("V1", v2, v2, radius=0.20, weight=3)
    v4 = choose("V4", v3, None if step is None else v3 + step,
                forward=True, radius=0.12)
    v5 = choose("V5", v4, None if step is None or v4 is None else v4 + step,
                forward=True, radius=0.15)
    choose("V6", v3, None if step is None else v3 + step,
           radius=0.18, weight=1)
    return {label: chosen[label] for label in LEADS if label in chosen}


def _basis(cloud):
    points = np.asarray(cloud.points)
    axes = np.linalg.eigh(np.cov(points.T))[1][:, ::-1]
    axes[:, -1] *= np.sign(np.linalg.det(axes))
    return axes


def _fit_template(template, target_cloud, source_anchor, target_anchor,
                  voxel, penalty=lambda transform: 0.0):
    source_axes, target_axes = _basis(template.cloud), _basis(target_cloud)
    best = None
    for order in permutations(range(3)):
        for signs in product((-1, 1), repeat=3):
            orientation = np.eye(3)[:, order] @ np.diag(signs)
            if np.linalg.det(orientation) < 0:
                continue
            transform = np.eye(4)
            transform[:3, :3] = target_axes @ orientation @ source_axes.T
            transform[:3, 3] = target_anchor - transform[:3, :3] @ source_anchor
            result = o3d.pipelines.registration.registration_icp(
                template.cloud, target_cloud, 2 * voxel, transform,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40),
            )
            transform = result.transformation.copy()
            anchor = trimesh.transform_points([source_anchor], transform)[0]
            transform[:3, 3] += target_anchor - anchor
            fit = o3d.pipelines.registration.evaluate_registration(
                template.cloud, target_cloud, 2.5 * voxel, transform
            )
            score = fit.fitness - fit.inlier_rmse - penalty(transform)
            match = (score, transform)
            best = match if best is None or match[0] > best[0] else best
    return best[1], best[0]


def _register(template, target, target_hsv, lead, seeds, voxel, occupied=()):
    seed = seeds[lead]
    distance = np.linalg.norm(target - seed, axis=1)
    # White/grey plastic and cable are much less saturated than skin.
    local = target[(distance < 0.105) & (
        ((target_hsv[:, 1] < 0.25) & (target_hsv[:, 2] > 0.25)) |
        (distance < 0.022)
    )]

    def penalty(transform):
        contact = trimesh.transform_points([template.contact], transform)[0]
        clearance = min((np.linalg.norm(contact - point)
                         for other, point in seeds.items() if other != lead),
                        default=np.inf)
        value = 20 * max(0, 0.025 - clearance)
        if occupied:
            clearance = min(np.linalg.norm(contact - point) for point in occupied)
            value += 30 * max(0, 0.020 - clearance)
        if lead == "V1" and "V2" in seeds and "V3" in seeds:
            direction = seeds["V3"] - seeds["V2"]
            direction /= np.linalg.norm(direction)
            value += 20 * max(0, np.dot(contact - seed, direction))
        return value

    return _fit_template(
        template, _cloud(local, voxel), template.marker, seed, voxel, penalty
    )


def _geometric_seeds(mesh, voxel, return_foreground=False):
    """Find raised candidates without forcing a six-electrode layout."""
    points = np.asarray(mesh.vertices)
    low, high = np.percentile(points[:, :2], (1, 99), axis=0)
    extent = high - low
    # Include the medial chest: V1 may be separated from the lateral electrodes.
    low += extent * (0.30, 0.30)
    high -= extent * (0.18, 0.32)
    shape = np.ceil((high - low) / voxel).astype(int) + 1
    cells = np.floor((points[:, :2] - low) / voxel).astype(int)
    inside = np.all((cells >= 0) & (cells < shape), axis=1)
    height = np.full(shape, -np.inf)
    np.maximum.at(height, (cells[inside, 0], cells[inside, 1]), points[inside, 2])
    valid = np.isfinite(height)
    if not valid.any():
        return ([], np.empty((0, 3))) if return_foreground else []
    nearest = distance_transform_edt(~valid, return_distances=False,
                                     return_indices=True)
    surface = gaussian_filter(height[tuple(nearest)], 1)
    radius = max(2, int(round(0.021 / voxel)))
    grid = np.arange(-radius, radius + 1)
    disk = grid[:, None] ** 2 + grid[None, :] ** 2 <= radius ** 2
    residual = surface - grey_opening(surface, footprint=disk)
    support = distance_transform_edt(np.pad(valid, 1))[1:-1, 1:-1] * voxel
    regions, count = connected_components((residual > 0.0025) & (support > 0.009))
    allowed = np.zeros(count + 1, dtype=bool)
    for region in range(1, count + 1):
        indices = np.argwhere(regions == region)
        extent = np.linalg.norm(np.ptp(indices, axis=0) * voxel)
        allowed[region] = len(indices) >= 8 and 0.018 < extent < 0.15
    residual[~allowed[regions]] = -np.inf
    maxima = maximum_filter(residual, max(3, int(0.020 / voxel)))
    indices = np.argwhere((residual == maxima) & (residual > 0.003) &
                          (residual < 0.020))
    order = np.argsort(residual[tuple(indices.T)])[::-1]
    candidates = []
    for index in indices[order]:
        point = low + (index + 0.5) * voxel
        if all(np.linalg.norm(point - other) > 0.025 for other in candidates):
            candidates.append(point)
    tree = cKDTree(points[:, :2])
    positions = []
    for point in candidates[:40]:
        nearby = tree.query_ball_point(point, 2 * voxel)
        positions.append(points[nearby[np.argmax(points[nearby, 2])]])
    indices = np.flatnonzero(inside)
    xy = tuple(cells[inside].T)
    raised = ((residual[xy] > 0.0015) &
              (np.abs(points[inside, 2] - height[xy]) < 0.008))
    foreground = points[indices[raised]]
    return (positions, foreground) if return_foreground else positions


def _register_geometry(template, target, position, voxel):
    local = target[np.linalg.norm(target - position, axis=1) < 0.090]
    if len(local) < 12:
        raise ValueError("Insufficient raised geometry around candidate")
    cloud = _cloud(local, voxel)
    points = np.asarray(template.cloud.points)
    holder = points[np.linalg.norm(points - template.contact, axis=1) < 0.045]
    if len(holder) < 12:
        raise ValueError("Insufficient template holder geometry")
    holder_template = ElectrodeTemplate(_cloud(holder), template.marker, template.contact)
    transform, _ = _fit_template(
        holder_template, cloud, template.contact, position, 0.6 * voxel
    )
    fit = o3d.pipelines.registration.registration_icp(
        holder_template.cloud, cloud, 1.5 * voxel, transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    return fit.transformation, fit.fitness - fit.inlier_rmse


def _deduplicate(detections, min_distance):
    """Keep the best-scoring detection within each spatial neighbourhood."""
    positions = [d.position if d.label is None else d.marker_position
                 for d in detections]
    keep = []
    for index in sorted(range(len(detections)),
                        key=lambda i: detections[i].score, reverse=True):
        if all(np.linalg.norm(positions[index] - positions[i])
               >= min_distance for i in keep):
            keep.append(index)
    return [detection for i, detection in enumerate(detections) if i in keep]


def _best_match(templates, registration):
    matches = []
    for template in templates:
        try:
            matches.append((registration(template), template))
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            pass
    return max(matches, key=lambda item: item[0][1]) if matches else None


def _detect_coloured(mesh, templates, voxel):
    points = np.asarray(mesh.vertices)
    hsv = rgb_to_hsv(vertex_colours(mesh))
    try:
        seeds = _colour_seeds(mesh)
    except (RuntimeError, ValueError):
        return _detect_geometry(mesh, templates, voxel)
    geometry = _detect_geometry(mesh, templates, voxel)
    geometry_by_lead = {}
    labels = list(seeds)
    if geometry and labels:
        rows, columns = linear_sum_assignment(np.linalg.norm(
            np.array([seeds[label] for label in labels])[:, None] -
            np.array([item.position for item in geometry])[None], axis=2
        ))
        geometry_by_lead = {labels[row]: geometry[column]
                            for row, column in zip(rows, columns)}
    detections = []

    # Colour proposes and labels a region; template ICP verifies its geometry.
    for label in labels:
        marker = seeds[label]
        occupied = [detection.position for detection in detections]
        match = _best_match(
            templates,
            lambda candidate: _register(
                candidate, points, hsv, label, seeds, voxel, occupied
            ),
        )
        if match is None:
            continue
        ((transform, score), template) = match
        position = trimesh.transform_points([template.contact], transform)[0]
        geometric = geometry_by_lead.get(label)
        detections.append(
            ElectrodeDetection(
                label, position, marker, transform, score,
                colour_position=marker,
                geometry_position=None if geometric is None else geometric.position,
                geometry_score=None if geometric is None else geometric.score,
            )
        )
    return detections


def _detect_geometry(mesh, templates, voxel):
    detections = []
    try:
        positions, foreground = _geometric_seeds(mesh, voxel, return_foreground=True)
    except (RuntimeError, ValueError):
        return detections
    for position in positions:
        match = _best_match(
            templates,
            lambda candidate: _register_geometry(
                candidate, foreground, position, voxel
            ),
        )
        if match is None:
            continue
        ((transform, score), template) = match
        if score < 0.70:
            continue
        marker, contact = trimesh.transform_points(
            [template.marker, template.contact], transform
        )
        detections.append(
            ElectrodeDetection(
                None, contact, marker, transform, score,
                geometry_position=contact, geometry_score=score,
            )
        )
    detections = sorted(detections, key=lambda item: item.score, reverse=True)
    detections = _deduplicate(detections, 0.045)
    if len(detections) > 1:
        # Electrodes form a connected chest group; reject isolated bed/arm fits.
        xy = np.array([item.position[:2] for item in detections])
        groups = np.asarray(_cloud(np.c_[xy, np.zeros(len(xy))]).cluster_dbscan(
            eps=0.15, min_points=1, print_progress=False
        ))
        group = np.argmax(np.bincount(groups))
        detections = [item for item, index in zip(detections, groups) if index == group]
    return detections[:6]


def detect_electrodes(scan, templates, voxel=0.003, seed=0,
                      min_distance=0.025):
    """Detect six contacts; labels are ``None`` when RGB is unavailable."""
    o3d.utility.random.seed(seed)
    mesh = load_coloured_mesh(scan) if not isinstance(scan, trimesh.Trimesh) else scan
    templates = load_templates(templates, voxel) if isinstance(templates, (str, Path)) else templates
    coloured = vertex_colours(mesh, required=False) is not None
    detections = (_detect_coloured(mesh, templates, voxel) if coloured else
                  _detect_geometry(mesh, templates, voxel))
    return mesh, _deduplicate(detections, min_distance)


def visualise(mesh, detections, max_points=40000, ax=None):
    """Plot the textured scan and detected contact points; return (figure, axes)."""
    figure = plt.figure(figsize=(10, 8)) if ax is None else ax.figure
    ax = figure.add_subplot(111, projection="3d") if ax is None else ax
    points, colours = np.asarray(mesh.vertices), vertex_colours(mesh, required=False)
    take = np.linspace(0, len(points) - 1, min(max_points, len(points))).astype(int)
    ax.scatter(*points[take].T, c=colours[take] if colours is not None else "0.75",
               s=0.5, alpha=0.9)
    for detection in detections:
        colour = LEAD_COLOURS.get(detection.label, "dodgerblue")
        ax.scatter(*detection.position, color=colour, s=50)
        if detection.label:
            offset = {"V1": (-0.025, 0.015, 0), "V2": (0.01, 0.02, 0)}.get(
                detection.label, (0.008, 0.008, 0)
            )
            ax.text(*(detection.position + offset), detection.label,
                    fontsize=11, weight="bold")
    ax.set(xlabel="X", ylabel="Y", zlabel="Z")
    ax.set_box_aspect(np.ptp(points, axis=0))
    ax.view_init(elev=90, azim=-90)
    return figure, ax


def visualise_template(template, output=None, interactive=True,
                       size=(900, 700)):
    """Show the cropped template, its visible marker and shifted contact."""
    template = prepare_template(template) if isinstance(template, (str, Path)) else template
    actors = [
        vedo.Points(np.asarray(template.cloud.points), r=4, c="lightgray"),
        vedo.Points([template.marker], r=14, c="orange"),
        vedo.Points([template.contact], r=14, c="lime"),
        vedo.Text2D("pomarańczowy: znacznik\nzielony: styk +10 mm",
                    pos="bottom-left", c="black"),
    ]
    plotter = vedo.Plotter(size=size, bg="white", offscreen=not interactive)
    plotter.show(*actors, axes=1, interactive=False)
    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        plotter.screenshot(str(output))
    if interactive:
        plotter.interactive()
    else:
        plotter.close()
    return plotter


def visualise_vedo(mesh, detections, output_dir=None, interactive=True,
                   size=(1200, 900), show_intermediate=True):
    """Render a front view in Vedo and optionally save PNG and CSV outputs."""
    points, colours = np.asarray(mesh.vertices), vertex_colours(mesh, required=False)
    torso = vedo.Mesh([points, np.asarray(mesh.faces)])
    if colours is None:
        torso.color("lightgray")
    else:
        torso.pointcolors = np.c_[np.uint8(np.clip(colours, 0, 1) * 255),
                                  np.full(len(points), 255, dtype=np.uint8)]
    torso.alpha(0.9)

    scale = np.ptp(points, axis=0).max()
    centre = points.mean(axis=0)
    plotter = vedo.Plotter(size=size, bg="white", offscreen=not interactive)
    plotter.camera.SetPosition(centre + (0, 0, 2.2 * scale))
    plotter.camera.SetFocalPoint(centre)
    plotter.camera.SetViewUp(0, 1, 0)
    actors = [torso]
    for detection in detections:
        colour = LEAD_COLOURS.get(detection.label, "dodgerblue")
        if show_intermediate and detection.colour_position is not None:
            actors.append(vedo.Points([detection.colour_position], r=8,
                                      c=colour, alpha=0.45))
        if show_intermediate and detection.colour_position is not None \
                and detection.geometry_position is not None:
            actors.append(vedo.Points([detection.geometry_position], r=8,
                                      c="cyan", alpha=0.65))
        actors.append(vedo.Points([detection.position], r=18, c=colour))
        if detection.label:
            label = vedo.Text3D(detection.label,
                                pos=detection.position + scale * np.array((0.017, 0.020, 0.080)),
                                s=0.018 * scale,
                                c="white" if detection.label in ("V4", "V5") else "black",
                                depth=0.001 * scale)
            actors.append(label)

    if detections:
        score_text = "Dopasowanie\n" + "\n".join(
            f"{d.label or 'P' + str(i)}: {d.score:.3f}"
            for i, d in enumerate(detections, 1)
        )
        actors.append(vedo.Text2D(score_text, pos="top-right", s=0.9,
                                  bg="white", c="black", alpha=0.8,
                                  justify="top-right"))
        if show_intermediate and any(d.colour_position is not None for d in detections):
            actors.append(vedo.Text2D(
                "maly kolorowy: tylko RGB\ncyan: tylko geometria\nduzy: wynik",
                pos="bottom-left", s=0.7, c="black", bg="white", alpha=0.8,
            ))
    plotter.show(*actors, resetcam=False, interactive=False)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plotter.screenshot(str(output_dir / "front_view.png"))
        with (output_dir / "scores.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(("stage", "label", "x", "y", "z", "score"))
            for detection in detections:
                if detection.colour_position is not None:
                    writer.writerow(("colour", detection.label or "",
                                     *detection.colour_position, ""))
                if detection.geometry_position is not None:
                    writer.writerow(("geometry", detection.label or "",
                                     *detection.geometry_position,
                                     "" if detection.geometry_score is None else
                                     detection.geometry_score))
                writer.writerow(("combined", detection.label or "",
                                 *detection.position, detection.score))
    if interactive:
        plotter.interactive()
    else:
        plotter.close()
    return plotter
