import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import vedo
from vedo import Assembly, Box, Mesh, Text2D
from vedo.addons import Axes
import trimesh

from src.utils.data_preparation import pad_bounding_box_max_z


AXIS_LABELS = (
    "X (left-right)",
    "Y (legs-head)",
    "Z (back-front)",
)


def plot_bounding_box(ax, min_pt, max_pt, color='g'):
    max_pt = pad_bounding_box_max_z(max_pt)
    corners = np.array([
        [min_pt[0], min_pt[1], min_pt[2]],
        [min_pt[0], min_pt[1], max_pt[2]],
        [min_pt[0], max_pt[1], min_pt[2]],
        [min_pt[0], max_pt[1], max_pt[2]],
        [max_pt[0], min_pt[1], min_pt[2]],
        [max_pt[0], min_pt[1], max_pt[2]],
        [max_pt[0], max_pt[1], min_pt[2]],
        [max_pt[0], max_pt[1], max_pt[2]],
    ])
    edges = [
        [0,1],[0,2],[0,4],[1,3],[1,5],[2,3],[2,6],[3,7],
        [4,5],[4,6],[5,7],[6,7]
    ]
    lines = [(corners[e[0]], corners[e[1]]) for e in edges]
    lc = Line3DCollection(lines, colors=color, linewidths=2)
    ax.add_collection3d(lc)


def set_3d_plot_bounds(ax, values, minimum, maximum):
    padded_maximum = pad_bounding_box_max_z(maximum)
    lower = np.minimum(values.min(axis=0), minimum)
    upper = np.maximum(values.max(axis=0), padded_maximum)
    extent = np.maximum(upper - lower, 1e-6)
    ax.set_xlim(lower[0], upper[0])
    ax.set_ylim(lower[1], upper[1])
    ax.set_zlim(lower[2], upper[2])
    ax.set_box_aspect(extent)


def pair_3d_rotation(figure, axes):
    previous_callback = getattr(figure, "_paired_rotation_callback", None)
    if previous_callback is not None:
        figure.canvas.mpl_disconnect(previous_callback)

    axes = tuple(axes)
    supports_roll = all(hasattr(ax, "roll") for ax in axes)

    def view(ax):
        angles = (ax.elev, ax.azim)
        return angles + (ax.roll,) if supports_roll else angles

    views = {ax: view(ax) for ax in axes}

    def synchronize(event):
        source = event.inaxes
        if source not in views:
            return

        source_view = view(source)
        if source_view == views[source]:
            return

        for target in axes:
            if target is source:
                continue
            if supports_roll:
                target.view_init(
                    elev=source_view[0],
                    azim=source_view[1],
                    roll=source_view[2],
                )
            else:
                target.view_init(elev=source_view[0], azim=source_view[1])
            views[target] = source_view
        views[source] = source_view
        figure.canvas.draw_idle()

    figure._paired_rotation_callback = figure.canvas.mpl_connect(
        "motion_notify_event", synchronize
    )


def plot_crop_comparison(figure, original, cropped, minimum, maximum):
    figure.clear()
    axes = figure.subplots(1, 2, subplot_kw={"projection": "3d"})
    pair_3d_rotation(figure, axes)
    for ax, values, title, color in (
        (axes[0], original, "Preprocessed", "blue"),
        (axes[1], cropped, "Manual crop", "red"),
    ):
        ax.scatter(
            values[:, 0], values[:, 1], values[:, 2], s=1, c=color, alpha=0.45
        )
        plot_bounding_box(ax, minimum, maximum, color="green")
        ax.set_title(title)
        ax.set_xlabel(AXIS_LABELS[0])
        ax.set_ylabel(AXIS_LABELS[1])
        ax.set_zlabel(AXIS_LABELS[2])
        set_3d_plot_bounds(ax, values, minimum, maximum)


def show_meshes_vedo(raw_mesh: trimesh.Trimesh,
                     prediction_mesh: trimesh.Trimesh,
                     full_prediction_mesh=None,
                     full_body_mesh=None,
                     cropped_points=None,
                     crop_bounds=None,
                     losses=None,
                     joints=None,
                     plotter=None,
                     score=None,
                     preprocessed_mesh=None):
    scan_mesh = preprocessed_mesh if preprocessed_mesh is not None else raw_mesh
    scan_mesh = scan_mesh.copy()
    scan_mesh.apply_translation([-0.75, 0, 0])
    scan = Mesh([scan_mesh.vertices, scan_mesh.faces])
    pred = Mesh([prediction_mesh.vertices, prediction_mesh.faces])

    scan.c("dodgerblue")

    prediction_label = (
        "cropped prediction" if full_prediction_mesh is not None else "prediction"
    )
    scan_label = "preprocessed scan" if preprocessed_mesh is not None else "scan"
    label_text = f"{scan_label} = blue\n{prediction_label} = Turbo colormap"
    meshes = [scan, pred]

    if full_prediction_mesh is not None:
        full_prediction_mesh = full_prediction_mesh.copy()
        full_prediction_mesh.apply_translation([0.0, 0, -0.001])
        full_pred = Mesh(
            [full_prediction_mesh.vertices, full_prediction_mesh.faces]
        )
        full_pred.c("lightgray", alpha=0.2)
        label_text += "\nfull prediction = gray"
        meshes.append(full_pred)

    if full_body_mesh is not None:
        full_body_mesh = full_body_mesh.copy()
        full_body_mesh.apply_translation([0.75, 0, 0])
        full = Mesh([full_body_mesh.vertices, full_body_mesh.faces])
        full.c("green")
        label_text += "\nfull-body ground truth = green"
        meshes.append(full)

    if cropped_points is not None or crop_bounds is not None:
        minimum, maximum = crop_bounds if crop_bounds is not None else (
            cropped_points.squeeze(0).min(dim=0).values,
            cropped_points.squeeze(0).max(dim=0).values,
        )
        maximum = pad_bounding_box_max_z(maximum)
        xmin, ymin, zmin = minimum
        xmax, ymax, zmax = maximum
        bbox = Box(
            pos=((xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2),
            length=xmax-xmin,
            width=ymax-ymin,
            height=zmax-zmin,
        ).wireframe()
        meshes.append(bbox)

    if losses is not None:
        pred.cmap("turbo", losses * 1000, vmin=0, vmax=50)
        pred.add_scalarbar(label_format=":.3f mm")

    if score is not None:
        label_text += f"\nAverage distance (inside bounding box): {score*1000:.3f} mm"

    if joints is not None:
        joints_pts = vedo.Points(joints, r=10, c="red")
        meshes.append(joints_pts)

    label = Text2D(
        label_text,
        pos="top-left",
        font="Courier",
        s=1.0,
    )
    axes = Axes(
        Assembly(meshes),
        xtitle="X",
        ytitle="Y",
        ztitle="Z",
        c="black",
    )

    if plotter is None:
        plotter = vedo.Plotter(title="Fit result 3D")
    plotter.clear(deep=True)
    plotter.show(
        *meshes,
        axes,
        label,
        axes=0,
        bg="white",
        title="Scan vs Prediction",
        resetcam=True,
    )
    return plotter
