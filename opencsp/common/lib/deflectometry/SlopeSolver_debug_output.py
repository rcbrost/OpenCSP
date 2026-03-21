"""Functions supporting diagnosis of slope solution for a deflectometry setup."""

import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation

from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.deflectometry.SlopeSolverDataDebug import SlopeSolverDataDebug
from opencsp.app.sofast.lib.DebugOpticsGeometry import DebugOpticsGeometry
from opencsp.app.sofast.lib.DefinitionFacet import DefinitionFacet
import opencsp.app.sofast.lib.sofast_debug_figure_support as sdfs
from opencsp.app.sofast.lib.SpatialOrientation import SpatialOrientation
from opencsp.common.lib.deflectometry.Surface2DAbstract import Surface2DAbstract
import opencsp.common.lib.geometry.TransformXYZ as txyz
from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz
import opencsp.common.lib.render.view_spec as vs
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.tool.log_tools as lt


# INTERSECTION SURFACE, DEFINING VERTICES, CAMERA


def figure_intersection_surface_situation(
    figure_title: str,
    v_facet_corners_hires_1: Vxyz | None,
    v_facet_corners_hires_2: Vxyz | None,
    surface: Surface2DAbstract,
    idx1: int,
    idx2: int,
    debug: SlopeSolverDataDebug,
):
    """Sets up and draws a figure showing SOFAST layout, including only screen and camera."""
    view_spec_az_el_roll_list = [
        (vs.view_spec_3d(), None),
        (vs.view_spec_3d(), (0, 90, 90)),  # xy
        (vs.view_spec_3d(), (-90, 0, 0)),  # xz
        (vs.view_spec_3d(), (0, 0, 0)),  # yz
        (vs.view_spec_xy(), None),  # Doesn't show intersection surface.
        (vs.view_spec_xz(), None),  # Doesn't show intersection surface.
        (vs.view_spec_yz(), None),  # Doesn't show intersection surface.
    ]
    for view_spec_az_el_roll in view_spec_az_el_roll_list:
        az_el_roll_deg = view_spec_az_el_roll[1]
        if az_el_roll_deg is None:
            this_title = figure_title
        else:
            this_title = figure_title + ' (Az,El,Roll)=' + str(az_el_roll_deg)
        figure_intersection_surface_situation_aux(
            this_title,
            v_facet_corners_hires_1,
            v_facet_corners_hires_2,
            surface,
            idx1,
            idx2,
            debug,
            view_spec_az_el_roll_deg=view_spec_az_el_roll,
        )


def figure_intersection_surface_situation_aux(
    title: str,
    v_facet_corners_hires_1: Vxyz | None,
    v_facet_corners_hires_2: Vxyz | None,
    surface: Surface2DAbstract,
    idx1: int,
    idx2: int,
    debug: SlopeSolverDataDebug,
    view_spec_az_el_roll_deg: list[dict, tuple[float, float, float]] = None,
) -> None:
    """Supports routine without aux extension."""
    # Fetch axis data.
    view_spec = view_spec_az_el_roll_deg[0]
    az_el_roll_deg = view_spec_az_el_roll_deg[1]

    # Create a new figure.
    full_title = f"Slope Solver ({idx1:d}, {idx2:d}): " + title
    fig_rec = sdfs.start_debug_3d_figure(figure_title=full_title, view_spec=view_spec, figsize=(12, 9))

    # # Plot original facet corners.
    # Use draw_line(), because we only want one legend entry, not a legend entry for every point.
    debug.optic_data.v_facet_corners.draw_line(
        fig_rec, style=rcps.marker(marker='o', color='lightgreen', markersize=5), label="Facet Vertices"
    )

    # Plot high-resolution facet corners, before fit_slopes() execution.
    # Use draw_line(), because we only want one legend entry, not a legend entry for every point.
    if v_facet_corners_hires_1 is not None:
        if v_facet_corners_hires_2 is None:
            label_str = "Facet Vertices (High Resolution)"
        else:
            label_str = "Hires Vertices (Before Fit)"
        v_facet_corners_hires_1.draw_line(
            fig_rec, style=rcps.marker(marker='.', color='b', markersize=4), label=label_str
        )

    # Plot high-resolution facet corners, after fit_slopes() execution.
    # Use draw_line(), because we only want one legend entry, not a legend entry for every point.
    if v_facet_corners_hires_2 is not None:
        label_str = "Hires Vertices (After Fit)"
        v_facet_corners_hires_2.draw_line(
            fig_rec, style=rcps.marker(marker='.', color='red', markersize=2), label=label_str
        )

    # Set view direction, if desired.
    if az_el_roll_deg is not None:
        if not fig_rec.view.is_3d():
            lt.error_and_raise(
                ValueError, "In SlopeSolver._plot_debug_plots_2(), asked to set view direction for a non-3d plot."
            )
        azimuth_deg = az_el_roll_deg[0]
        elevation_deg = az_el_roll_deg[1]
        roll_deg = az_el_roll_deg[2]
        lt.info(
            'In SlopeSolver._plot_debug_plots_2(), setting view (azimuth, elevation, roll) to '
            + str((azimuth_deg, elevation_deg, roll_deg))
            + ' degrees.'
        )
        fig_rec.view.axis.view_init(azim=azimuth_deg, elev=elevation_deg, roll=roll_deg)

    # Plot intersection points.
    # The trisurf plot is only supported for 3-d axes.
    if fig_rec.view.is_3d():
        surface.plot_intersection_points(
            fig_rec.view.axis,
            debug.slope_solver_point_downsample,
            debug.slope_solver_camera_rays_length,
            debug.slope_solver_plot_camera_screen_points,
        )

    # Add legend.
    plt.legend()

    # Save and close.
    full_title_for_file = (
        full_title.replace(' ', '_').replace(':', '').replace('(', '').replace(')', '').replace(',', '')
    )
    sdfs.finish_debug_3d_figure(full_title_for_file, 'solver', fig_rec, debug.debug_geometry)


# INITIAL REPROJECTION SUMMARY, AFTER SNAP TO EDGE


def reproj_snap(
    pts_reproj: Vxy, hires_pts_reproj_1: Vxy, hires_pts_reproj_snap_1: Vxy, debug: SlopeSolverDataDebug
) -> None:
    reproj_snap_aux(pts_reproj, hires_pts_reproj_1, hires_pts_reproj_snap_1, label_points=True, debug=debug)
    reproj_snap_aux(None, hires_pts_reproj_1, None, label_points=False, debug=debug)
    reproj_snap_aux(None, None, hires_pts_reproj_snap_1, label_points=False, debug=debug)


def reproj_snap_aux(
    pts_reproj: Vxy | None,
    hires_pts_reproj_1: Vxy | None,
    hires_pts_reproj_snap_1: Vxy | None,
    label_points: bool,
    debug: SlopeSolverDataDebug,
) -> None:
    figure_title = "Reprojected Points, and Snap to Edges"
    fig_rec = sdfs.start_debug_image_figure(figure_title)
    fig_rec.view.imshow(debug.debug_geometry.mask_processed, cmap="gray")
    if pts_reproj is not None:
        sdfs.plot_labeled_points(
            pts_reproj,
            marker_size=30,
            point_color='lightgreen',
            label_points=label_points,
            label_color='lightgreen',
            label_size="small",
            legend_label='Reprojected After solvePnP(), then Refined Distance',
        )
    if hires_pts_reproj_1 is not None:
        sdfs.plot_labeled_points(
            hires_pts_reproj_1,
            marker_size=12,
            point_color='blue',
            label_points=label_points,
            label_color='blue',
            label_size="xx-small",
            legend_label='High-Resolution Points, Before Modification',
        )
    if hires_pts_reproj_snap_1 is not None:
        sdfs.plot_labeled_points(
            hires_pts_reproj_snap_1,
            marker_size=8,
            point_color='magenta',
            label_points=label_points,
            label_color='magenta',
            label_size="xx-small",
            legend_label='High-Resolution Points, Snapped to Image Edge',
        )
    # Legend
    fig_rec.view.axis.legend()
    sdfs.finish_debug_image_figure(figure_title, 'solver', fig_rec, debug.debug_geometry)


# REPROJECTION OF SNAP TO EDGE, SNAP TO SURFACE REPROJECTION


def reproj_after_snap_snap(hires_pts_reproj_snap_1: Vxy, hires_pts_reproj_3: Vxy, debug: SlopeSolverDataDebug) -> None:
    reproj_after_snap_snap_aux(hires_pts_reproj_snap_1, hires_pts_reproj_3, label_points=True, debug=debug)
    reproj_after_snap_snap_aux(hires_pts_reproj_snap_1, hires_pts_reproj_3, label_points=False, debug=debug)


def reproj_after_snap_snap_aux(
    hires_pts_reproj_snap_1: Vxy, hires_pts_reproj_3: Vxy, label_points: bool, debug: SlopeSolverDataDebug
) -> None:
    figure_title = "Snap-to-Edge Points, Snapped to Surface and Reprojected"
    fig_rec = sdfs.start_debug_image_figure(figure_title)
    fig_rec.view.imshow(debug.debug_geometry.mask_processed, cmap="gray")
    sdfs.plot_labeled_points(
        hires_pts_reproj_snap_1,
        marker_size=20,
        point_color='magenta',
        label_points=label_points,
        label_color='magenta',
        label_size="xx-small",
        legend_label='High-Resolution Points, Snapped to Image Edge',
    )
    sdfs.plot_labeled_points(
        hires_pts_reproj_3,
        marker_size=8,
        point_color='green',
        label_points=label_points,
        label_color='green',
        label_size="xx-small",
        legend_label='Snapped-to-Edge Points, Reprojected After SolvePnP',
    )
    # Legend
    fig_rec.view.axis.legend()
    sdfs.finish_debug_image_figure(figure_title, 'solver', fig_rec, debug.debug_geometry)


# REPROJECTION OF SNAP TO EDGE, SNAP TO SURFACE REPROJECTION


def reproj_after_fit(
    hires_pts_reproj_snap_1: Vxy, hires_pts_reproj_3: Vxy, hires_pts_reproj_4: Vxy, debug: SlopeSolverDataDebug
) -> None:
    reproj_after_fit_aux(hires_pts_reproj_snap_1, hires_pts_reproj_3, hires_pts_reproj_4, debug=debug)


def reproj_after_fit_aux(
    hires_pts_reproj_snap_1: Vxy, hires_pts_reproj_3: Vxy, hires_pts_reproj_4: Vxy, debug: SlopeSolverDataDebug
) -> None:
    figure_title = "After Fitting Surface to Slopes, Snapping to Surface, Reprojecting"
    fig_rec = sdfs.start_debug_image_figure(figure_title)
    fig_rec.view.imshow(debug.debug_geometry.mask_processed, cmap="gray")
    sdfs.plot_labeled_points(
        hires_pts_reproj_snap_1,
        marker_size=20,
        point_color='magenta',
        label_points=False,
        label_color='magenta',
        label_size="xx-small",
        legend_label='High-Resolution 3-d Points, Reprojected and Snapped to Image Edge',
    )
    sdfs.plot_labeled_points(
        hires_pts_reproj_3,
        marker_size=8,
        point_color='green',
        label_points=False,
        label_color='green',
        label_size="xx-small",
        legend_label='Snap-to-Edge Points => SolvePnP New Camera Pose => 3-d Pts Reprojected',
    )
    sdfs.plot_labeled_points(
        hires_pts_reproj_4,
        marker_size=12,
        point_color='red',
        label_points=True,
        label_color='red',
        label_size="xx-small",
        legend_label='High-Res 3-d Points, Snapped to New Fit Surface, Reprojected w/New Pose',
    )
    # Legend
    fig_rec.view.axis.legend(fontsize='small')
    sdfs.finish_debug_image_figure(figure_title, 'solver', fig_rec, debug.debug_geometry)
