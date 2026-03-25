"""Functions to simplify code that outputs SOFAST debugging figures."""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.spatial.transform import Rotation

from opencsp.app.sofast.lib.SpatialOrientation import SpatialOrientation
from opencsp.common.lib.camera.Camera import Camera
from opencsp.app.sofast.lib.DebugOpticsGeometry import DebugOpticsGeometry
from opencsp.app.sofast.lib.DefinitionFacet import DefinitionFacet
import opencsp.common.lib.geometry.TransformXYZ as txyz
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render.view_spec as vs
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlFigure as rcfg
import opencsp.common.lib.render_control.RenderControlSofastSetup as rcss
import opencsp.common.lib.tool.log_tools as lt


def plot_labeled_points(
    pts: Vxy,
    marker_size: int = 30,
    point_color: str = 'r',
    label_points: bool = True,
    label_color: str = 'g',
    label_size: str = "medium",
    legend_label: str = '',
) -> None:
    """Plots labeled points on axis for debugging"""
    plt.scatter(*pts.data, s=marker_size, c=point_color, label=legend_label)
    if label_points:
        for idx, pt in enumerate(pts):
            plt.text(*pt.data, idx, color=label_color, size=label_size)


# GENERAL FIGURES


def start_debug_image_figure(figure_title: str) -> rcfg.RenderControlFigure:
    """Begins a debug figure setup to show an image,
    possibly with other annotations."""
    fig_rec = fm.setup_figure(
        figure_control=rcfg.RenderControlFigure(tile=False),
        axis_control=rca.image(grid=False, axis_prefix="Image "),
        view_spec=vs.view_spec_im(),
        title=figure_title,
    )
    return fig_rec


def finish_debug_image_figure(
    figure_title: str, phase: str, fig_rec: rcfg.RenderControlFigure, debug: DebugOpticsGeometry
) -> None:
    """Closes and saves debug image figure."""
    figure_title_clean = figure_title.replace(' ', '_').replace(',', '')
    figure_file_body = f"{debug.figure_idx:02d}_{phase}_{figure_title_clean}"
    debug.figure_idx += 1
    fig_rec.save(
        output_dir=debug.save_dir,
        output_file_body=figure_file_body,
        dpi=200,
        format='png',
        close_after_save=True,
        include_view_suffix=False,
        include_limit_suffix=False,
    )


# 3-D FIGURES


def start_debug_3d_figure(
    figure_title: str,
    view_spec: dict = vs.view_spec_3d(),
    equal: bool = True,
    grid: bool = True,
    axis_prefix: str = None,
    figsize: tuple[float, float] = (6.4, 4.8),  # Inch
) -> rcfg.RenderControlFigure:
    """Begins a debug figure setup to show data in a 3-d space."""
    fig_rec = fm.setup_figure_for_3d_data(
        figure_control=rcfg.RenderControlFigure(tile=False, figsize=figsize),
        axis_control=rca.meters(grid=grid, axis_prefix=axis_prefix),
        view_spec=view_spec,
        equal=equal,
        title=figure_title,
    )
    return fig_rec


def finish_debug_3d_figure(
    figure_title: str,
    phase: str,
    fig_rec: rcfg.RenderControlFigure,
    debug: DebugOpticsGeometry,
    axis_prefix: str = None,
) -> None:
    """Closes and saves debug 3-d figure."""
    show = False  # True  # &&&& DELETE-SCAFFOLDING -- HANDLE SOURCE, PASS FROM CALLERS

    # Show now, so that all drawn element labels appear in the plot legend.
    # Set view axes to match the extent of the system.
    # Also set equal axes to prevent z exaggeration.
    if fig_rec.view.is_3d():
        fig_rec.view.show(
            equal=True,
            x_limits=debug.world_box[0],
            y_limits=debug.world_box[1],
            z_limits=debug.world_box[2],
            show=show,
            legend=False,
        )
    else:
        fig_rec.view.show(equal=True, show=show, legend=True)

    # Construct filename and save figure.
    figure_title_clean = figure_title.replace(' ', '_').replace(',', '')
    axis_prefix_clean = axis_prefix.replace(' ', '').lower()
    figure_file_body = f"{debug.figure_idx:02d}_{phase}_{figure_title_clean}_{axis_prefix_clean}"
    debug.figure_idx += 1
    fig_rec.save(
        output_dir=debug.save_dir,
        output_file_body=figure_file_body,
        dpi=200,
        format='png',
        close_after_save=True,
        include_view_suffix=True,
        include_limit_suffix=False,
    )


# SOFAST SETUP FIGURES


def start_and_draw_sofast_setup_figure(
    figure_title: str,
    view_spec: dict,
    camera: Camera,
    facet_data: DefinitionFacet,
    # Locations.
    trans_cam_screen: txyz.TransformXYZ | None,
    trans_mirror_screen: txyz.TransformXYZ | None,
    # Debug information carrier.
    debug: DebugOpticsGeometry,
    az_el_roll_deg: tuple[float, float, float] = None,
    grid: bool = True,
    axis_prefix: str = None,
) -> tuple[rcfg.RenderControlFigure, txyz.TransformXYZ, txyz.TransformXYZ, txyz.TransformXYZ]:
    """
    Sets up and draws a figure showing SOFAST world, screen, camera, and mirror.
    Returns a handle to the figure so that additional items can be added.

    Assumes world_transform is the identity transform.
    """
    # Since SOFAST layout are complex and include multiple features and labels
    # in close proximity, we make the figure larger, which has the effect of
    # making the default axis linewidths, fonts, etc effectively smaller.
    # fig_size = (12.8, 9.6)  # Inch.  Normal is (6.4, 4.8).
    fig_size = (9.6, 7.2)  # Inch.  Normal is (6.4, 4.8).
    fig_rec = start_debug_3d_figure(
        figure_title, view_spec=view_spec, equal=True, grid=grid, axis_prefix=axis_prefix, figsize=fig_size
    )

    # Set view direction, if desired.
    if az_el_roll_deg is not None:
        if not fig_rec.view.is_3d():
            lt.error_and_raise(
                ValueError,
                "In start_draw_and_finish_sofast_setup_figure(), asked to set view direction for a non-3d plot.",
            )
        azimuth_deg = az_el_roll_deg[0]
        elevation_deg = az_el_roll_deg[1]
        roll_deg = az_el_roll_deg[2]
        lt.info(
            'In start_draw_and_finish_sofast_setup_figure(), setting view (azimuth, elevation, roll) to '
            + str((azimuth_deg, elevation_deg, roll_deg))
            + ' degrees.'
        )
        fig_rec.view.axis.view_init(azim=azimuth_deg, elev=elevation_deg, roll=roll_deg)

    # Set the same tick mark interval for all axes, while still allowing automatic axis limits.
    # Source - https://stackoverflow.com/a/36229671
    # Posted by jthomas
    # Retrieved 2026-02-26, License - CC BY-SA 3.0
    #    import matplotlib.ticker as ticker
    if fig_rec.view.is_3d():
        tick_spacing = 0.5  # &&&& DELETE-SCAFFOLDING -- CONTROL OR PASS THIS IN
    else:
        tick_spacing = 0.1  # &&&& DELETE-SCAFFOLDING -- CONTROL OR PASS THIS IN
    fig_rec.view.axis.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    fig_rec.view.axis.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    if fig_rec.view.is_3d():
        fig_rec.view.axis.zaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    # Camera pose.
    if trans_cam_screen is None:
        trans_camera_world = None
    else:
        trans_camera_world = debug.trans_screen_world * trans_cam_screen

    # Mirror pose.
    if trans_mirror_screen is None:
        trans_mirror_world = None
    else:
        trans_mirror_world = debug.trans_screen_world * trans_mirror_screen

    # Select render control.
    if (trans_cam_screen is None) and (trans_mirror_screen is None):
        sofast_setup_style = rcss.NoCameraNoMirror()
    elif trans_cam_screen is None:
        sofast_setup_style = rcss.NoCamera()
    elif trans_mirror_screen is None:
        sofast_setup_style = rcss.NoMirror()
    else:
        sofast_setup_style = rcss.RenderControlSofastSetup()

    # Draw SOFAST setup.
    # Import here, to avoid circular import.
    import opencsp.app.sofast.lib.SofastConfiguration as sfcfg

    sfcfg.draw_sofast_setup(
        # Where to draw
        view=fig_rec.view,
        # Objects
        sofast_is_fringe=True,
        sofast_is_fixed=False,
        camera=camera,
        display=debug.display,  # Used only for diagnostic rendering.
        dot_locations=None,  # &&&& DELETE-SCAFFOLDING -- PASS THIS IN
        mirror=debug.mirror,  # Used only for diagnostic rendering.
        facet_data=facet_data,
        # Extent
        world_box=debug.world_box,
        # Locations
        world_transform=None,
        screen_transform=debug.trans_screen_world,
        camera_transform=trans_camera_world,
        mirror_transform=trans_mirror_world,
        # Render control
        sofast_setup_style=sofast_setup_style,
        z_axis_fov_distance=debug.z_axis_fov_distance,
        mirror_needle_length=debug.mirror_needle_length,
        axis_length=debug.axis_length,
    )

    # Return.
    return fig_rec, debug.trans_screen_world, trans_camera_world, trans_mirror_world
