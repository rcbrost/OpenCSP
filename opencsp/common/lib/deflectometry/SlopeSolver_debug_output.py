"""Functions supporting diagnosis of slope solution for a deflectometry setup."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.deflectometry.SlopeSolverDataDebug import SlopeSolverDataDebug
from opencsp.app.sofast.lib.DebugOpticsGeometry import DebugOpticsGeometry
from opencsp.app.sofast.lib.DefinitionFacet import DefinitionFacet
from opencsp.app.sofast.lib.DisplayShape import DisplayShape as Display
import opencsp.app.sofast.lib.sofast_debug_figure_support as sdfs
from opencsp.app.sofast.lib.SpatialOrientation import SpatialOrientation
from opencsp.common.lib.deflectometry.Surface2DAbstract import Surface2DAbstract
import opencsp.common.lib.geometry.TransformXYZ as txyz
from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz
import opencsp.common.lib.render.view_spec as vs
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.render_control.RenderControlSofastSetup as rcssp
import opencsp.common.lib.tool.log_tools as lt


# SOLVER LOOP PROGRESS SUMMARY


def fit_surface_loop_record_column_headings() -> str:
    #       22  7777777 7777777 7777777 7777777 7777777 7777777   999999999    999999999   7777777 7777777 7777777 7777777 7777777 7777777 7777777
    return "idx    c0     c1x     c2x2    c3y    c4xy    c5y2    Dcorner_min  Dcorner_max    rcx     rcy     rcz     tcx     tcy     tcz     |tc|    n_int"


def fit_surface_loop_record_column_headings_units() -> str:
    #      "idx    c0     c1x     c2x2    c3y    c4xy    c5y2    Dcorner_min  Dcorner_max    rcx     rcy     rcz     tcx     tcy     tcz     |tc|    n_int"
    return " -     (m)     -      (1/m)    -     (1/m)   (1/m)       (m)          (m)         (Rodriguez vector)     (m)     (m)     (m)     (m)       -"


def fit_surface_loop_record_column_headings_separator() -> str:
    #      "idx    c0     c1x     c2x2    c3y    c4xy    c5y2    Dcorner_min  Dcorner_max    rcx     rcy     rcz     tcx     tcy     tcz     |tc|    n_int"
    return "----------------------------------------------------------------------------------------------------------------------------------------------"


def fit_surface_loop_record_str(loop_record: dict) -> str:
    idx_str = f"{loop_record['loop_idx']:2d}"
    # Surface coefficients
    c0_str = f"{loop_record['surf_coefs'][0]:7.4f}"
    c1x_str = f"{loop_record['surf_coefs'][1]:7.4f}"
    c2x2_str = f"{loop_record['surf_coefs'][2]:7.4f}"
    c3y_str = f"{loop_record['surf_coefs'][3]:7.4f}"
    c4xy_str = f"{loop_record['surf_coefs'][4]:7.4f}"
    c5y2_str = f"{loop_record['surf_coefs'][5]:7.4f}"
    # Change in corners
    vxyz_corner_change = loop_record['vxyz_corner_change']
    change_min = vxyz_corner_change.data[2, :].min()
    change_max = vxyz_corner_change.data[2, :].max()
    Dcorner_min_str = f"{change_min:9.6f}"
    Dcorner_max_str = f"{change_max:9.6f}"
    # Camera pose
    rcx_str = f"{loop_record['r_cam_optic'].as_rotvec()[0]:7.4f}"
    rcy_str = f"{loop_record['r_cam_optic'].as_rotvec()[1]:7.4f}"
    rcz_str = f"{loop_record['r_cam_optic'].as_rotvec()[2]:7.4f}"
    tcx = loop_record['v_cam_optic_cam'].x[0]
    tcy = loop_record['v_cam_optic_cam'].y[0]
    tcz = loop_record['v_cam_optic_cam'].z[0]
    tcx_str = f"{tcx:7.4f}"
    tcy_str = f"{tcy:7.4f}"
    tcz_str = f"{tcz:7.4f}"
    norm_tc = np.sqrt((tcx * tcx) + (tcy * tcy) + (tcz * tcz))
    tcy_str = f"{loop_record['v_cam_optic_cam'].y[0]:7.4f}"
    tcz_str = f"{loop_record['v_cam_optic_cam'].z[0]:7.4f}"
    # Camera-to-mirror distance
    norm_tc_str = f"{norm_tc:7.4f}"
    # Number of intersection points
    n_int_str = f"{loop_record['n_intersect']:7d}"
    return f"{idx_str}  {c0_str} {c1x_str} {c2x2_str} {c3y_str} {c4xy_str} {c5y2_str}   {Dcorner_min_str}    {Dcorner_max_str}   {rcx_str} {rcy_str} {rcz_str} {tcx_str} {tcy_str} {tcz_str} {norm_tc_str}  {n_int_str}"


# INTERSECTION SURFACE, DEFINING VERTICES, CAMERA


def figure_intersection_surface_situation(
    figure_title: str,
    v_facet_corners_hires_1: Vxyz | None,
    v_facet_corners_hires_2: Vxyz | None,
    surface: Surface2DAbstract,
    orientation: SpatialOrientation,
    loop_idx: int,
    debug: SlopeSolverDataDebug,
    sofast_setup_style: rcssp.RenderControlSofastSetup = rcssp.RenderControlSofastSetup(),
    sofast_setup_axis_length: float = 0.1,  # meters
    sofast_setup_z_axis_fov_distance=1.0,  # meters
    mirror_needle_length: float = 0.5,  # meters
    plot_camera_rays: bool = True,
    camera_ray_downsample: int = 500,
    camera_ray_length: float = 1.0,  # meters
    plot_intersection_points: bool = True,
    intersection_points_downsample: int = 50,
    plot_screen_points: bool = True,
    screen_points_downsample: int = 50,
):
    """Sets up and draws a figure showing SOFAST layout, including mirror, screen, and camera."""
    camera = debug.debug_geometry.camera  # &&&& DELETE-SCAFFOLDING -- PASS THIS IN?
    display = debug.debug_geometry.display  # &&&& DELETE-SCAFFOLDING -- PASS THIS IN?

    # Draw views in the world coordinate system.
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
        figure_intersection_surface_situation_world(
            this_title,
            camera=camera,
            display=display,
            v_facet_corners_hires_1=v_facet_corners_hires_1,
            v_facet_corners_hires_2=v_facet_corners_hires_2,
            surface=surface,
            orientation=orientation,
            loop_idx=loop_idx,
            debug=debug,
            sofast_setup_style=sofast_setup_style,
            sofast_setup_axis_length=sofast_setup_axis_length,
            sofast_setup_z_axis_fov_distance=sofast_setup_z_axis_fov_distance,
            mirror_needle_length=mirror_needle_length,
            plot_camera_rays=plot_camera_rays,
            camera_ray_downsample=camera_ray_downsample,
            camera_ray_length=camera_ray_length,
            plot_intersection_points=plot_intersection_points,
            intersection_points_downsample=intersection_points_downsample,
            plot_screen_points=plot_screen_points,
            screen_points_downsample=screen_points_downsample,
            view_spec_az_el_roll_deg=view_spec_az_el_roll,
        )

    # Draw views in the optic coordinate system.
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
        figure_intersection_surface_situation_optic(
            this_title,
            camera=camera,
            display=display,
            v_facet_corners_hires_1=v_facet_corners_hires_1,
            v_facet_corners_hires_2=v_facet_corners_hires_2,
            surface=surface,
            orientation=orientation,
            loop_idx=loop_idx,
            debug=debug,
            sofast_setup_style=sofast_setup_style,
            sofast_setup_axis_length=sofast_setup_axis_length,
            sofast_setup_z_axis_fov_distance=sofast_setup_z_axis_fov_distance,
            mirror_needle_length=mirror_needle_length,
            plot_camera_rays=plot_camera_rays,
            camera_ray_downsample=camera_ray_downsample,
            camera_ray_length=camera_ray_length,
            plot_intersection_points=plot_intersection_points,
            intersection_points_downsample=intersection_points_downsample,
            plot_screen_points=plot_screen_points,
            screen_points_downsample=screen_points_downsample,
            view_spec_az_el_roll_deg=view_spec_az_el_roll,
        )


def figure_intersection_surface_situation_world(
    title: str,
    camera: Camera,
    display: Display,
    v_facet_corners_hires_1: Vxyz | None,
    v_facet_corners_hires_2: Vxyz | None,
    surface: Surface2DAbstract,
    orientation: SpatialOrientation,
    loop_idx: int,
    debug: SlopeSolverDataDebug,
    downsample: int = 500,
    sofast_setup_style: rcssp.RenderControlSofastSetup = rcssp.RenderControlSofastSetup(),
    sofast_setup_axis_length: float = 0.1,  # meters
    sofast_setup_z_axis_fov_distance: float = 1.0,  # meters
    mirror_needle_length: float = 1.0,  # meters
    plot_camera_rays: bool = True,
    camera_ray_downsample: int = 500,
    camera_ray_length: float = 0.0,  # meters
    plot_intersection_points: bool = True,
    intersection_points_downsample: int = 50,
    plot_screen_points: bool = True,
    screen_points_downsample: int = 50,
    view_spec_az_el_roll_deg: list[dict, tuple[float, float, float]] = None,
) -> None:
    """Supports routine without aux extension."""
    # Fetch axis data.
    view_spec = view_spec_az_el_roll_deg[0]
    az_el_roll_deg = view_spec_az_el_roll_deg[1]

    # Create a new figure.
    full_title = f"Slope Solver (loop_idx={loop_idx:d}): " + title
    axis_prefix = "World "
    fig_rec = sdfs.start_debug_3d_figure(
        figure_title=full_title, view_spec=view_spec, figsize=(12, 9), axis_prefix=axis_prefix
    )

    # Import here, to avoid circular import.
    import opencsp.app.sofast.lib.SofastConfiguration as sfcfg

    trans_screen_world = debug.debug_geometry.trans_screen_world
    trans_camera_world = trans_screen_world * orientation.trans_screen_cam.inv()
    trans_mirror_world = trans_screen_world * orientation.trans_screen_optic.inv()

    # Plot camera rays
    if plot_camera_rays:
        u_active_pixel_pointing_optic_downsample = surface.u_active_pixel_pointing_optic[::camera_ray_downsample]
        u_active_pixel_pointing_cam = u_active_pixel_pointing_optic_downsample.rotate(orientation.r_cam_optic.inv())
        u_active_pixel_pointing_world = u_active_pixel_pointing_cam.rotate(trans_camera_world.R)
        for vxyz_ray in u_active_pixel_pointing_world:
            xs = [trans_camera_world.V.x, trans_camera_world.V.x + (vxyz_ray.x * camera_ray_length)]
            ys = [trans_camera_world.V.y, trans_camera_world.V.y + (vxyz_ray.y * camera_ray_length)]
            zs = [trans_camera_world.V.z, trans_camera_world.V.z + (vxyz_ray.z * camera_ray_length)]
            vxyz_ray = Vxyz([xs, ys, zs])
            vxyz_ray.draw_line(fig_rec, style=rcps.outline(color="pink"))  # Don't label -- too many rays

    # Plot camera-ray-to-mirror intersection points.
    # The trisurf plot is only supported for 3-d axes.
    if fig_rec.view.is_3d() and plot_intersection_points:
        # # In optic coordinates.
        v_surf_int_pts_optic = surface.v_surf_int_pts_optic
        # surface.plot_intersection_points(fig_rec.view.axis, downsample=intersection_points_downsample)
        # # In screen coordinates.
        v_surf_int_pts_screen = orientation.trans_screen_optic.inv().apply(v_surf_int_pts_optic)
        # In world coordinates.
        v_surf_int_pts_world = trans_screen_world.apply(v_surf_int_pts_screen)
        fig_rec.view.axis.plot_trisurf(
            *v_surf_int_pts_world[::intersection_points_downsample].data,
            edgecolor="none",
            alpha=0.5,
            linewidth=0,
            antialiased=False,
        )

    # Plot screen points.
    # The trisurf plot is only supported for 3-d axes.
    if fig_rec.view.is_3d() and plot_screen_points:
        # # In optic coordinates.
        v_screen_points_optic = surface.v_screen_points_optic
        # surface.plot_screen_points(fig_rec.view.axis, downsample=screen_points_downsample)
        # # In screen coordinates.
        v_screen_points_screen = orientation.trans_screen_optic.inv().apply(v_screen_points_optic)
        # In world coordinates.
        v_screen_points_world = trans_screen_world.apply(v_screen_points_screen)
        try:
            fig_rec.view.axis.plot_trisurf(
                *v_screen_points_world[::screen_points_downsample].data,
                edgecolor="none",
                alpha=0.5,
                linewidth=0,
                antialiased=False,
            )
        except RuntimeError as e:
            # If the screen is perfectly flat, then the screen points are degenerate, because
            # they do not span a 3-d convex hull with volume.  This causes the Delauney
            # triangulation to throw an error.
            # To rectify this situation, we will dither a single point to make it non-degenerate.
            if "singular input data" in str(e):
                try:
                    v_screen_points_world_downsample = v_screen_points_world[::screen_points_downsample]
                    v_screen_points_world_downsample.data[0][0] += 0.0005
                    v_screen_points_world_downsample.data[1][0] += 0.0005
                    v_screen_points_world_downsample.data[2][0] += 0.0005
                    fig_rec.view.axis.plot_trisurf(
                        *v_screen_points_world_downsample.data,
                        edgecolor="none",
                        alpha=0.5,
                        linewidth=0,
                        antialiased=False,
                    )
                except RuntimeError as e2:
                    if "singular input data" in str(e2):
                        # Then our attempted fix is still singular, meaning that the constants we
                        # added caused the point to shift within the degenerate plane.  Now we do
                        # this again, but with a different offset.  They can't both be within the
                        # same degenerate plane.
                        v_screen_points_world_downsample = v_screen_points_world[::screen_points_downsample]
                        v_screen_points_world_downsample.data[1][0] -= 0.001
                        fig_rec.view.axis.plot_trisurf(
                            *v_screen_points_world_downsample.data,
                            edgecolor="none",
                            alpha=0.5,
                            linewidth=0,
                            antialiased=False,
                        )

    # Draw screen, camera, and mirror.
    sfcfg.draw_sofast_setup(
        # Where to draw
        view=fig_rec.view,
        # Objects
        sofast_is_fringe=True,
        sofast_is_fixed=False,
        camera=camera,
        display=display,
        dot_locations=None,  # &&&& DELETE-SCAFFOLDING -- PASS THIS IN
        mirror=debug.debug_geometry.mirror,  # &&&& DELETE-SCAFFOLDING -- PASS THIS IN?
        facet_data=debug.debug_geometry.facet_data,
        # Extent
        world_box=debug.debug_geometry.world_box,
        # Locations
        world_transform=None,
        screen_transform=trans_screen_world,
        camera_transform=trans_camera_world,
        mirror_transform=trans_mirror_world,
        # Render control
        sofast_setup_style=sofast_setup_style,
        z_axis_fov_distance=sofast_setup_z_axis_fov_distance,
        mirror_needle_length=mirror_needle_length,
        axis_length=sofast_setup_axis_length,
    )

    # Plot high-resolution facet corners, before fit_slopes() execution.
    # Use draw_line(), because we only want one legend entry, not a legend entry for every point.
    if v_facet_corners_hires_1 is not None:
        if v_facet_corners_hires_2 is None:
            label_str = "Facet Vertices (High Resolution)"
        else:
            label_str = "Hires Vertices (Before Fit)"
        transformed_v_facet_corners_hires_1 = trans_mirror_world.apply(v_facet_corners_hires_1)
        transformed_v_facet_corners_hires_1.draw_line(
            fig_rec, style=rcps.marker(marker='.', color='b', markersize=4), label=label_str
        )

    # Plot high-resolution facet corners, after fit_slopes() execution.
    # Use draw_line(), because we only want one legend entry, not a legend entry for every point.
    if v_facet_corners_hires_2 is not None:
        label_str = "Hires Vertices (After Fit)"
        transformed_v_facet_corners_hires_2 = trans_mirror_world.apply(v_facet_corners_hires_2)
        transformed_v_facet_corners_hires_2.draw_line(
            fig_rec, style=rcps.marker(marker='.', color='red', markersize=2), label=label_str
        )

    # Set view direction, if desired.
    if az_el_roll_deg is not None:
        if not fig_rec.view.is_3d():
            lt.error_and_raise(
                ValueError,
                "In SlopeSolver_debug_outupt.py:figure_intersection_surface_situation_aux(), asked to set view direction for a non-3d plot.",
            )
        azimuth_deg = az_el_roll_deg[0]
        elevation_deg = az_el_roll_deg[1]
        roll_deg = az_el_roll_deg[2]
        lt.info(
            'In SlopeSolver_debug_outupt.py:figure_intersection_surface_situation_aux(), setting view (azimuth, elevation, roll) to '
            + str((azimuth_deg, elevation_deg, roll_deg))
            + ' degrees.'
        )
        fig_rec.view.axis.view_init(azim=azimuth_deg, elev=elevation_deg, roll=roll_deg)

    # Add legend.
    plt.legend()

    # Save and close.
    full_title_for_file = (
        full_title.replace(' ', '_').replace(':', '').replace('(', '').replace(')', '').replace(',', '')
    )
    sdfs.finish_debug_3d_figure(full_title_for_file, 'solver', fig_rec, debug.debug_geometry, axis_prefix=axis_prefix)


def figure_intersection_surface_situation_optic(
    title: str,
    camera: Camera,
    display: Display,
    v_facet_corners_hires_1: Vxyz | None,
    v_facet_corners_hires_2: Vxyz | None,
    surface: Surface2DAbstract,
    orientation: SpatialOrientation,
    loop_idx: int,
    debug: SlopeSolverDataDebug,
    sofast_setup_style: rcssp.RenderControlSofastSetup = rcssp.RenderControlSofastSetup(),
    sofast_setup_axis_length: float = 0.1,  # meters
    sofast_setup_z_axis_fov_distance=1.0,  # meters
    mirror_needle_length: float = 1.0,  # meters
    plot_camera_rays: bool = True,
    camera_ray_downsample: int = 500,
    camera_ray_length: float = 1.0,  # meters
    plot_intersection_points: bool = True,
    intersection_points_downsample: int = 50,
    plot_screen_points: bool = True,
    screen_points_downsample: int = 50,
    view_spec_az_el_roll_deg: list[dict, tuple[float, float, float]] = None,
) -> None:
    """Supports routine without aux extension."""
    # Fetch axis data.
    view_spec = view_spec_az_el_roll_deg[0]
    az_el_roll_deg = view_spec_az_el_roll_deg[1]

    # Create a new figure.
    full_title = f"Slope Solver (loop_idx={loop_idx:d}): " + title
    axis_prefix = "Optic "
    fig_rec = sdfs.start_debug_3d_figure(
        figure_title=full_title, view_spec=view_spec, figsize=(12, 9), axis_prefix=axis_prefix
    )

    # Plot camera rays
    if plot_camera_rays:
        for ray in surface.u_active_pixel_pointing_optic[::camera_ray_downsample]:
            xs = [orientation.v_optic_cam_optic.x, orientation.v_optic_cam_optic.x + (ray.x * camera_ray_length)]
            ys = [orientation.v_optic_cam_optic.y, orientation.v_optic_cam_optic.y + (ray.y * camera_ray_length)]
            zs = [orientation.v_optic_cam_optic.z, orientation.v_optic_cam_optic.z + (ray.z * camera_ray_length)]
            vxyz_ray = Vxyz([xs, ys, zs])
            vxyz_ray.draw_line(fig_rec, style=rcps.outline(color="pink"))  # Don't label -- too many rays

    # Plot camera-ray-to-mirror intersection points.
    # The trisurf plot is only supported for 3-d axes.
    if fig_rec.view.is_3d() and plot_intersection_points:
        surface.plot_intersection_points(fig_rec.view.axis, downsample=intersection_points_downsample)

    # Plot screen points.
    # The trisurf plot is only supported for 3-d axes.
    if fig_rec.view.is_3d() and plot_screen_points:
        surface.plot_screen_points(fig_rec.view.axis, downsample=screen_points_downsample)

    # Import here, to avoid circular import.
    import opencsp.app.sofast.lib.SofastConfiguration as sfcfg

    # Plot camera axes and field of view.
    camera_transform = txyz.TransformXYZ.from_R_V(orientation.r_optic_cam.inv(), orientation.v_optic_cam_optic)
    sfcfg.draw_camera(
        fig_rec.view,
        camera=camera,
        transform=camera_transform,
        sofast_camera_style=sofast_setup_style.sofast_camera_style,
        z_axis_fov_distance=sofast_setup_z_axis_fov_distance,
        axis_length=sofast_setup_axis_length,
    )

    # Plot screen axes and boundary.
    screen_transform = txyz.TransformXYZ.from_R_V(orientation.r_optic_screen.inv(), orientation.v_optic_screen_optic)
    sfcfg.draw_screen(
        fig_rec.view,
        sofast_is_fringe=True,
        sofast_is_fixed=False,
        display=display,
        dot_locations=None,
        transform=screen_transform,
        sofast_screen_style=sofast_setup_style.sofast_screen_style,
        axis_length=sofast_setup_axis_length,
    )

    # Plot original facet corners.
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

    # Plot fit normal at align point
    v_fit = surface.normal_fit_at_align_point()
    v_fit_pt1 = surface.v_align_point_optic
    v_fit_pt2 = surface.v_align_point_optic + (v_fit.as_Vxyz() * mirror_needle_length)
    fit_normal_in_place = v_fit_pt1.concatenate(v_fit_pt2)
    fit_normal_in_place.draw_line(fig_rec, style=rcps.outline(color='m'), label="Fit Normal")
    # Plot design normal at align point
    v_des = surface.normal_design_at_align_point()
    v_des_pt1 = surface.v_align_point_optic
    v_des_pt2 = surface.v_align_point_optic + (v_des.as_Vxyz() * mirror_needle_length)
    design_normal_in_place = v_des_pt1.concatenate(v_des_pt2)
    design_normal_in_place.draw_line(fig_rec, style=rcps.outline(color='k', linestyle="--"), label="Design Normal")

    # Plot other points
    # Use draw_line(), because we only want one legend entry, not a legend entry for every point.
    # &&&& DELETE-SCAFFOLDING -- MARKERSIZE WAS 3, COLOR WAS 'cyan'
    surface.v_align_point_optic.draw_line(
        fig_rec, style=rcps.marker(marker='.', color='red', markersize=15), label="Align Point"
    )
    # orientation.v_optic_cam_optic.draw_line(
    #     fig_rec, style=rcps.marker(marker='*', color='k', markersize=7), label="Camera"
    # )
    # orientation.v_optic_screen_optic.draw_line(
    #     fig_rec, style=rcps.marker(marker='+', color='k', markersize=7), label="Screen Center"
    # )

    # Set view direction, if desired.
    if az_el_roll_deg is not None:
        if not fig_rec.view.is_3d():
            lt.error_and_raise(
                ValueError,
                "In SlopeSolver_debug_outupt.py:figure_intersection_surface_situation_aux(), asked to set view direction for a non-3d plot.",
            )
        azimuth_deg = az_el_roll_deg[0]
        elevation_deg = az_el_roll_deg[1]
        roll_deg = az_el_roll_deg[2]
        lt.info(
            'In SlopeSolver_debug_outupt.py:figure_intersection_surface_situation_aux(), setting view (azimuth, elevation, roll) to '
            + str((azimuth_deg, elevation_deg, roll_deg))
            + ' degrees.'
        )
        fig_rec.view.axis.view_init(azim=azimuth_deg, elev=elevation_deg, roll=roll_deg)

    # Add legend.
    plt.legend()

    # Save and close.
    full_title_for_file = (
        full_title.replace(' ', '_').replace(':', '').replace('(', '').replace(')', '').replace(',', '')
    )
    sdfs.finish_debug_3d_figure(full_title_for_file, 'solver', fig_rec, debug.debug_geometry, axis_prefix=axis_prefix)


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


# INITIAL REPROJECTION SUMMARY, AFTER SNAP TO EDGE (WITHOUT COARSE VERTICES)


def reproj_snap_2(hires_pts_reproj_1: Vxy, hires_pts_reproj_snap_1: Vxy, debug: SlopeSolverDataDebug) -> None:
    reproj_snap_2_aux(hires_pts_reproj_1, hires_pts_reproj_snap_1, label_points=True, debug=debug)
    reproj_snap_2_aux(hires_pts_reproj_1, None, label_points=False, debug=debug)
    reproj_snap_2_aux(None, hires_pts_reproj_snap_1, label_points=False, debug=debug)


def reproj_snap_2_aux(
    hires_pts_reproj_1: Vxy | None, hires_pts_reproj_snap_1: Vxy | None, label_points: bool, debug: SlopeSolverDataDebug
) -> None:
    figure_title = "Reprojected Points, and Snap to Edges"
    fig_rec = sdfs.start_debug_image_figure(figure_title)
    fig_rec.view.imshow(debug.debug_geometry.mask_processed, cmap="gray")
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
