"""Functions supporting diagnosis of slope solution for a deflectometry setup."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from opencsp.common.lib.camera.Camera import Camera
import opencsp.common.lib.deflectometry.slope_fitting_2d as sf2
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
import opencsp.common.lib.render_control.RenderControlFigure as rcfg
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.render_control.RenderControlSofastSetup as rcssp
import opencsp.common.lib.tool.log_tools as lt

# SOLVER LOOP PROGRESS SUMMARY


def fit_surface_loop_record_column_headings() -> str:
    #       22.22.22  7777777 7777777 7777777 7777777 7777777 7777777   999999999    999999999   7777777 7777777 7777777 7777777 7777777 7777777 7777777
    return "idx          c0     c1x     c2x2    c3y    c4xy    c5y2    Dcorner_min  Dcorner_max    rcx     rcy     rcz     tcx     tcy     tcz     |tc|    n_int"


def fit_surface_loop_record_column_headings_units() -> str:
    #      "idx          c0     c1x     c2x2    c3y    c4xy    c5y2    Dcorner_min  Dcorner_max    rcx     rcy     rcz     tcx     tcy     tcz     |tc|    n_int"
    return " -           (m)     -      (1/m)    -     (1/m)   (1/m)       (m)          (m)         (Rodriguez vector)     (m)     (m)     (m)     (m)       -"


def fit_surface_loop_record_column_headings_separator() -> str:
    #      "idx          c0     c1x     c2x2    c3y    c4xy    c5y2    Dcorner_min  Dcorner_max    rcx     rcy     rcz     tcx     tcy     tcz     |tc|    n_int"
    return "----------------------------------------------------------------------------------------------------------------------------------------------------"


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
    return f"{idx_str}  {c0_str} {c1x_str} {c2x2_str} {c3y_str} {c4xy_str} {c5y2_str}   {Dcorner_min_str}    {Dcorner_max_str}   {rcx_str} {rcy_str} {rcz_str} WHAT?? {tcx_str} {tcy_str} {tcz_str} {norm_tc_str}  {n_int_str}"


# SOLVER LOOP PROGRESS SUMMARY 2


def fit_surface_loop_record_column_headings_2() -> str:
    #        0.00.00   [  0.000000,   0.000000,   1.000000]     [  0.000000,   0.000000,   1.000000]     [  0.000000,   0.000000,   1.000000]
    #        2.00.00   [  0.611923,  -0.185481,   0.768861]     [  0.603300,  -0.215173,   0.767939]     [ -0.008624,  -0.029692,  -0.000922]     [-6.78254892  0.94170392  5.70583218]
    return "idx                       avg_fit                                avg_measured                      avg_fit_minus_measured                        r_align_step_str"


def fit_surface_loop_record_column_headings_units_2() -> str:
    #      "idx                       avg_fit                                avg_measured                      avg_fit_minus_measured                        r_align_step_str"
    return " -                          (m)                                       (m)                                    (m)                                  (deg rx,ry,rz)"


def fit_surface_loop_record_column_headings_separator_2() -> str:
    #      "idx                       avg_fit                                avg_measured                      avg_fit_minus_measured                        r_align_step_str"
    return "---------------------------------------------------------------------------------------------------------------------------------------------------------------------------"


def fit_surface_loop_record_str_2(loop_record: dict) -> str:
    idx_str = f"{loop_record['loop_idx']:2d}"
    # Average surface normals
    avg_fit_str = loop_record["u_avg_fit_normal"].to_str()
    avg_meas_str = loop_record["u_avg_measured_normal"].to_str()
    avg_meas_minus_fit_str = loop_record["u_avg_measured_minus_fit"].to_str()
    if loop_record["r_align_step"] is None:
        #                   [-6.78254892  0.94170392  5.70583218]
        r_align_step_str = "                  -                  "
    else:
        r_align_step_str = loop_record["r_align_step"].as_euler('XYZ', degrees=True)
    return f"{idx_str}   {avg_fit_str}     {avg_meas_str}     {avg_meas_minus_fit_str}     {r_align_step_str}"


# SOLVER LOOP PROGRESS SUMMARY - ACTIONS


def fit_surface_loop_record_column_headings_action() -> str:
    return "idx        actions"


def fit_surface_loop_record_column_headings_separator_action() -> str:
    return "--------------------------------------------------------------------------------------------------------------------------------------------------"


def fit_surface_loop_record_str_action(loop_record: dict, only_loop_select_actions=False) -> str:
    # Action list
    # # &&&& DELETE-SCAFFOLDING -- DELETE AFTER TESTING
    # action_list_str = str(loop_record["loop_action_list"])
    action_list = loop_record["loop_action_list"]
    # Add items.
    # Loop index.
    action_list_str = f"{loop_record['loop_idx']:2d}"
    # Subloop index.
    if "subloop_idx" in loop_record.keys():
        action_list_str += f".{loop_record['subloop_idx']:02d}"
    else:
        action_list_str += "   "
    # Subsubloop index.
    if "subsubloop_idx" in loop_record.keys():
        action_list_str += f".{loop_record['subsubloop_idx']:02d}"
    else:
        action_list_str += "   "
    # Pose refinement.
    action_list_str += "   "
    if "find_best_camera_pose_preserving_aim_and_distance" in action_list:
        action_list_str += "ref_pos              "
    elif "find_best_camera_translation_of_distance_r" in action_list:
        action_list_str += f"ref_tr(r={(1000*loop_record['r']):4.1f}mm)     "
    elif "find_best_camera_rotation_of_dtheta_step" in action_list:
        action_list_str += f"ref_rot(dth={(1000*loop_record['dtheta_step']):4.1f}mrad)"
    else:
        #                   ref_rot(dth= 1.7mrad)
        action_list_str += "          -          "
    if not only_loop_select_actions:
        # Orient mirror and camera.
        action_list_str += "   "
        if "orient_optic_cam" in action_list:
            action_list_str += "orient"
        else:
            action_list_str += "  -   "
    if not only_loop_select_actions:
        # Intersect camera rays with optical surface.
        action_list_str += "   "
        if "intersect camera rays" in action_list:
            action_list_str += "int_ray"
        else:
            action_list_str += "   -   "
    if not only_loop_select_actions:
        # Calculate slopes.
        action_list_str += "   "
        if "calculate_slopes" in action_list:
            action_list_str += "clc_slp"
        else:
            action_list_str += "   -   "
    # Align slopes.
    action_list_str += "   "
    if "align fit and measured average slopes" in action_list:
        action_list_str += "align_slope"
    else:
        action_list_str += "     -     "
    if not only_loop_select_actions:
        # Orient mirror and camera.
        action_list_str += "   "
        if "orient_optic_cam_2" in action_list:
            action_list_str += "orient_2"
        else:
            action_list_str += "   -    "
    if not only_loop_select_actions:
        # Intersect camera rays with optical surface, for new camera pose.
        action_list_str += "   "
        if "intersect camera rays for new pose" in action_list:
            action_list_str += "int_ray_2"
        else:
            action_list_str += "    -    "
    if not only_loop_select_actions:
        # Calculate slopes, after align fit and measured.
        action_list_str += "   "
        if "calculate_slopes_after_align" in action_list:
            action_list_str += "clc_slp_2"
        else:
            action_list_str += "    -    "
    # Fit optical surface model to calculated slopes.
    action_list_str += "   "
    if "fit_slopes" in action_list:
        action_list_str += "fit_slp"
    else:
        action_list_str += "   -   "
    if not only_loop_select_actions:
        # Snap facet corners to new embedding surface model.
        action_list_str += "   "
        if "facet corner z values to new surface" in action_list:
            action_list_str += "z_2_sfc"
        else:
            action_list_str += "   -   "
    if not only_loop_select_actions:
        # Project facet corners to image.
        action_list_str += "   "
        if "project updated facet corners to image" in action_list:
            action_list_str += "proj_2_img"
        else:
            action_list_str += "    -     "

    return action_list_str


# SOLVER LOOP PROGRESS SUMMARY - CONVERGENCE PARAMETERS


def fit_surface_loop_record_column_headings_5() -> str:
    return "idx        actions                                           best_dx_dy    best_dth  image_rms         avg_fit_minus_measured             r_align_step_str          D_corner"


def fit_surface_loop_record_column_headings_units_5() -> str:
    #      "idx        actions                                           best_dx_dy    best_dth  image_rms         avg_fit_minus_measured             r_align_step_str          D_corner"
    return " -            -                                                 (mm)        (mrad)     (pix)                    (m)                        (deg rx,ry,rz)              (m)"


def fit_surface_loop_record_column_headings_separator_5() -> str:
    return "------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"


def fit_surface_loop_record_str_5(loop_record: dict) -> str:
    # Loop index and action list
    line_str = fit_surface_loop_record_str_action(loop_record, only_loop_select_actions=True)

    # Camera pose adjustment.
    line_str += "  "
    if "current_best_dx_dy_dtheta" not in loop_record:
        #            [ 23.16,  -4.13]   52.36
        line_str += "       -             -  "
    else:
        best_dx_dy_dtheta = loop_record["current_best_dx_dy_dtheta"]
        best_dx = best_dx_dy_dtheta[0]
        best_dy = best_dx_dy_dtheta[1]
        best_dth = best_dx_dy_dtheta[2]
        line_str += f"[{(1000*best_dx):6.2f}, {(1000*best_dy):6.2f}]"
        line_str += "  "
        line_str += f"{(1000*best_dth):6.2f}"

    # Camera pose reprojection RMS error.
    line_str += "  "
    line_str += "  "  # Add clearance for column heading.
    if ("pose_rms" not in loop_record) or (loop_record["pose_rms"] is None):
        #             4.3489
        line_str += "   -   "
    else:
        line_str += f"{loop_record['pose_rms']:7.4f}"

    # Average surface normals
    line_str += "  "
    if "u_avg_measured_minus_fit" not in loop_record:
        #            [  0.000000,   0.000000,   1.000000]
        line_str += "                   -                "
    else:
        line_str += loop_record["u_avg_measured_minus_fit"].to_str()
    line_str += "  "
    if ("r_align_step" not in loop_record) or (loop_record["r_align_step"] is None):
        #            [-0.03693,0.007841, 0.03119]
        line_str += "             -              "
    else:
        euler_angles = loop_record["r_align_step"].as_euler('XYZ', degrees=True)
        line_str += f"[{euler_angles[0]:8.4f},{euler_angles[1]:8.4f},{euler_angles[2]:8.4f}]"

    # Change in corners
    line_str += "  "
    if "vxyz_corner_change" not in loop_record:
        line_str += "    -    "
    else:
        vxyz_corner_change = loop_record["vxyz_corner_change"]
        change_min = vxyz_corner_change.data[2, :].min()
        change_max = vxyz_corner_change.data[2, :].max()
        abs_change_max = max(abs(change_min), abs(change_max))
        line_str += f"{abs_change_max:9.6f}"

    # Return
    return line_str


# SOLVER LOOP PROGRESS SUMMARY -- SOLUTION HEALTH PARAMETERS


def fit_surface_loop_record_column_headings_6() -> str:
    return "idx        actions                                          c0      c1x    c2x2     c3y    c4xy    c5y2     rcx     rcy     rcz     tcx     tcy     tcz     |s2m|      n_int"


def fit_surface_loop_record_column_headings_units_6() -> str:
    #      "idx        actions                                          c0      c1x    c2x2     c3y    c4xy    c5y2     rcx     rcy     rcz     tcx     tcy     tcz     |s2m|      n_int"
    return " -            -                                             (m)      -     (1/m)     -     (1/m)   (1/m)    (Rodriguez vector)      (m)     (m)     (m)      (m)         -"


def fit_surface_loop_record_column_headings_separator_6() -> str:
    #      "idx        actions                                          c0      c1x    c2x2     c3y    c4xy    c5y2     rcx     rcy     rcz     tcx     tcy     tcz     |s2m|      n_int"
    return "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"


def fit_surface_loop_record_str_6(loop_record: dict) -> str:
    # Loop index and action list
    line_str = fit_surface_loop_record_str_action(loop_record, only_loop_select_actions=True)

    # Camera pose reprojection RMS error.
    # Surface coefficients
    if "surf_coefs" not in loop_record:
        line_str += "    -   "  # c0_str
        line_str += "    -   "  # c1x_str
        line_str += "    -   "  # c2x2_str
        line_str += "    -   "  # c3y_str
        line_str += "    -   "  # c4xy_str
        line_str += "    -   "  # c5y2_str
    else:
        line_str += f" {loop_record['surf_coefs'][0]:7.4f}"  # c0_str
        line_str += f" {loop_record['surf_coefs'][1]:7.4f}"  # c1x_str
        line_str += f" {loop_record['surf_coefs'][2]:7.4f}"  # c2x2_str
        line_str += f" {loop_record['surf_coefs'][3]:7.4f}"  # c3y_str
        line_str += f" {loop_record['surf_coefs'][4]:7.4f}"  # c4xy_str
        line_str += f" {loop_record['surf_coefs'][5]:7.4f}"  # c5y2_str
    # Camera pose
    if "r_cam_optic" not in loop_record:
        line_str += f"    -   "  # rcx_str
        line_str += f"    -   "  # rcy_str
        line_str += f"    -   "  # rcz_str
    else:
        line_str += f" {loop_record['r_cam_optic'].as_rotvec()[0]:7.4f}"  # rcx_str
        line_str += f" {loop_record['r_cam_optic'].as_rotvec()[1]:7.4f}"  # rcy_str
        line_str += f" {loop_record['r_cam_optic'].as_rotvec()[2]:7.4f}"  # rcz_str
    if "v_cam_optic_cam" not in loop_record:
        line_str += "    -   "  # tcx_str
        line_str += "    -   "  # tcy_str
        line_str += "    -   "  # tcz_str
    else:
        line_str += f" {loop_record['v_cam_optic_cam'].x[0]:7.4f}"  # tcx_str
        line_str += f" {loop_record['v_cam_optic_cam'].y[0]:7.4f}"  # tcy_str
        line_str += f" {loop_record['v_cam_optic_cam'].z[0]:7.4f}"  # tcz_str
    # Screen-to-mirror distance
    if ("pose_dist_optic_screen" not in loop_record) or (loop_record["pose_dist_optic_screen"] is None):
        line_str += "      -    "  # norm_s2m_str
    else:
        line_str += f" {loop_record['pose_dist_optic_screen']:9.5f}m"  # norm_s2m_str
    # Number of intersection points
    if "n_intersect" not in loop_record:
        line_str += f"      -    "
    else:
        line_str += f" {loop_record['n_intersect']:8d}"  # n_int_str
    return line_str


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
        # # &&&& DELETE-SCAFFOLDING -- TEMPORARY SHUTOFF
        # (vs.view_spec_3d(), (0, 90, 90)),  # xy
        (vs.view_spec_3d(), (-90, 0, 0)),  # xz
        # (vs.view_spec_3d(), (0, 0, 0)),  # yz
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

    # # &&&& DELETE-SCAFFOLDING -- TEMPORARY SHUTOFF
    # # Draw views in the optic coordinate system.
    # view_spec_az_el_roll_list = [
    #     (vs.view_spec_3d(), None),
    #     (vs.view_spec_3d(), (0, 90, 90)),  # xy
    #     (vs.view_spec_3d(), (-90, 0, 0)),  # xz
    #     (vs.view_spec_3d(), (0, 0, 0)),  # yz
    #     (vs.view_spec_xy(), None),  # Doesn't show intersection surface.
    #     (vs.view_spec_xz(), None),  # Doesn't show intersection surface.
    #     (vs.view_spec_yz(), None),  # Doesn't show intersection surface.
    # ]
    # for view_spec_az_el_roll in view_spec_az_el_roll_list:
    #     az_el_roll_deg = view_spec_az_el_roll[1]
    #     if az_el_roll_deg is None:
    #         this_title = figure_title
    #     else:
    #         this_title = figure_title + ' (Az,El,Roll)=' + str(az_el_roll_deg)
    #     figure_intersection_surface_situation_optic(
    #         this_title,
    #         camera=camera,
    #         display=display,
    #         v_facet_corners_hires_1=v_facet_corners_hires_1,
    #         v_facet_corners_hires_2=v_facet_corners_hires_2,
    #         surface=surface,
    #         orientation=orientation,
    #         loop_idx=loop_idx,
    #         debug=debug,
    #         sofast_setup_style=sofast_setup_style,
    #         sofast_setup_axis_length=sofast_setup_axis_length,
    #         sofast_setup_z_axis_fov_distance=sofast_setup_z_axis_fov_distance,
    #         mirror_needle_length=mirror_needle_length,
    #         plot_camera_rays=plot_camera_rays,
    #         camera_ray_downsample=camera_ray_downsample,
    #         camera_ray_length=camera_ray_length,
    #         plot_intersection_points=plot_intersection_points,
    #         intersection_points_downsample=intersection_points_downsample,
    #         plot_screen_points=plot_screen_points,
    #         screen_points_downsample=screen_points_downsample,
    #         view_spec_az_el_roll_deg=view_spec_az_el_roll,
    #     )


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
    full_title = f"loop={loop_idx:d}: " + title
    axis_prefix = "World "
    fig_rec = sdfs.start_debug_3d_figure(
        figure_title=full_title, view_spec=view_spec, figsize=(12, 9), axis_prefix=axis_prefix
    )

    # Import here, to avoid circular import.
    import opencsp.app.sofast.lib.SofastConfiguration as sfcfg

    trans_screen_world = debug.debug_geometry.trans_screen_world
    trans_camera_world = trans_screen_world * orientation.trans_screen_cam.inv()
    trans_mirror_world = trans_screen_world * orientation.trans_screen_optic.inv()

    # &&&& DELETE-SCAFFOLDING -- ONCE THIS IS STABLE AND IN THE RIGHT DRAW ORDER, DELETE OBSOLETE CODE BLOCK BELOW
    # # Plot rays
    # if plot_camera_rays:
    #     if not hasattr(surface, 'slopes'):
    #         # Then simply draw the camera pixel pointing rays.
    #         u_active_pixel_pointing_optic_downsample = surface.u_active_pixel_pointing_optic[::camera_ray_downsample]
    #         u_active_pixel_pointing_cam = u_active_pixel_pointing_optic_downsample.rotate(orientation.r_cam_optic.inv())
    #         u_active_pixel_pointing_world = u_active_pixel_pointing_cam.rotate(trans_camera_world.R)
    #         for vxyz_ray in u_active_pixel_pointing_world:
    #             xs = [trans_camera_world.V.x, trans_camera_world.V.x + (vxyz_ray.x * camera_ray_length)]
    #             ys = [trans_camera_world.V.y, trans_camera_world.V.y + (vxyz_ray.y * camera_ray_length)]
    #             zs = [trans_camera_world.V.z, trans_camera_world.V.z + (vxyz_ray.z * camera_ray_length)]
    #             vxyz_ray = Vxyz([xs, ys, zs])
    #             vxyz_ray.draw_line(fig_rec, style=rcps.outline(color="pink"))  # Don't label -- too many rays
    #     else:
    #         # Draw rays showing the camera-to-mirror-to-screen reflections
    #         u_active_pixel_pointing_optic = surface.u_active_pixel_pointing_optic
    #         v_surf_int_pts_optic = surface.v_surf_int_pts_optic
    #         slopes_optic = surface.slopes
    #         n_pixel_pointing = u_active_pixel_pointing_optic.len()
    #         n_intersect_pts = v_surf_int_pts_optic.len()
    #         n_slopes = slopes_optic.shape[1]
    #         if n_pixel_pointing != n_intersect_pts:
    #             lt.error_and_raise(
    #                 f"ERROR: In figure_intersection_surface_situation_world(), n_pixel_pointing={n_pixel_pointing} does not match n_intersect_pts={n_intersect_pts}"
    #             )
    #         if n_pixel_pointing != n_slopes:
    #             lt.error_and_raise(
    #                 f"ERROR: In figure_intersection_surface_situation_world(), n_pixel_pointing={n_pixel_pointing} does not match n_slopes={n_slopes}"
    #             )
    #         vxyz_cam_world = trans_camera_world.V
    #         for idx in range(0, n_pixel_pointing, 10000):
    #             # &&&& DELETE-SCAFFOLDING -- CONSIDER VECTORIZING THIS
    #             # Fetch needed data.
    #             uxyz_pixel_ray_optic = u_active_pixel_pointing_optic[idx]
    #             vxyz_int_pt_optic = v_surf_int_pts_optic[idx]
    #             slope_x = slopes_optic[0, idx]
    #             slope_y = slopes_optic[1, idx]
    #             # Construct, check, and draw camera-to-mirror ray.
    #             vxyz_int_pt_world = trans_mirror_world.apply(vxyz_int_pt_optic)
    #             vxyz_camera_to_mirror = vxyz_int_pt_world - vxyz_cam_world
    #             uxyz_camera_to_mirror = vxyz_camera_to_mirror.normalize()
    #             # If the input vectors are both unit length, then the cross product equals
    #             # the sine of the angle.  But if the angle is small, this is simply the
    #             # angle, in radians.  Since we expect the angle to be very small, we'll
    #             # skip the arcsin() call here.
    #             uxyz_pixel_ray_cam = uxyz_pixel_ray_optic.rotate(orientation.r_cam_optic.inv())
    #             uxyz_pixel_ray_world = uxyz_pixel_ray_cam.rotate(trans_camera_world.R)
    #             comparison_angle = uxyz_pixel_ray_world.cross(uxyz_camera_to_mirror).magnitude()[0]
    #             if comparison_angle > 0.0001:  # 0.1 milliradian tolerance
    #                 lt.error_and_raise(
    #                     ValueError,
    #                     f"ERROR: In figure_intersection_surface_situation_world(), comparison_angle={comparison_angle} is not near zero.",
    #                 )
    #             camera_to_mirror_xs = [trans_camera_world.V.x, trans_camera_world.V.x + vxyz_camera_to_mirror.x]
    #             camera_to_mirror_ys = [trans_camera_world.V.y, trans_camera_world.V.y + vxyz_camera_to_mirror.y]
    #             camera_to_mirror_zs = [trans_camera_world.V.z, trans_camera_world.V.z + vxyz_camera_to_mirror.z]
    #             vxyz_camera_to_mirror_world = Vxyz([camera_to_mirror_xs, camera_to_mirror_ys, camera_to_mirror_zs])
    #             vxyz_camera_to_mirror_world.draw_line(fig_rec, style=rcps.outline(color="pink"))
    #             # Construct, check, and draw mirror surface normal.
    #             uxyz_normal_optic = Uxyz([-slope_x, -slope_y, 1])
    #             uxyz_normal_world = uxyz_normal_optic.rotate(trans_mirror_world.R)
    #             normal_world_xs = [vxyz_int_pt_world.x, vxyz_int_pt_world.x + (uxyz_normal_world.x * 0.2)]
    #             normal_world_ys = [vxyz_int_pt_world.y, vxyz_int_pt_world.y + (uxyz_normal_world.y * 0.2)]
    #             normal_world_zs = [vxyz_int_pt_world.z, vxyz_int_pt_world.z + (uxyz_normal_world.z * 0.2)]
    #             vxyz_normal_world = Vxyz([normal_world_xs, normal_world_ys, normal_world_zs])
    #             vxyz_normal_world.draw_line(fig_rec, style=rcps.outline(color="pink", linestyle='--', linewidth=0.25))
    #             # Construct, check, and draw mirror-to-screen ray.
    #             # Given an incident vector i and a surface normal n, with normalized versions u_i and u_n,
    #             # respectively, then the formula for the unit-length reflected ray u_r is:
    #             #
    #             #    u_r = u_i - [2(u_i dot u_n) u_n]
    #             #
    #             # For details, see https://math.stackexchange.com/questions/13261/how-to-get-a-reflection-vector#:~:text=Reflection%20Vector%20Formula:%20The%20reflection%20vector%20(r),the%20formula%20r%20=%20d%20-%202(d%E2%8B%85n)n.
    #             #
    #             # Set u_i and u_n, as Vxyz objects so they can be scaled.
    #             u_i = uxyz_pixel_ray_world.as_Vxyz()
    #             u_n = uxyz_normal_world.as_Vxyz()
    #             vxyz_mirror_to_screen = u_i - (u_n * (2 * (u_i.dot(u_n))))
    #             uxyz_mirror_to_screen = Uxyz(vxyz_mirror_to_screen.data)
    #             mirror_to_screen_xs1 = [vxyz_int_pt_world.x, vxyz_int_pt_world.x + (uxyz_mirror_to_screen.x * 1.2)]
    #             mirror_to_screen_ys1 = [vxyz_int_pt_world.y, vxyz_int_pt_world.y + (uxyz_mirror_to_screen.y * 1.2)]
    #             mirror_to_screen_zs1 = [vxyz_int_pt_world.z, vxyz_int_pt_world.z + (uxyz_mirror_to_screen.z * 1.2)]
    #             vxyz_mirror_to_screen1_world = Vxyz([mirror_to_screen_xs1, mirror_to_screen_ys1, mirror_to_screen_zs1])
    #             vxyz_mirror_to_screen1_world.draw_line(fig_rec, style=rcps.outline(color="blue"))
    #             # Draw reflection points on screen.
    #             vxyz_reflection_pt = sf2.propagate_rays_to_plane(
    #                 uxyz_mirror_to_screen, vxyz_int_pt_world, Vxyz([0, 1, 0]), (Uxyz([0, -1, 0]))
    #             )
    #             vxyz_reflection_pt.draw_line(fig_rec, style=rcps.marker(color="pink"))
    #             # Construct and draw reflected line.
    #             mirror_to_screen_xs = [vxyz_int_pt_world.x, vxyz_reflection_pt.x]
    #             mirror_to_screen_ys = [vxyz_int_pt_world.y, vxyz_reflection_pt.y]
    #             mirror_to_screen_zs = [vxyz_int_pt_world.z, vxyz_reflection_pt.z]
    #             vxyz_mirror_to_screen_world = Vxyz([mirror_to_screen_xs, mirror_to_screen_ys, mirror_to_screen_zs])
    #             vxyz_mirror_to_screen_world.draw_line(fig_rec, style=rcps.outline(color="magenta"))

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
        draw_mirror_centroid=False,
        draw_mirror_centroid_normal=False,
        mirror_needle_length=mirror_needle_length,
        axis_length=sofast_setup_axis_length,
    )

    # Plot rays
    if plot_camera_rays:
        draw_equation_slope_reflections_world(
            fig_rec=fig_rec,
            v_surf_pts_optic=v_facet_corners_hires_1,
            surface=surface,
            orientation=orientation,
            trans_camera_world=trans_camera_world,
            trans_mirror_world=trans_mirror_world,
        )
        draw_measured_slope_reflections_world(
            fig_rec=fig_rec,
            surface=surface,
            orientation=orientation,
            trans_camera_world=trans_camera_world,
            trans_mirror_world=trans_mirror_world,
            camera_ray_downsample=camera_ray_downsample,
            camera_ray_length=camera_ray_length,
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

    # Plot fit normal at align point
    v_fit = surface.normal_fit_at_align_point()
    v_fit_pt1 = surface.v_align_point_optic
    v_fit_pt2 = surface.v_align_point_optic + (v_fit.as_Vxyz() * mirror_needle_length * 1.5)
    fit_normal_in_place = v_fit_pt1.concatenate(v_fit_pt2)
    transformed_fit_normal = trans_mirror_world.apply(fit_normal_in_place)
    transformed_fit_normal.draw_line(fig_rec, style=rcps.outline(color='r'), label="Fit Normal")
    # Plot design normal at align point
    v_des = surface.normal_design_at_align_point()
    v_des_pt1 = surface.v_align_point_optic
    v_des_pt2 = surface.v_align_point_optic + (v_des.as_Vxyz() * mirror_needle_length * 0.5)
    design_normal_in_place = v_des_pt1.concatenate(v_des_pt2)
    transformed_design_normal = trans_mirror_world.apply(design_normal_in_place)
    transformed_design_normal.draw_line(fig_rec, style=rcps.outline(color='k', linestyle="--"), label="Design Normal")

    # Plot other points
    # Use draw_line(), because we only want one legend entry, not a legend entry for every point.
    align_point_world = trans_mirror_world.apply(surface.v_align_point_optic)
    align_point_world.draw_line(fig_rec, style=rcps.marker(marker='.', color='m', markersize=5), label="Align Point")

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


def draw_equation_slope_reflections_world(
    fig_rec: rcfg.RenderControlFigure,
    v_surf_pts_optic: Vxyz | None,
    surface: Surface2DAbstract,
    orientation: SpatialOrientation,
    trans_camera_world: txyz.TransformXYZ,
    trans_mirror_world: txyz.TransformXYZ,
):
    # &&&& DELETE-SCAFFOLDING -- THERE IS A LOT OF CODE REPLICATION AMONG THE REFLECTION-DRAWING ROUTINES.  CONSIDER MERGING.
    # Draw rays showing the camera-to-mirror-to-screen reflections.
    # Find intersection point closest to alignment point.
    # &&&& DELETE-SCAFFOLDING -- CONSIDER VECTORIZING THIS
    draw_equation_slope_reflections_world_aux(
        fig_rec,
        [0],
        surface.v_align_point_optic,  # A Vxyz can be either singleton or list.
        surface,
        orientation,
        trans_camera_world,
        trans_mirror_world,
        draw_camera_to_mirror=True,
        camera_to_mirror_style=rcps.outline(color='blue', linewidth=0.75),
        draw_normal=True,
        normal_style=rcps.outline(color='blue', linewidth=0.35, linestyle='--'),
        draw_mirror_to_screen=True,
        mirror_to_screen_style=rcps.outline(color='blue', linewidth=0.75),
        draw_screen_pts=True,
        screen_pts_style=rcps.marker(color='blue'),
    )
    # # Select sample points to draw.
    idx_list = []
    for idx in range(0, v_surf_pts_optic.len(), 3):
        idx_list.append(idx)
    # Draw reflections at selected points, including rays.
    draw_equation_slope_reflections_world_aux(
        fig_rec, idx_list, v_surf_pts_optic, surface, orientation, trans_camera_world, trans_mirror_world
    )
    # Select many sample points to draw points on screen.
    idx_list = []
    # We set range to same as measured slope points, to enable direct comparison.
    for idx in range(0, surface.v_surf_int_pts_optic.len(), 200):
        idx_list.append(idx)
    # Draw reflections at selected points, including rays.
    draw_equation_slope_reflections_world_aux(
        fig_rec,
        idx_list,
        surface.v_surf_int_pts_optic,
        surface,
        orientation,
        trans_camera_world,
        trans_mirror_world,
        draw_camera_to_mirror=False,
        draw_normal=False,
        draw_mirror_to_screen=False,
        draw_screen_pts=True,
        screen_pts_style=rcps.marker(color='lightblue', markersize=1.5),
    )


def draw_equation_slope_reflections_world_aux(
    fig_rec: rcfg.RenderControlFigure,
    idx_list: list[int],
    v_surf_pts_optic: Vxyz,
    surface: Surface2DAbstract,
    orientation: SpatialOrientation,
    trans_camera_world: txyz.TransformXYZ,
    trans_mirror_world: txyz.TransformXYZ,
    draw_camera_to_mirror: bool = True,
    camera_to_mirror_style: rcps.RenderControlPointSeq = rcps.outline(color='lightblue', linewidth=0.75),
    draw_normal: bool = True,
    normal_style: rcps.RenderControlPointSeq = rcps.outline(color='lightblue', linewidth=0.35, linestyle='--'),
    draw_mirror_to_screen: bool = True,
    mirror_to_screen_style: rcps.RenderControlPointSeq = rcps.outline(color='lightblue', linewidth=0.75),
    draw_screen_pts: bool = True,
    screen_pts_style: rcps.RenderControlPointSeq = rcps.marker(color='lightblue'),
):
    vxyz_cam_world = trans_camera_world.V
    for idx in idx_list:
        # Fetch needed data.
        vxyz_surf_pt_optic = v_surf_pts_optic[idx]
        # Convert to world coordinates.
        vxyz_surf_pt_world = trans_mirror_world.apply(vxyz_surf_pt_optic)
        # Construct and draw camera-to-mirror ray.
        vxyz_camera_to_mirror_world = vxyz_surf_pt_world - vxyz_cam_world
        uxyz_camera_to_mirror_world = Uxyz(vxyz_camera_to_mirror_world.data)
        if draw_camera_to_mirror:
            camera_to_mirror_xs = [vxyz_cam_world.x, vxyz_surf_pt_world.x]
            camera_to_mirror_ys = [vxyz_cam_world.y, vxyz_surf_pt_world.y]
            camera_to_mirror_zs = [vxyz_cam_world.z, vxyz_surf_pt_world.z]
            vxyz_camera_to_mirror_world = Vxyz([camera_to_mirror_xs, camera_to_mirror_ys, camera_to_mirror_zs])
            vxyz_camera_to_mirror_world.draw_line(fig_rec, style=camera_to_mirror_style)
        # Construct, check, and draw mirror surface normal.
        uxyz_normal_optic = surface.normal_fit_at_point(vxyz_surf_pt_optic)
        uxyz_normal_world = uxyz_normal_optic.rotate(trans_mirror_world.R)
        # Construct, check, and draw mirror-to-screen ray.
        # Given an incident vector i and a surface normal n, with normalized versions u_i and u_n,
        # respectively, then the formula for the unit-length reflected ray u_r is:
        #
        #    u_r = u_i - [2(u_i dot u_n) u_n]
        #
        # For details, see https://math.stackexchange.com/questions/13261/how-to-get-a-reflection-vector#:~:text=Reflection%20Vector%20Formula:%20The%20reflection%20vector%20(r),the%20formula%20r%20=%20d%20-%202(d%E2%8B%85n)n.
        #
        # Set u_i and u_n, as Vxyz objects so they can be scaled.
        u_i = uxyz_camera_to_mirror_world.as_Vxyz()
        u_n = uxyz_normal_world.as_Vxyz()
        vxyz_mirror_to_screen = u_i - (u_n * (2 * (u_i.dot(u_n))))
        uxyz_mirror_to_screen = Uxyz(vxyz_mirror_to_screen.data)
        if draw_normal:
            normal_world_xs = [vxyz_surf_pt_world.x, vxyz_surf_pt_world.x + (uxyz_normal_world.x * 0.2)]
            normal_world_ys = [vxyz_surf_pt_world.y, vxyz_surf_pt_world.y + (uxyz_normal_world.y * 0.2)]
            normal_world_zs = [vxyz_surf_pt_world.z, vxyz_surf_pt_world.z + (uxyz_normal_world.z * 0.2)]
            vxyz_normal_world = Vxyz([normal_world_xs, normal_world_ys, normal_world_zs])
            vxyz_normal_world.draw_line(fig_rec, style=normal_style)
        # Draw reflection points on screen.
        if draw_screen_pts:
            vxyz_reflection_pt = sf2.propagate_rays_to_plane(
                uxyz_mirror_to_screen, vxyz_surf_pt_world, Vxyz([0, 1, 0]), (Uxyz([0, -1, 0]))
            )
            vxyz_reflection_pt.draw_line(fig_rec, style=screen_pts_style)
        # Construct and draw reflected line.
        if draw_mirror_to_screen:
            mirror_to_screen_xs = [vxyz_surf_pt_world.x, vxyz_reflection_pt.x]
            mirror_to_screen_ys = [vxyz_surf_pt_world.y, vxyz_reflection_pt.y]
            mirror_to_screen_zs = [vxyz_surf_pt_world.z, vxyz_reflection_pt.z]
            vxyz_mirror_to_screen_world = Vxyz([mirror_to_screen_xs, mirror_to_screen_ys, mirror_to_screen_zs])
            vxyz_mirror_to_screen_world.draw_line(fig_rec, style=mirror_to_screen_style)


def draw_measured_slope_reflections_world(
    fig_rec: rcfg.RenderControlFigure,
    surface: Surface2DAbstract,
    orientation: SpatialOrientation,
    trans_camera_world: txyz.TransformXYZ,
    trans_mirror_world: txyz.TransformXYZ,
    camera_ray_downsample: int = 500,
    camera_ray_length: float = 0.0,  # meters
):
    # &&&& DELETE-SCAFFOLDING -- THERE IS A LOT OF CODE REPLICATION AMONG THE REFLECTION-DRAWING ROUTINES.  CONSIDER MERGING.
    if not hasattr(surface, 'slopes'):
        # Then simply draw the camera pixel pointing rays.
        u_active_pixel_pointing_optic_downsample = surface.u_active_pixel_pointing_optic[::camera_ray_downsample]
        u_active_pixel_pointing_cam = u_active_pixel_pointing_optic_downsample.rotate(orientation.r_cam_optic.inv())
        u_active_pixel_pointing_world = u_active_pixel_pointing_cam.rotate(trans_camera_world.R)
        for vxyz_ray in u_active_pixel_pointing_world:
            xs = [trans_camera_world.V.x, trans_camera_world.V.x + (vxyz_ray.x * camera_ray_length)]
            ys = [trans_camera_world.V.y, trans_camera_world.V.y + (vxyz_ray.y * camera_ray_length)]
            zs = [trans_camera_world.V.z, trans_camera_world.V.z + (vxyz_ray.z * camera_ray_length)]
            vxyz_ray = Vxyz([xs, ys, zs])
            vxyz_ray.draw_line(fig_rec, style=rcps.outline(color="pink"))  # Don't label -- too many rays
    else:
        # Draw rays showing the camera-to-mirror-to-screen reflections.
        # Find intersection point closest to alignment point.
        # &&&& DELETE-SCAFFOLDING -- CONSIDER VECTORIZING THIS
        align_point_world = trans_mirror_world.apply(surface.v_align_point_optic)
        v_surf_int_pts_world = trans_mirror_world.apply(surface.v_surf_int_pts_optic)
        n_pts = surface.v_surf_int_pts_optic.len()
        if n_pts > 0:
            min_dist_so_far = np.inf
            min_idx_so_far = -999
            for idx in range(n_pts):
                vxyz_int_pt_world = v_surf_int_pts_world[idx]
                dist = (vxyz_int_pt_world - align_point_world).magnitude()
                if dist < min_dist_so_far:
                    min_dist_so_far = dist
                    min_idx_so_far = idx
            align_idx_list = [min_idx_so_far]
            draw_measured_slope_reflections_world_aux(
                fig_rec,
                align_idx_list,
                surface.u_active_pixel_pointing_optic,
                surface.v_surf_int_pts_optic,
                surface.slopes,
                orientation,
                trans_camera_world,
                trans_mirror_world,
                draw_camera_to_mirror=True,
                camera_to_mirror_style=rcps.outline(color='magenta', linewidth=0.75),
                draw_normal=True,
                normal_style=rcps.outline(color='magenta', linewidth=0.35, linestyle='--'),
                draw_mirror_to_screen=True,
                mirror_to_screen_style=rcps.outline(color='magenta', linewidth=0.75),
                draw_screen_pts=True,
                screen_pts_style=rcps.marker(color='magenta'),
            )
        # Select sample points to draw.
        idx_list = []
        for idx in range(0, surface.v_surf_int_pts_optic.len(), 5000):
            idx_list.append(idx)
        # Draw reflections at selected points, including rays.
        draw_measured_slope_reflections_world_aux(
            fig_rec,
            idx_list,
            surface.u_active_pixel_pointing_optic,
            surface.v_surf_int_pts_optic,
            surface.slopes,
            orientation,
            trans_camera_world,
            trans_mirror_world,
        )
        # Select many sample points to draw points on screen.
        idx_list = []
        for idx in range(0, surface.v_surf_int_pts_optic.len(), 200):
            idx_list.append(idx)
        # Draw reflections at selected points, including rays.
        draw_measured_slope_reflections_world_aux(
            fig_rec,
            idx_list,
            surface.u_active_pixel_pointing_optic,
            surface.v_surf_int_pts_optic,
            surface.slopes,
            orientation,
            trans_camera_world,
            trans_mirror_world,
            draw_camera_to_mirror=False,
            draw_normal=False,
            draw_mirror_to_screen=False,
            draw_screen_pts=True,
            screen_pts_style=rcps.marker(color='pink', markersize=1.5),
        )


def draw_measured_slope_reflections_world_aux(
    fig_rec: rcfg.RenderControlFigure,
    idx_list: list[int],
    u_active_pixel_pointing_optic: Vxyz,
    v_surf_int_pts_optic: Vxyz,
    slopes_optic: np.ndarray,
    orientation: SpatialOrientation,
    trans_camera_world: txyz.TransformXYZ,
    trans_mirror_world: txyz.TransformXYZ,
    draw_camera_to_mirror: bool = True,
    camera_to_mirror_style: rcps.RenderControlPointSeq = rcps.outline(color='pink', linewidth=0.75),
    draw_normal: bool = True,
    normal_style: rcps.RenderControlPointSeq = rcps.outline(color='pink', linewidth=0.35, linestyle='--'),
    draw_mirror_to_screen: bool = True,
    mirror_to_screen_style: rcps.RenderControlPointSeq = rcps.outline(color='pink', linewidth=0.75),
    draw_screen_pts: bool = True,
    screen_pts_style: rcps.RenderControlPointSeq = rcps.marker(color='pink'),
):
    n_pixel_pointing = u_active_pixel_pointing_optic.len()
    n_intersect_pts = v_surf_int_pts_optic.len()
    n_slopes = slopes_optic.shape[1]
    if n_pixel_pointing != n_intersect_pts:
        lt.error_and_raise(
            ValueError,
            f"ERROR: In figure_intersection_surface_situation_world(), n_pixel_pointing={n_pixel_pointing} does not match n_intersect_pts={n_intersect_pts}",
        )
    if n_pixel_pointing != n_slopes:
        lt.error_and_raise(
            ValueError,
            f"ERROR: In figure_intersection_surface_situation_world(), n_pixel_pointing={n_pixel_pointing} does not match n_slopes={n_slopes}",
        )
    vxyz_cam_world = trans_camera_world.V
    for idx in idx_list:
        # Fetch needed data.
        uxyz_pixel_ray_optic = u_active_pixel_pointing_optic[idx]
        vxyz_int_pt_optic = v_surf_int_pts_optic[idx]
        slope_x = slopes_optic[0, idx]
        slope_y = slopes_optic[1, idx]
        # Convert to world coordinates.
        vxyz_int_pt_world = trans_mirror_world.apply(vxyz_int_pt_optic)
        uxyz_pixel_ray_cam = uxyz_pixel_ray_optic.rotate(orientation.r_cam_optic.inv())
        uxyz_pixel_ray_world = uxyz_pixel_ray_cam.rotate(trans_camera_world.R)
        # Construct and draw camera-to-mirror ray.
        if draw_camera_to_mirror:
            vxyz_camera_to_mirror = vxyz_int_pt_world - vxyz_cam_world
            # If the input vectors are both unit length, then the cross product equals
            # the sine of the angle.  But if the angle is small, this is simply the
            # angle, in radians.  Since we expect the angle to be very small, we'll
            # skip the arcsin() call here.
            uxyz_camera_to_mirror = vxyz_camera_to_mirror.normalize()
            comparison_angle = uxyz_pixel_ray_world.cross(uxyz_camera_to_mirror).magnitude()[0]
            if comparison_angle > 0.0001:  # 0.1 milliradian tolerance
                # &&&& DELETE-SCAFFOLDING -- TEMPORARY ERROR SHUTOFF
                # lt.error_and_raise(
                #     ValueError,
                lt.info(
                    f"ERROR: In figure_intersection_surface_situation_world(), comparison_angle={comparison_angle} is not near zero."
                )
            camera_to_mirror_xs = [vxyz_cam_world.x, vxyz_cam_world.x + vxyz_camera_to_mirror.x]
            camera_to_mirror_ys = [vxyz_cam_world.y, vxyz_cam_world.y + vxyz_camera_to_mirror.y]
            camera_to_mirror_zs = [vxyz_cam_world.z, vxyz_cam_world.z + vxyz_camera_to_mirror.z]
            vxyz_camera_to_mirror_world = Vxyz([camera_to_mirror_xs, camera_to_mirror_ys, camera_to_mirror_zs])
            vxyz_camera_to_mirror_world.draw_line(fig_rec, style=camera_to_mirror_style)
        # Construct, check, and draw mirror surface normal.
        uxyz_normal_optic = Uxyz([-slope_x, -slope_y, 1])
        uxyz_normal_world = uxyz_normal_optic.rotate(trans_mirror_world.R)
        # Construct, check, and draw mirror-to-screen ray.
        # Given an incident vector i and a surface normal n, with normalized versions u_i and u_n,
        # respectively, then the formula for the unit-length reflected ray u_r is:
        #
        #    u_r = u_i - [2(u_i dot u_n) u_n]
        #
        # For details, see https://math.stackexchange.com/questions/13261/how-to-get-a-reflection-vector#:~:text=Reflection%20Vector%20Formula:%20The%20reflection%20vector%20(r),the%20formula%20r%20=%20d%20-%202(d%E2%8B%85n)n.
        #
        # Set u_i and u_n, as Vxyz objects so they can be scaled.
        u_i = uxyz_pixel_ray_world.as_Vxyz()
        u_n = uxyz_normal_world.as_Vxyz()
        vxyz_mirror_to_screen = u_i - (u_n * (2 * (u_i.dot(u_n))))
        uxyz_mirror_to_screen = Uxyz(vxyz_mirror_to_screen.data)
        if draw_normal:
            normal_world_xs = [vxyz_int_pt_world.x, vxyz_int_pt_world.x + (uxyz_normal_world.x * 0.2)]
            normal_world_ys = [vxyz_int_pt_world.y, vxyz_int_pt_world.y + (uxyz_normal_world.y * 0.2)]
            normal_world_zs = [vxyz_int_pt_world.z, vxyz_int_pt_world.z + (uxyz_normal_world.z * 0.2)]
            vxyz_normal_world = Vxyz([normal_world_xs, normal_world_ys, normal_world_zs])
            vxyz_normal_world.draw_line(fig_rec, style=normal_style)
        # Draw reflection points on screen.
        if draw_screen_pts:
            vxyz_reflection_pt = sf2.propagate_rays_to_plane(
                uxyz_mirror_to_screen, vxyz_int_pt_world, Vxyz([0, 1, 0]), (Uxyz([0, -1, 0]))
            )
            vxyz_reflection_pt.draw_line(fig_rec, style=screen_pts_style)
        # Construct and draw reflected line.
        if draw_mirror_to_screen:
            mirror_to_screen_xs = [vxyz_int_pt_world.x, vxyz_reflection_pt.x]
            mirror_to_screen_ys = [vxyz_int_pt_world.y, vxyz_reflection_pt.y]
            mirror_to_screen_zs = [vxyz_int_pt_world.z, vxyz_reflection_pt.z]
            vxyz_mirror_to_screen_world = Vxyz([mirror_to_screen_xs, mirror_to_screen_ys, mirror_to_screen_zs])
            vxyz_mirror_to_screen_world.draw_line(fig_rec, style=mirror_to_screen_style)


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
    full_title = f"loop={loop_idx:d}: " + title
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
    v_fit_pt2 = surface.v_align_point_optic + (v_fit.as_Vxyz() * mirror_needle_length * 1.5)
    fit_normal_in_place = v_fit_pt1.concatenate(v_fit_pt2)
    fit_normal_in_place.draw_line(fig_rec, style=rcps.outline(color='r'), label="Fit Normal")
    # Plot design normal at align point
    v_des = surface.normal_design_at_align_point()
    v_des_pt1 = surface.v_align_point_optic
    v_des_pt2 = surface.v_align_point_optic + (v_des.as_Vxyz() * mirror_needle_length)
    design_normal_in_place = v_des_pt1.concatenate(v_des_pt2)
    design_normal_in_place.draw_line(fig_rec, style=rcps.outline(color='m', linestyle="--"), label="Design Normal")

    # Plot other points
    # Use draw_line(), because we only want one legend entry, not a legend entry for every point.
    # &&&& DELETE-SCAFFOLDING -- MARKERSIZE WAS 3, COLOR WAS 'cyan'
    surface.v_align_point_optic.draw_line(
        fig_rec, style=rcps.marker(marker='.', color='m', markersize=5), label="Align Point"
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
    pts_reproj: Vxy, hires_pts_reproj_1: Vxy, hires_pts_reproj_snap_1: Vxy, loop_idx: int, debug: SlopeSolverDataDebug
) -> None:
    reproj_snap_aux(
        pts_reproj, hires_pts_reproj_1, hires_pts_reproj_snap_1, label_points=True, loop_idx=loop_idx, debug=debug
    )
    reproj_snap_aux(None, hires_pts_reproj_1, None, label_points=False, loop_idx=loop_idx, debug=debug)
    reproj_snap_aux(None, None, hires_pts_reproj_snap_1, label_points=False, loop_idx=loop_idx, debug=debug)


def reproj_snap_aux(
    pts_reproj: Vxy | None,
    hires_pts_reproj_1: Vxy | None,
    hires_pts_reproj_snap_1: Vxy | None,
    label_points: bool,
    loop_idx: int,
    debug: SlopeSolverDataDebug,
) -> None:
    figure_title = f"loop={loop_idx:d}: " + "Reprojected Points, and Snap to Edges"
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

    # Save and close.
    full_title_for_file = (
        figure_title.replace(' ', '_').replace(':', '').replace('(', '').replace(')', '').replace(',', '')
    )
    sdfs.finish_debug_image_figure(full_title_for_file, 'solver', fig_rec, debug.debug_geometry)


# INITIAL REPROJECTION SUMMARY, AFTER SNAP TO EDGE (WITHOUT COARSE VERTICES)


# &&&& DELETE-SCAFFOLDING -- CHOOSE A BETTER NAME FOR THIS ROUTINE?
def reproj_snap_2(
    hires_pts_reproj_1: Vxy, hires_pts_reproj_snap_1: Vxy, loop_idx: int, debug: SlopeSolverDataDebug
) -> None:
    reproj_snap_2_aux(hires_pts_reproj_1, hires_pts_reproj_snap_1, label_points=True, loop_idx=loop_idx, debug=debug)
    reproj_snap_2_aux(hires_pts_reproj_1, None, label_points=False, loop_idx=loop_idx, debug=debug)
    reproj_snap_2_aux(None, hires_pts_reproj_snap_1, label_points=False, loop_idx=loop_idx, debug=debug)


def reproj_snap_2_aux(
    hires_pts_reproj_1: Vxy | None,
    hires_pts_reproj_snap_1: Vxy | None,
    label_points: bool,
    loop_idx: int,
    debug: SlopeSolverDataDebug,
) -> None:
    figure_title = f"loop={loop_idx:d}: " + "Reprojected Points, and Snap to Edges"
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

    # Save and close.
    full_title_for_file = (
        figure_title.replace(' ', '_').replace(':', '').replace('(', '').replace(')', '').replace(',', '')
    )
    sdfs.finish_debug_image_figure(full_title_for_file, 'solver', fig_rec, debug.debug_geometry)


# INITIAL REPROJECTION SUMMARY, DURING CAMERA POSE (DX,DY,DTHETA) SEARCH


# &&&& DELETE-SCAFFOLDING -- CHOOSE A BETTER NAME FOR THIS ROUTINE?
def reproj_snap_3(
    hires_pts_reproj_1: Vxy,
    hires_pts_reproj_snap_1: Vxy,
    status_str: str | None,
    camera_dx_dy_dtheta: tuple[float, float, float],
    rms: float,
    debug: SlopeSolverDataDebug,
) -> None:
    reproj_snap_3_aux(
        hires_pts_reproj_1,
        hires_pts_reproj_snap_1,
        label_points=True,
        status_str=status_str,
        camera_dx_dy_dtheta=camera_dx_dy_dtheta,
        rms=rms,
        debug=debug,
    )
    # reproj_snap_3_aux(
    #     hires_pts_reproj_1, None, label_points=False, camera_dx_dy_dtheta=camera_dx_dy_dtheta, rms=rms, debug=debug
    # )
    # reproj_snap_3_aux(
    #     None, hires_pts_reproj_snap_1, label_points=False, camera_dx_dy_dtheta=camera_dx_dy_dtheta, rms=rms, debug=debug
    # )


def reproj_snap_3_aux(
    hires_pts_reproj_1: Vxy | None,
    hires_pts_reproj_snap_1: Vxy | None,
    label_points: bool,
    status_str: str | None,
    camera_dx_dy_dtheta: tuple[float, float, float],
    rms: float,
    debug: SlopeSolverDataDebug,
) -> None:
    dx = camera_dx_dy_dtheta[0]
    dy = camera_dx_dy_dtheta[1]
    dtheta = camera_dx_dy_dtheta[2]
    dx_dy_dtheta_str = f"({dx:.3f}m, {dy:.3f}m, {np.degrees(dtheta):.2f}deg)"
    figure_title = f"dx_dy_dth={dx_dy_dtheta_str}, rms={rms:.4f}pix"
    if status_str is not None:
        figure_title = status_str + figure_title
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

    # Save and close.
    full_title_for_file = (
        figure_title.replace(' ', '_').replace(':', '').replace('(', '').replace(')', '').replace(',', '')
    )
    sdfs.finish_debug_image_figure(full_title_for_file, 'solver', fig_rec, debug.debug_geometry)


# REPROJECTION OF SNAP TO EDGE, SNAP TO SURFACE REPROJECTION


def reproj_after_snap_snap(
    hires_pts_reproj_snap_1: Vxy, hires_pts_reproj_3: Vxy, loop_idx: int, debug: SlopeSolverDataDebug
) -> None:
    reproj_after_snap_snap_aux(
        hires_pts_reproj_snap_1, hires_pts_reproj_3, label_points=True, loop_idx=loop_idx, debug=debug
    )
    reproj_after_snap_snap_aux(
        hires_pts_reproj_snap_1, hires_pts_reproj_3, label_points=False, loop_idx=loop_idx, debug=debug
    )


def reproj_after_snap_snap_aux(
    hires_pts_reproj_snap_1: Vxy,
    hires_pts_reproj_3: Vxy,
    label_points: bool,
    loop_idx: int,
    debug: SlopeSolverDataDebug,
) -> None:
    figure_title = f"loop={loop_idx:d}: " + "Snap-to-Edge Points, Snapped to Surface and Reprojected"
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

    # Save and close.
    full_title_for_file = (
        figure_title.replace(' ', '_').replace(':', '').replace('(', '').replace(')', '').replace(',', '')
    )
    sdfs.finish_debug_image_figure(full_title_for_file, 'solver', fig_rec, debug.debug_geometry)


# REPROJECTION OF SNAP TO EDGE, SNAP TO SURFACE REPROJECTION


def reproj_after_fit(
    hires_pts_reproj_snap_1: Vxy,
    hires_pts_reproj_3: Vxy,
    hires_pts_reproj_4: Vxy,
    loop_idx: int,
    debug: SlopeSolverDataDebug,
) -> None:
    reproj_after_fit_aux(
        hires_pts_reproj_snap_1, hires_pts_reproj_3, hires_pts_reproj_4, loop_idx=loop_idx, debug=debug
    )


def reproj_after_fit_aux(
    hires_pts_reproj_snap_1: Vxy,
    hires_pts_reproj_3: Vxy,
    hires_pts_reproj_4: Vxy,
    loop_idx: int,
    debug: SlopeSolverDataDebug,
) -> None:
    figure_title = f"loop={loop_idx:d}: " + "After Fitting Surface to Slopes, Snapping to Surface, Reprojecting"
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

    # Save and close.
    full_title_for_file = (
        figure_title.replace(' ', '_').replace(':', '').replace('(', '').replace(')', '').replace(',', '')
    )
    sdfs.finish_debug_image_figure(full_title_for_file, 'solver', fig_rec, debug.debug_geometry)
