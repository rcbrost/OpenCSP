import copy
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

import opencsp.app.sofast.lib.image_processing as ip
import opencsp.app.sofast.lib.sofast_debug_figure_support as sdfs
import opencsp.app.sofast.lib.spatial_processing as sp
from opencsp.app.sofast.lib.SpatialOrientation import SpatialOrientation
import opencsp.common.lib.deflectometry.SlopeSolver_debug_output as ssdo
from opencsp.common.lib.deflectometry.SlopeSolverDataDebug import SlopeSolverDataDebug
from opencsp.common.lib.deflectometry.SlopeSolverData import SlopeSolverData
import opencsp.common.lib.deflectometry.slope_fitting_2d as sf2
from opencsp.common.lib.deflectometry.Surface2DAbstract import Surface2DAbstract
from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.geometry.TransformXYZ import TransformXYZ
import opencsp.common.lib.render.view_spec as vs
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.tool.log_tools as lt


class SlopeSolver:
    """Class that solves for the surface slopes of optics in deflectometry
    systems."""

    def __init__(
        self,
        v_optic_cam_optic: Vxyz,
        u_active_pixel_pointing_optic: Uxyz,
        u_measure_pixel_pointing_optic: Uxyz,
        v_screen_points_screen: Vxyz,
        v_screen_points_facet: Vxyz,
        v_optic_screen_optic: Vxyz,
        v_align_point_optic: Vxyz,
        dist_optic_screen: float,
        surface: Surface2DAbstract,  # &&&& DELETE-SCAFFOLDING -- Surface2dParabolic INSTEAD?
        debug: SlopeSolverDataDebug = SlopeSolverDataDebug(),
    ) -> "SlopeSolver":
        """
        Initializes the slope solving object.

        Parameters
        ----------
        v_optic_cam_optic : Vxyz
            Optic to camera vector in optic coordinates.
        u_active_pixel_pointing_optic : Uxyz
            Active pixel pointing directions in optic coordinates.
        u_measure_pixel_pointing_optic : Uxyz
            Measure pixel pointing direction in optic cooridinates.
        v_screen_points_screen : Vxyz
            Positions of screen points in screen coordinates.
        v_screen_points_facet : Vxyz
            Positions of screen points in optic coordinates.
        v_optic_screen_optic : Vxyz
            Optic to screen vector in optic coordinates.
        v_align_point_optic : Vxyz
            Position of align point in optic coordinates.
        dist_optic_screen : float
            Measured optic to screen distance.
        surface : Surface2DAbstract
            2D surface definition class.
        debug: SlopeSolverDataDebug
            SlopeSolverDataDebug object for debugging.
        """
        # Store inputs in class
        self.surface = surface
        self.v_optic_cam_optic = v_optic_cam_optic
        self.u_active_pixel_pointing_optic = u_active_pixel_pointing_optic
        self.u_measure_pixel_pointing_optic = u_measure_pixel_pointing_optic
        self.v_screen_points_screen = v_screen_points_screen
        self.v_screen_points_facet = v_screen_points_facet
        self.v_optic_screen_optic = v_optic_screen_optic
        self.v_align_point_optic = v_align_point_optic
        self.dist_optic_screen = dist_optic_screen
        self.debug = debug

        # Load initialization data in surface fit object
        self.surface.set_spatial_data(
            u_active_pixel_pointing_optic,
            v_screen_points_facet,  # &&&& DELETE-SCAFFOLDING -- THIS DOESN'T MATCH CALLED ROUTINE NAME!
            v_optic_cam_optic,
            u_measure_pixel_pointing_optic,
            v_align_point_optic,
            v_optic_screen_optic,
        )

        self._data = SlopeSolverData()

    def get_data(self) -> SlopeSolverData:
        """Returns data output object"""
        return self._data

    # def fit_surface(self) -> None:
    #     """
    #     Performs the initial fine-tuning alignment of the facet, screen, and
    #     camera. Fits a surface to the calculated slope data.

    #     """
    #     # Gather inputs
    #     v_optic_cam_optic = self.v_optic_cam_optic
    #     u_measure_pixel_pointing_optic = self.u_measure_pixel_pointing_optic
    #     v_optic_screen_optic = self.v_optic_screen_optic
    #     v_align_point_optic = self.v_align_point_optic
    #     dist_optic_screen = self.dist_optic_screen

    #     # Instantiate alignment transform
    #     trans_align = TransformXYZ.from_zero_zero()

    #     # &&&& DELETE-SCAFFOLDING -- TEMPORARY PASS-THROUGH HACK
    #     for idx1 in range(4):
    #         # for idx1 in range(1):  # &&&& DELETE-SCAFFOLDING -- TEMPORARY PASS-THROUGH HACK
    #         # &&&& DELETE-SCAFFOLDING -- TEMPORARY PASS-THROUGH HACK
    #         for idx2 in range(3):
    #             # for idx2 in range(1):  # &&&& DELETE-SCAFFOLDING -- TEMPORARY PASS-THROUGH HACK
    #             # Calculate surface intersection points
    #             self.surface.calculate_surface_intersect_points()

    #             # Check for invalid points
    #             num_nans = np.isnan(self.surface.v_surf_int_pts_optic.data)
    #             if np.any(num_nans):
    #                 warnings.warn(
    #                     f"{num_nans.sum():d} / {num_nans.size:d} values are NANs in surface intersection points in iteration: ({idx1:d}, {idx2:d}).",
    #                     stacklevel=2,
    #                 )

    #             # Calculate measurement point slopes
    #             self.surface.calculate_slopes()

    #             # Check for invalid points
    #             num_nans = np.isnan(self.surface.slopes)
    #             if np.any(num_nans):
    #                 warnings.warn(
    #                     f"{num_nans.sum():d} / {num_nans.size:d} values are NANs in slope data in iteration: ({idx1:d}, {idx2:d}).",
    #                     stacklevel=2,
    #                 )

    #             # # Plot debug plot
    #             # if self.debug.debug_active:
    #             #     self._plot_debug_plots("Before Slope Fit", idx1, idx2)
    #             #     # self._plot_debug_plots("Before Slope Fit", idx1, idx2, az_el_roll_deg=(0, 0, 0))
    #             #     # self._plot_debug_plots("Before Slope Fit", idx1, idx2, az_el_roll_deg=(-90, 0, 0))
    #             #     # self._plot_debug_plots("Before Slope Fit", idx1, idx2, az_el_roll_deg=(0, 90, 90))

    #             # Update slope fit
    #             self.surface.fit_slopes()

    #             # Plot debug plot
    #             if self.debug.debug_active:
    #                 self._plot_debug_plots("After Slope Fit", idx1, idx2)
    #                 self._plot_debug_plots("After Slope Fit", idx1, idx2, az_el_roll_deg=(0, 0, 0))
    #                 self._plot_debug_plots("After Slope Fit", idx1, idx2, az_el_roll_deg=(-90, 0, 0))
    #                 self._plot_debug_plots("After Slope Fit", idx1, idx2, az_el_roll_deg=(0, 90, 90))

    #         # Calculate measure point intersection point with existing fitting function
    #         v_meas_pts_surf_int_optic = self.surface.intersect(u_measure_pixel_pointing_optic, v_optic_cam_optic)

    #         # Calculate design normal at alignment point
    #         n_design = self.surface.normal_design_at_align_point()

    #         # Calculate measured normal at alignment point
    #         n_meas = self.surface.normal_fit_at_align_point()

    #         # Calculate the rotation needed to align the normal vectors
    #         r_align_step = n_meas.align_to(n_design)

    #         # Rotate all points about alignment point
    #         self.surface.rotate_all(r_align_step)

    #         # # Plot debug plot
    #         # if self.debug.debug_active:
    #         #     self._plot_debug_plots("After Rotate All", idx1, idx2)
    #         #     self._plot_debug_plots("After Rotate All", idx1, idx2, az_el_roll_deg=(0, 0, 0))
    #         #     self._plot_debug_plots("After Rotate All", idx1, idx2, az_el_roll_deg=(-90, 0, 0))
    #         #     self._plot_debug_plots("After Rotate All", idx1, idx2, az_el_roll_deg=(0, 90, 90))

    #         # Calculate scale so that align-point to screen matches measurement
    #         args = (
    #             dist_optic_screen,
    #             v_align_point_optic,
    #             v_optic_cam_optic,
    #             v_optic_screen_optic,
    #             v_meas_pts_surf_int_optic,
    #         )
    #         out = minimize(sf2.dist_optic_screen_error, np.array([1.0]), args=args)
    #         scale = out.x[0]
    #         v_align_optic_step = (v_optic_cam_optic - v_align_point_optic) * (scale - 1)

    #         # Shift all points along align-point to camera axis
    #         self.surface.shift_all(v_align_optic_step)

    #         # # Plot debug plot
    #         # if self.debug.debug_active:
    #         #     self._plot_debug_plots("After Shift All", idx1, idx2)
    #         #     # self._plot_debug_plots("After Shift All", idx1, idx2, az_el_roll_deg=(0, 0, 0))
    #         #     # self._plot_debug_plots("After Shift All", idx1, idx2, az_el_roll_deg=(-90, 0, 0))
    #         #     # self._plot_debug_plots("After Shift All", idx1, idx2, az_el_roll_deg=(0, 90, 90))

    #         # Calculate alignment transform
    #         trans_step = TransformXYZ.from_R_V(r_align_step, v_align_optic_step)
    #         trans_align = trans_step * trans_align

    #     # Store alignment parameters
    #     self._data.surf_coefs_facet = self.surface.surf_coefs
    #     self._data.slope_coefs_facet = self.surface.slope_coefs
    #     # &&&& DELETE-SCAFFOLDING -- TEMPORARY PASS-THROUGH HACK
    #     self._data.trans_alignment = trans_align
    #     # self._data.trans_alignment = TransformXYZ.identity()  # &&&& DELETE-SCAFFOLDING -- TEMPORARY PASS-THROUGH HACK

    def solve_slopes(self) -> None:
        """
        Solves the surface slopes of the optic using camera position
        and alignment transform from self.fit_surface.

        Raises
        ------
        Exception
            Raises ValueError if initial alignment has not been performed
            prior to calling this function (self.fit_surface()).

        """
        # Check alignment has been completed
        if self._data.trans_alignment is None:
            raise ValueError("Initial alignment needs to be completed before final slope fitting (self.solve_slopes).")

        # Apply alignment transforms about alignment point
        # &&&& DELETE-SCAFFOLDING -- THERE IS self.v_align_point_optic AND self.surface.v_align_point_optic.  WHY BOTH?  IS THERE A DIFFERENCE?
        # &&&& DELETE-SCAFFOLDING -- AT SOME POINT, ELIMINATE THIS REDUNDANCY/AMBIGUITY.
        trans_shift_1 = TransformXYZ.from_V(-self.surface.v_align_point_optic)
        trans_shift_2 = TransformXYZ.from_V(self.surface.v_align_point_optic)
        trans: TransformXYZ = trans_shift_2 * self._data.trans_alignment * trans_shift_1

        u_active_pixel_pointing_optic = self.u_active_pixel_pointing_optic.rotate(trans.R)
        v_screen_points_facet = trans.apply(self.v_screen_points_facet)

        # Calculate intersection points on optic surface
        v_surf_points_facet = self.surface.intersect(u_active_pixel_pointing_optic, self.surface.v_optic_cam_optic)

        # Calculate pixel slopes (assuming parabolic surface intersection)
        slopes_facet_xy = sf2.calc_slopes(v_surf_points_facet, self.surface.v_optic_cam_optic, v_screen_points_facet)

        self._data.v_surf_points_facet = v_surf_points_facet
        self._data.slopes_facet_xy = slopes_facet_xy

    def _plot_debug_plots(self, title: str, idx1: int, idx2: int, az_el_roll_deg: tuple[float, float, float] = None):
        # Create figure and axes
        if self.debug.slope_solver_single_plot and isinstance(self.debug.slope_solver_figures, list):
            # Create first figure if needed
            fig = plt.figure()
            axes = fig.add_subplot(projection="3d")
            self.debug.slope_solver_figures = fig
            # Plot facet corners
            facet_outline = self.debug.optic_data.v_facet_corners.data
            axes.scatter(*facet_outline, color="k")
            # Format
            axes.set_title("Slope Solver: " + title)
        elif self.debug.slope_solver_single_plot:
            # Get axes for single plot
            axes = self.debug.slope_solver_figures.gca()
        else:
            # Create a new figure
            fig = plt.figure(figsize=(12, 9))
            axes = fig.add_subplot(projection="3d", proj_type='ortho')
            self.debug.slope_solver_figures.append(fig)
            # Plot facet corners
            facet_outline = self.debug.optic_data.v_facet_corners.data
            axes.scatter(
                *facet_outline, color="lightgreen", label="Facet Vertices"
            )  # &&&& DELETE-SCAFFOLDING -- COLOR WAS "k"
            # Plot high-resolution facet corners
            facet_outline = self.debug.optic_data.v_facet_corners_hires.data
            axes.scatter(*facet_outline, s=10, color="blue", label="Facet Vertices (High Resolution)")
            # Format
            axes.set_title(f"Slope Solver ({idx1:d}, {idx2:d}): " + title)

        # Set view direction, if desired.
        if az_el_roll_deg is not None:
            azimuth_deg = az_el_roll_deg[0]
            elevation_deg = az_el_roll_deg[1]
            roll_deg = az_el_roll_deg[2]
            axes.view_init(azim=azimuth_deg, elev=elevation_deg, roll=roll_deg)

        # Plot intersection points
        self.surface.plot_intersection_points(
            axes,
            self.debug.slope_solver_point_downsample,
            self.debug.slope_solver_camera_rays_length,
            self.debug.slope_solver_plot_camera_screen_points,
        )

        # Add legend
        plt.legend()  # &&&& DELETE-SCAFFOLDING -- NEW.  KEEP?

    # # &&&& DELETE-SCAFFOLDING -- BEGIN NEW GEN 2 SOLVER ALGORITHM

    # def fit_surface_2(self, original_orientation: SpatialOrientation) -> None:
    #     """
    #     # &&&& DELETE-SCAFFOLDING -- FIX THIS DOCSTRING
    #     Performs the initial fine-tuning alignment of the facet, screen, and
    #     camera. Fits a surface to the calculated slope data.

    #     """
    #     # &&&& DELETE-SCAFFOLDING -- PASS THIS IN
    #     vxyz_corners_hires_0 = self.debug.debug_geometry.v_facet_corners_hires
    #     # &&&& DELETE-SCAFFOLDING -- PASS THIS IN
    #     camera = self.debug.debug_geometry.camera
    #     # &&&& DELETE-SCAFFOLDING -- PASS THIS IN
    #     mask_processed = self.debug.debug_geometry.mask_processed

    #     # # &&&& DELETE-SCAFFOLDING -- DELETE ONCE CLEAR NOT NEEDED
    #     # # Save initial camera pose for later comparison.
    #     # r_cam_optic_original = self.debug.debug_geometry.r_cam_optic_refine_1
    #     # v_cam_optic_cam_original = self.debug.debug_geometry.v_cam_optic_cam_refine_2

    #     # &&&& DELETE-SCAFFOLDING -- MOVE OUTSIDE LOOP?
    #     # Calculate pixel pointing directions (camera coordinates)
    #     u_pixel_pointing_cam = ip.calculate_active_pixels_vectors(mask_processed, camera)

    #     # Construct search spiral for snapping boundary points to nearest edge.
    #     search_spiral = ip.construct_search_spiral(max_radius=50)
    #     # &&&& DELETE-SCAFFOLDING -- TEMPORARY
    #     print('In fit_surface_2(), spiral complete.  Spiral length=', len(search_spiral))

    #     # Iteratively serach for a compatible solution:
    #     #    Stable inputs:
    #     #        Facet vertices V defining the facet boundary on the (x,y) plane.
    #     #        Camera model
    #     #        Mask image
    #     #        Pixel (x,y,z) reflection points RF on screen/target.
    #     #    Begin with:
    #     #        (a) Hypothesized surface equation COEFFS = c0, c1x, c2x2, c3y, c4xy, c5y2.
    #     #                z = c0 + c1x*x + c2x2*x^2 + c3y*y + c4xy*x*y + c5y2*y^2.
    #     #        (b) Initial camera pose estimate POSE = (R_cam,T_cam).
    #     #    Repeat:
    #     #        1. Project facet boundary vertices along z onto surface equation COEFFS ==> 3-d facet vertices V'.
    #     #        2. Use camera model and current POSE estimate to project 3-d vertices onto image.
    #     #        3. Refine projected image points by snapping onto edges in mask image.
    #     #        4. Use refined image points to call solvePnP() and compute a refined camera POSE'.
    #     #        5. Using POSE' project rays from camera to COEFFS surface, finding intersection points INT.
    #     #        6. Use intersection points and reflection points RF to compute surface normals at INT points.
    #     #        7. Using surface normals at points, compute regression fit for slope coefficients.
    #     #        8. Convert fit slope coefficients to new surface COEFFS' = c0', c1x', c2x2', c3y', c4xy', c5y2'.
    #     #    Until new COEFFS' agree with original COEFFS, up to a tolerance.

    #     # Initial values.
    #     loop_idx = 0
    #     vxyz_corners_entering_loop = vxyz_corners_hires_0
    #     r_cam_optic_entering_loop = self.debug.debug_geometry.r_cam_optic_refine_1
    #     v_cam_optic_cam_entering_loop = self.debug.debug_geometry.v_cam_optic_cam_refine_2

    #     # Keep track of loop progress.
    #     loop_record_0 = {
    #         "loop_idx": loop_idx,
    #         "surf_coefs": self.surface.surf_coefs,
    #         "slope_coefs": self.surface.slope_coefs,
    #         "r_cam_optic": r_cam_optic_entering_loop,
    #         "v_cam_optic_cam": v_cam_optic_cam_entering_loop,
    #         "vxyz_corners": vxyz_corners_entering_loop,
    #         "vxyz_corner_change": (vxyz_corners_entering_loop - vxyz_corners_entering_loop),  # Zero change.
    #         "n_intersect": 0,
    #         "u_avg_fit_normal": Uxyz([0, 0, 1]),
    #         "u_avg_measured_normal": Uxyz([0, 0, 1]),
    #         "u_avg_measured_minus_fit": Uxyz([0, 0, 1]),
    #         "r_align_step": None,
    #     }
    #     loop_record_list = [loop_record_0]

    #     # Main loop.
    #     while True:
    #         loop_idx += 1
    #         loop_record = {"loop_idx": loop_idx}
    #         loop_record_list.append(loop_record)

    #         # 1. Project facet boundary vertices along z onto surface equation COEFFS ==> 3-d facet vertices V'.
    #         z_corners_sfc = sf2.coef_to_points(vxyz_corners_entering_loop, self.surface.surf_coefs, 2)
    #         vxyz_corners_sfc = copy.deepcopy(vxyz_corners_entering_loop)
    #         vxyz_corners_sfc.data[2, :] = z_corners_sfc
    #         # Note how far the vertices moved.
    #         vxyz_corners_change = vxyz_corners_sfc - vxyz_corners_entering_loop
    #         loop_record["vxyz_corners"] = vxyz_corners_sfc
    #         loop_record["vxyz_corner_change"] = vxyz_corners_change

    #         # 2. Use camera model and current POSE estimate to project 3-d vertices onto image.
    #         vxy_corners_sfc_reproj = self.debug.debug_geometry.camera.project(
    #             vxyz_corners_sfc, r_cam_optic_entering_loop.inv(), v_cam_optic_cam_entering_loop
    #         )

    #         # 3. Refine projected image points by snapping onto edges in mask image.
    #         sfc_pt_reproj_pt_snapped_pt_list = ip.snap_points_to_nearest_edge(
    #             vxyz_corners_sfc, vxy_corners_sfc_reproj, mask_processed, search_spiral
    #         )
    #         vxyz_corners_sfc_matched = Vxyz.from_list([a[0] for a in sfc_pt_reproj_pt_snapped_pt_list])
    #         vxy_corners_sfc_reproj_matched = Vxy.from_list([a[1] for a in sfc_pt_reproj_pt_snapped_pt_list])
    #         vxy_corners_sfc_reproj_snap = Vxy.from_list([a[2] for a in sfc_pt_reproj_pt_snapped_pt_list])
    #         # Plot reprojected points over mask image
    #         if self.debug.debug_active:
    #             ssdo.reproj_snap_2(
    #                 vxy_corners_sfc_reproj_matched, vxy_corners_sfc_reproj_snap, loop_idx=loop_idx, debug=self.debug
    #             )

    #         # 4. Use refined image points to call solvePnP() and compute a refined camera POSE'.
    #         r_optic_cam_new, v_cam_optic_cam_new = sp.calc_rt_from_img_pts(
    #             vxy_corners_sfc_reproj_snap,
    #             vxyz_corners_sfc_matched,
    #             camera,
    #             initial_rotation=r_cam_optic_entering_loop,
    #             initial_vxyz=v_cam_optic_cam_entering_loop,
    #         )
    #         # &&&& DELETE-SCAFFOLDING -- SHOULD THIS NAME BE INVERTED?
    #         r_cam_optic_new = r_optic_cam_new.inv()
    #         # Orient optic
    #         # &&&& DELETE-SCAFFOLDING -- TEMPORARY, OR DOCUMENT
    #         if loop_idx == 1:  # loop_idx = 1 first pass through lopp.
    #             ori_new = copy.copy(original_orientation)
    #             ori_new.orient_optic_cam(r_cam_optic_new, v_cam_optic_cam_new)
    #         loop_record["r_cam_optic"] = ori_new.r_cam_optic
    #         loop_record["v_cam_optic_cam"] = ori_new.v_cam_optic_cam
    #         # Plot reprojected points over mask image
    #         # &&&& DELETE-SCAFFOLDING -- MOVE REPROJECTION INTO DEBUG FIGURE ROUTINE
    #         vxy_corners_sfc_reproj_new = self.debug.debug_geometry.camera.project(
    #             vxyz_corners_sfc, ori_new.r_cam_optic.inv(), ori_new.v_cam_optic_cam
    #         )
    #         if self.debug.debug_active:
    #             ssdo.reproj_after_snap_snap(
    #                 vxy_corners_sfc_reproj_snap, vxy_corners_sfc_reproj_new, loop_idx, self.debug
    #             )

    #         # 4b. Update cached values in surface to match new camera POSE'.
    #         #     From Surface2DParabolic.set_spatial_data()
    #         #         # Downsample and save measurement data
    #         #         self.u_active_pixel_pointing_optic = u_active_pixel_pointing_optic[:: self.downsample]
    #         #         self.v_screen_points_optic = v_screen_points_optic[:: self.downsample]
    #         #         # Save position data
    #         #         self.v_optic_cam_optic = v_optic_cam_optic
    #         #         self.u_measure_pixel_pointing_optic = u_measure_pixel_pointing_optic
    #         #         self.v_align_point_optic = v_align_point_optic
    #         #         self.v_optic_screen_optic = v_optic_screen_optic
    #         #         Convert pixel pointing directions to optic coordinates
    #         # The pixel pointing directions are rigidly connected to the camera.
    #         # So when the camera moves, these vectors change, when expressed in optic cordinates.
    #         u_active_pixel_pointing_optic_new = u_pixel_pointing_cam.rotate(ori_new.r_cam_optic)
    #         # The screen points are rigidly connected to the screen, which is rigidly connected
    #         # to the camera by an unchanging calibration transform.
    #         # So when the camera moves, the screen points move, when expressed in optic cordinates.
    #         v_screen_points_optic_new = ori_new.trans_screen_optic.apply(self.v_screen_points_screen)
    #         v_optic_cam_optic_new = ori_new.v_optic_cam_optic
    #         u_measure_pixel_pointing_optic_new = self.surface.u_measure_pixel_pointing_optic.rotate(ori_new.r_cam_optic)
    #         v_align_point_optic_new = self.v_align_point_optic  # Align point is defined in optic coords.
    #         v_optic_screen_optic_new = ori_new.v_optic_screen_optic
    #         # Update surface cache
    #         self.surface.u_active_pixel_pointing_optic = u_active_pixel_pointing_optic_new[:: self.surface.downsample]
    #         self.surface.v_screen_points_optic = v_screen_points_optic_new[:: self.surface.downsample]
    #         self.surface.v_optic_cam_optic = v_optic_cam_optic_new
    #         self.surface.u_measure_pixel_pointing_optic = u_measure_pixel_pointing_optic_new
    #         self.surface.v_align_point_optic = v_align_point_optic_new
    #         self.surface.v_optic_screen_optic = v_optic_screen_optic_new
    #         # 5. Using POSE' project rays from camera to COEFFS surface, finding intersection points INT.
    #         # Downsample measurement data
    #         u_active_pixel_pointing_optic_new_downsample = u_active_pixel_pointing_optic_new[:: self.surface.downsample]
    #         # Project camera pixel rays and intersect with fit surface
    #         self.surface.v_surf_int_pts_optic = self.surface.intersect(
    #             u_active_pixel_pointing_optic_new_downsample, ori_new.v_optic_cam_optic
    #         )
    #         loop_record["n_intersect"] = self.surface.v_surf_int_pts_optic.len()
    #         # Check for invalid points
    #         # &&&& DELETE-SCAFFOLDING -- MOVE INTO INTERSECT() ROUTINE
    #         num_nans = np.isnan(self.surface.v_surf_int_pts_optic.data)
    #         if np.any(num_nans):
    #             warnings.warn(
    #                 f"{num_nans.sum():d} / {num_nans.size:d} values are NANs in surface intersection points in iteration: ({loop_idx:d}).",
    #                 stacklevel=2,
    #             )
    #         # Plot debug plot
    #         if self.debug.debug_active:
    #             ssdo.figure_intersection_surface_situation(
    #                 "After Calculate Intersections", vxyz_corners_sfc, None, self.surface, ori_new, loop_idx, self.debug
    #             )

    #         # 6. Use intersection points and reflection points RF to compute surface normals at INT points.
    #         self.surface.calculate_slopes()
    #         # Check for invalid points
    #         # &&&& DELETE-SCAFFOLDING -- MOVE INTO CALCULATE_SLOPES() ROUTINE
    #         num_nans = np.isnan(self.surface.slopes)
    #         if np.any(num_nans):
    #             warnings.warn(
    #                 f"{num_nans.sum():d} / {num_nans.size:d} values are NANs in slope data in iteration: ({loop_idx:d}).",
    #                 stacklevel=2,
    #             )

    #         # &&&& DELETE-SCAFFOLDING -- DOCUMENT NEW APPROACH: ALIGN BASED ON AVERAGE SLOPE.
    #         u_avg_fit_normal = self.surface.average_fit_normal()
    #         u_avg_measured_normal = self.surface.average_measured_normal()
    #         loop_record["u_avg_fit_normal"] = u_avg_fit_normal
    #         loop_record["u_avg_measured_normal"] = u_avg_measured_normal
    #         loop_record["u_avg_measured_minus_fit"] = u_avg_measured_normal.as_Vxyz() - u_avg_fit_normal.as_Vxyz()
    #         print("\nIn fit_surface_2(), before alignment rotation:")
    #         print("   normal_design_at_align_point =", self.surface.normal_design_at_align_point().to_str())
    #         print("   normal_fit_at_align_point    =", self.surface.normal_fit_at_align_point().to_str())
    #         print("   u_avg_fit_normal             =", loop_record["u_avg_fit_normal"].to_str())
    #         print("   u_avg_measured_normal        =", loop_record["u_avg_measured_normal"].to_str())
    #         print("   Delta: measured minus fit    =", loop_record["u_avg_measured_minus_fit"].to_str())
    #         print("\n")

    #         # &&&& DELETE-SCAFFOLDING -- DOCUMENT NEW APPROACH: ALIGN BASED ON AVERAGE SLOPE.
    #         # # &&&& DELETE-SCAFFOLDING -- THRESHOLDS SET FOR A SPECIFIC MIRROR EXAMPLE
    #         if (loop_idx > 1) and (loop_idx <= 7):  # loop_idx = 1 first pass through loop.
    #             # Capture current camera rotation and translation.
    #             r_cam_optic_new_copy = copy.deepcopy(ori_new.r_cam_optic)
    #             v_cam_optic_cam_new_copy = copy.deepcopy(ori_new.v_cam_optic_cam)
    #             # Calculate the rotation needed to align the normal vectors
    #             # ORIGINAL: WRONG DIRECTION r_align_step = u_avg_fit_normal.align_to(u_avg_measured_normal)  # &&&& DELETE-SCAFFOLDING -- DELETE?
    #             r_align_step = u_avg_measured_normal.align_to(u_avg_fit_normal)
    #             loop_record["r_align_step"] = r_align_step
    #             # # Rotate all points about alignment point.
    #             # self.surface.rotate_all(r_align_step)
    #             # Update orientation.
    #             r_cam_optic_new_2 = r_align_step * r_cam_optic_new_copy
    #             v_cam_optic_cam_new_2 = v_cam_optic_cam_new_copy
    #             print("\nIn fit_surface_2(), alignment rotation:")
    #             print("   r_cam_optic_new_copy =", r_cam_optic_new_copy.as_euler('XYZ', degrees=True))
    #             print("   r_align_step         =", r_align_step.as_euler('XYZ', degrees=True))
    #             print("   r_cam_optic_new_2    =", r_cam_optic_new_2.as_euler('XYZ', degrees=True))
    #             print("\n")
    #             ori_new = copy.deepcopy(original_orientation)
    #             ori_new.orient_optic_cam(r_cam_optic_new_2, v_cam_optic_cam_new_2)
    #             loop_record["r_cam_optic"] = ori_new.r_cam_optic
    #             loop_record["v_cam_optic_cam"] = ori_new.v_cam_optic_cam
    #             # $$$$ BEGIN REPEAT
    #             # 4b. Update cached values in surface to match new camera POSE'.
    #             #     From Surface2DParabolic.set_spatial_data()
    #             #         # Downsample and save measurement data
    #             #         self.u_active_pixel_pointing_optic = u_active_pixel_pointing_optic[:: self.downsample]
    #             #         self.v_screen_points_optic = v_screen_points_optic[:: self.downsample]
    #             #         # Save position data
    #             #         self.v_optic_cam_optic = v_optic_cam_optic
    #             #         self.u_measure_pixel_pointing_optic = u_measure_pixel_pointing_optic
    #             #         self.v_align_point_optic = v_align_point_optic
    #             #         self.v_optic_screen_optic = v_optic_screen_optic
    #             #         Convert pixel pointing directions to optic coordinates
    #             # The pixel pointing directions are rigidly connected to the camera.
    #             # So when the camera moves, these vectors change, when expressed in optic cordinates.
    #             u_active_pixel_pointing_optic_new = u_pixel_pointing_cam.rotate(ori_new.r_cam_optic)
    #             # The screen points are rigidly connected to the screen, which is rigidly connected
    #             # to the camera by an unchanging calibration transform.
    #             # So when the camera moves, the screen points move, when expressed in optic cordinates.
    #             v_screen_points_optic_new = ori_new.trans_screen_optic.apply(self.v_screen_points_screen)
    #             v_optic_cam_optic_new = ori_new.v_optic_cam_optic
    #             u_measure_pixel_pointing_optic_new = self.surface.u_measure_pixel_pointing_optic.rotate(
    #                 ori_new.r_cam_optic
    #             )
    #             v_align_point_optic_new = self.v_align_point_optic  # Align point is defined in optic coords.
    #             v_optic_screen_optic_new = ori_new.v_optic_screen_optic
    #             # Update surface cache
    #             self.surface.u_active_pixel_pointing_optic = u_active_pixel_pointing_optic_new[
    #                 :: self.surface.downsample
    #             ]
    #             self.surface.v_screen_points_optic = v_screen_points_optic_new[:: self.surface.downsample]
    #             self.surface.v_optic_cam_optic = v_optic_cam_optic_new
    #             self.surface.u_measure_pixel_pointing_optic = u_measure_pixel_pointing_optic_new
    #             self.surface.v_align_point_optic = v_align_point_optic_new
    #             self.surface.v_optic_screen_optic = v_optic_screen_optic_new
    #             # 5. Using POSE' project rays from camera to COEFFS surface, finding intersection points INT.
    #             # Downsample measurement data
    #             u_active_pixel_pointing_optic_new_downsample = u_active_pixel_pointing_optic_new[
    #                 :: self.surface.downsample
    #             ]
    #             # Project camera pixel rays and intersect with fit surface
    #             self.surface.v_surf_int_pts_optic = self.surface.intersect(
    #                 u_active_pixel_pointing_optic_new_downsample, ori_new.v_optic_cam_optic
    #             )
    #             loop_record["n_intersect"] = self.surface.v_surf_int_pts_optic.len()
    #             # Check for invalid points
    #             # &&&& DELETE-SCAFFOLDING -- MOVE INTO INTERSECT() ROUTINE
    #             num_nans = np.isnan(self.surface.v_surf_int_pts_optic.data)
    #             if np.any(num_nans):
    #                 warnings.warn(
    #                     f"{num_nans.sum():d} / {num_nans.size:d} values are NANs in surface intersection points in iteration: ({loop_idx:d}).",
    #                     stacklevel=2,
    #                 )
    #             # Plot debug plot
    #             if self.debug.debug_active:
    #                 ssdo.figure_intersection_surface_situation(
    #                     "After Align Rotate, Then Calculate Intersections",
    #                     vxyz_corners_sfc,
    #                     None,
    #                     self.surface,
    #                     ori_new,
    #                     loop_idx,
    #                     self.debug,
    #                 )

    #             # 6. Use intersection points and reflection points RF to compute surface normals at INT points.
    #             self.surface.calculate_slopes()
    #             # Check for invalid points
    #             # &&&& DELETE-SCAFFOLDING -- MOVE INTO CALCULATE_SLOPES() ROUTINE
    #             num_nans = np.isnan(self.surface.slopes)
    #             if np.any(num_nans):
    #                 warnings.warn(
    #                     f"{num_nans.sum():d} / {num_nans.size:d} values are NANs in slope data in iteration: ({loop_idx:d}).",
    #                     stacklevel=2,
    #                 )
    #             # Plot debug plot
    #             if self.debug.debug_active:
    #                 ssdo.figure_intersection_surface_situation(
    #                     "After Align Rotate, Calculate Intersections, Calculate Slopes",
    #                     vxyz_corners_sfc,
    #                     None,
    #                     self.surface,
    #                     ori_new,
    #                     loop_idx,
    #                     self.debug,
    #                 )
    #             # $$$$ END REPEAT
    #             # Update normal summaries.
    #             u_avg_fit_normal = self.surface.average_fit_normal()
    #             u_avg_measured_normal = self.surface.average_measured_normal()
    #             loop_record["u_avg_fit_normal"] = u_avg_fit_normal
    #             loop_record["u_avg_measured_normal"] = u_avg_measured_normal
    #             loop_record["u_avg_measured_minus_fit"] = u_avg_measured_normal.as_Vxyz() - u_avg_fit_normal.as_Vxyz()
    #             print(
    #                 "\nIn fit_surface_2(), after alignment rotation, then calculate intersections, then calculate slopes:"
    #             )
    #             print("   normal_design_at_align_point =", self.surface.normal_design_at_align_point().to_str())
    #             print("   normal_fit_at_align_point    =", self.surface.normal_fit_at_align_point().to_str())
    #             print("   u_avg_fit_normal             =", loop_record["u_avg_fit_normal"].to_str())
    #             print("   u_avg_measured_normal        =", loop_record["u_avg_measured_normal"].to_str())
    #             print("   Delta: measured minus fit    =", loop_record["u_avg_measured_minus_fit"].to_str())
    #             print("\n")
    #         else:
    #             loop_record["r_align_step"] = None

    #         # # # &&&& DELETE-SCAFFOLDING -- TEMPORARY?
    #         # # Plot debug plot
    #         # if self.debug.debug_active:
    #         #     ssdo.figure_intersection_surface_situation(
    #         #         "After Alignment Rotation", vxyz_corners_sfc, None, self.surface, ori_new, loop_idx, self.debug
    #         #     )

    #         # 7. Using surface normals at points, compute regression fit for slope coefficients.
    #         # 8. Convert fit slope coefficients to new surface COEFFS' = c0', c1x', c2x2', c3y', c4xy', c5y2'.
    #         # # &&&& DELETE-SCAFFOLDING -- THRESHOLDS SET FOR A SPECIFIC MIRROR EXAMPLE
    #         if loop_idx > 7:
    #             self.surface.fit_slopes()
    #         loop_record["surf_coefs"] = self.surface.surf_coefs
    #         loop_record["slope_coefs"] = self.surface.slope_coefs

    #         # Set the fine facet boundary z values to lie on the new fit surface.
    #         z_facet_corners_hires_4 = sf2.coef_to_points(vxyz_corners_sfc, self.surface.surf_coefs, 2)
    #         v_facet_corners_hires_4 = copy.deepcopy(vxyz_corners_sfc)
    #         v_facet_corners_hires_4.data[2, :] = z_facet_corners_hires_4
    #         # Check fine facet boundary z values.
    #         v_facet_corners_hires_4_minus_1 = v_facet_corners_hires_4 - vxyz_corners_sfc
    #         largest_change = abs(v_facet_corners_hires_4_minus_1.data[2, :]).max()
    #         # &&&& DELETE-SCAFFOLDING -- FIX CONTAINING ROUTINE NAME IN MESSAGE BELOW
    #         lt.info(
    #             f"In fit_surface_2(), maximum z difference between high-resolution corners before and after fit = {largest_change} m."
    #         )

    #         # High-resolution points reprojected (after snap to new fit surface).
    #         # &&&& DELETE-SCAFFOLDING -- MOVE REPROJECTION INTO DEBUG FIGURE ROUTINE
    #         hires_pts_reproj_4 = self.debug.debug_geometry.camera.project(
    #             v_facet_corners_hires_4, ori_new.r_cam_optic.inv(), ori_new.v_cam_optic_cam
    #         )
    #         # Plot reprojected points over mask image
    #         if self.debug.debug_active:
    #             ssdo.reproj_after_fit(
    #                 vxy_corners_sfc_reproj_snap, vxy_corners_sfc_reproj_new, hires_pts_reproj_4, loop_idx, self.debug
    #             )
    #         # Plot debug plot
    #         if self.debug.debug_active:
    #             ssdo.figure_intersection_surface_situation(
    #                 "After Slope Fit",
    #                 vxyz_corners_sfc,
    #                 v_facet_corners_hires_4,
    #                 self.surface,
    #                 ori_new,
    #                 loop_idx,
    #                 self.debug,
    #             )

    #         # Summarize loop progress.
    #         if self.debug.debug_active:
    #             lt.info("\nIn fit_surface_2(), loop_record_list:")
    #             lt.info(ssdo.fit_surface_loop_record_column_headings())
    #             lt.info(ssdo.fit_surface_loop_record_column_headings_units())
    #             lt.info(ssdo.fit_surface_loop_record_column_headings_separator())
    #             for loop_record in loop_record_list:
    #                 lt.info(ssdo.fit_surface_loop_record_str(loop_record))
    #             lt.info(ssdo.fit_surface_loop_record_column_headings_separator())

    #             lt.info("\nIn fit_surface_2(), loop_record_list 2:")
    #             lt.info(ssdo.fit_surface_loop_record_column_headings_2())
    #             lt.info(ssdo.fit_surface_loop_record_column_headings_units_2())
    #             lt.info(ssdo.fit_surface_loop_record_column_headings_separator_2())
    #             for loop_record in loop_record_list:
    #                 lt.info(ssdo.fit_surface_loop_record_str_2(loop_record))
    #             lt.info(ssdo.fit_surface_loop_record_column_headings_separator_2())

    #         # Check loop termination.
    #         if loop_idx >= 20:  # 12:  # 7:  # 1:  # 20:  # &&&& DELETE-SCAFFOLDING -- NEEDS BETTER LOOP EXIT CONTROL.

    #             break
    #         else:
    #             # Update loop entry values.
    #             vxyz_corners_entering_loop = vxyz_corners_sfc
    #             r_cam_optic_entering_loop = ori_new.r_cam_optic
    #             v_cam_optic_cam_entering_loop = ori_new.v_cam_optic_cam

    #     # Update intersection points and slopes after final fit_slopes() call.
    #     # Convert final pixel pointing directions to optic coordinates
    #     u_active_pixel_pointing_optic_new = u_pixel_pointing_cam.rotate(ori_new.r_cam_optic)
    #     # Project final camera pixel rays and intersect with fit surface
    #     self.surface.v_surf_int_pts_optic = self.surface.intersect(
    #         u_active_pixel_pointing_optic_new, ori_new.v_optic_cam_optic
    #     )
    #     # &&&& DELETE-SCAFFOLDING -- USING self.v_screen_points_facet IS PROBABLY NOT RIGHT, BECAUSE I DON'T KNOW THAT IT HAS BEEN UPDATED TO NEW TRANSFORM
    #     slopes_facet_xy = sf2.calc_slopes(
    #         self.surface.v_surf_int_pts_optic, self.surface.v_optic_cam_optic, self.v_screen_points_facet
    #     )

    #     # Store loop results
    #     self._data.surf_coefs_facet = self.surface.surf_coefs
    #     self._data.slope_coefs_facet = self.surface.slope_coefs
    #     self._data.trans_alignment = TransformXYZ.identity()  # &&&& DELETE-SCAFFOLDING -- ELIMINATE THIS?  MULTI-FACET?
    #     self._data.v_surf_points_facet = self.surface.v_surf_int_pts_optic
    #     self._data.slopes_facet_xy = slopes_facet_xy

    # # &&&& DELETE-SCAFFOLDING -- END NEW GEN 2 SOLVER ALGORITHM

    # &&&& DELETE-SCAFFOLDING -- BEGIN NEW GEN 3 SOLVER ALGORITHM

    def fit_surface_sf2gen3(self, original_orientation: SpatialOrientation) -> None:
        """
        # &&&& DELETE-SCAFFOLDING -- FIX THIS DOCSTRING
        Performs the initial fine-tuning alignment of the facet, screen, and
        camera. Fits a surface to the calculated slope data.

        """
        # &&&& DELETE-SCAFFOLDING -- PASS THIS IN
        vxyz_corners_hires_0 = self.debug.debug_geometry.v_facet_corners_hires
        # &&&& DELETE-SCAFFOLDING -- PASS THIS IN
        camera = self.debug.debug_geometry.camera
        # &&&& DELETE-SCAFFOLDING -- PASS THIS IN
        mask_processed = self.debug.debug_geometry.mask_processed

        # # &&&& DELETE-SCAFFOLDING -- DELETE ONCE CLEAR NOT NEEDED
        # # Save initial camera pose for later comparison.
        # r_cam_optic_original = self.debug.debug_geometry.r_cam_optic_refine_1
        # v_cam_optic_cam_original = self.debug.debug_geometry.v_cam_optic_cam_refine_2

        # &&&& DELETE-SCAFFOLDING -- MOVE OUTSIDE LOOP?
        # Calculate pixel pointing directions (camera coordinates)
        u_pixel_pointing_cam = ip.calculate_active_pixels_vectors(mask_processed, camera)

        # Construct search spiral for snapping boundary points to nearest edge.
        search_spiral = ip.construct_search_spiral(max_radius=50)
        # &&&& DELETE-SCAFFOLDING -- TEMPORARY
        print('In fit_surface_sf2gen3(), spiral complete.  Spiral length=', len(search_spiral))

        # Iteratively serach for a compatible solution:
        #    Stable inputs:
        #        Facet vertices V defining the facet boundary on the (x,y) plane.
        #        Camera model
        #        Mask image
        #        Pixel (x,y,z) reflection points RF on screen/target.
        #    Begin with:
        #        (a) Hypothesized surface equation COEFFS = c0, c1x, c2x2, c3y, c4xy, c5y2.
        #                z = c0 + c1x*x + c2x2*x^2 + c3y*y + c4xy*x*y + c5y2*y^2.
        #        (b) Initial camera pose estimate POSE = (R_cam,T_cam).
        #    Repeat:
        #        1. Project facet boundary vertices along z onto surface equation COEFFS ==> 3-d facet vertices V'.
        #        2. Use camera model and current POSE estimate to project 3-d vertices onto image.
        #        3. Refine projected image points by snapping onto edges in mask image.
        #        4. Use refined image points to call solvePnP() and compute a refined camera POSE'.
        #        5. Using POSE' project rays from camera to COEFFS surface, finding intersection points INT.
        #        6. Use intersection points and reflection points RF to compute surface normals at INT points.
        #        7. Using surface normals at points, compute regression fit for slope coefficients.
        #        8. Convert fit slope coefficients to new surface COEFFS' = c0', c1x', c2x2', c3y', c4xy', c5y2'.
        #    Until new COEFFS' agree with original COEFFS, up to a tolerance.

        # Initial values.
        loop_idx = 0
        vxyz_corners_entering_loop = vxyz_corners_hires_0
        r_cam_optic_entering_loop = self.debug.debug_geometry.r_cam_optic_refine_1
        v_cam_optic_cam_entering_loop = self.debug.debug_geometry.v_cam_optic_cam_refine_2

        # Keep track of loop progress.
        loop_record_0 = {
            "loop_idx": loop_idx,
            "loop_action_list": ["initial condition"],
            "surf_coefs": self.surface.surf_coefs,
            "slope_coefs": self.surface.slope_coefs,
            "r_cam_optic": r_cam_optic_entering_loop,
            "v_cam_optic_cam": v_cam_optic_cam_entering_loop,
            "pose_rms": None,
            "pose_dist_optic_screen": None,
            "vxyz_corners": vxyz_corners_entering_loop,
            "vxyz_corner_change": (vxyz_corners_entering_loop - vxyz_corners_entering_loop),  # Zero change.
            "n_intersect": 0,
            "u_avg_fit_normal": Uxyz([0, 0, 1]),
            "u_avg_measured_normal": Uxyz([0, 0, 1]),
            "u_avg_measured_minus_fit": Uxyz([0, 0, 1]),
            "r_align_step": None,
        }
        loop_record_list = [loop_record_0]

        # Main loop.
        while True:
            loop_idx += 1
            loop_record = {"loop_idx": loop_idx, "loop_action_list": []}
            loop_record_list.append(loop_record)

            # 1. Project facet boundary vertices along z onto surface equation COEFFS ==> 3-d facet vertices V'.
            z_corners_sfc = sf2.coef_to_points(vxyz_corners_entering_loop, self.surface.surf_coefs, 2)
            vxyz_corners_sfc = copy.deepcopy(vxyz_corners_entering_loop)
            vxyz_corners_sfc.data[2, :] = z_corners_sfc
            # Note how far the vertices moved.
            vxyz_corners_change = vxyz_corners_sfc - vxyz_corners_entering_loop
            loop_record["vxyz_corners"] = vxyz_corners_sfc
            loop_record["vxyz_corner_change"] = vxyz_corners_change

            # 2. Use camera model and current POSE estimate to project 3-d vertices onto image.
            vxy_corners_sfc_reproj = self.debug.debug_geometry.camera.project(
                vxyz_corners_sfc, r_cam_optic_entering_loop.inv(), v_cam_optic_cam_entering_loop
            )

            # 3. Refine projected image points by snapping onto edges in mask image.
            sfc_pt_reproj_pt_snapped_pt_list = ip.snap_points_to_nearest_edge(
                vxyz_corners_sfc, vxy_corners_sfc_reproj, mask_processed, search_spiral
            )
            # vxyz_corners_sfc_matched = Vxyz.from_list([a[0] for a in sfc_pt_reproj_pt_snapped_pt_list])
            vxy_corners_sfc_reproj_matched = Vxy.from_list([a[1] for a in sfc_pt_reproj_pt_snapped_pt_list])
            vxy_corners_sfc_reproj_snap = Vxy.from_list([a[2] for a in sfc_pt_reproj_pt_snapped_pt_list])
            # Plot reprojected points over mask image
            if self.debug.debug_active:
                ssdo.reproj_snap_2(
                    vxy_corners_sfc_reproj_matched, vxy_corners_sfc_reproj_snap, loop_idx=loop_idx, debug=self.debug
                )

            # &&&& DELETE-SCAFFOLDING -- IMPROVE THIS LOOP CONTROL
            # ********
            lt.info(f'In fit_surface_sf2gen3, loop_idx={loop_idx}:')
            if (loop_idx == 1) or (loop_idx == 11):
                loop_record["loop_action_list"].append("find_best_camera_pose_preserving_aim_and_distance")
                # &&&& DELETE-SCAFFOLDING -- RENAME THESE VARIABLES, MOST LIKELY
                r_cam_optic_new, v_cam_optic_cam_new, camera_view_alignment_rms_error, camera_view_dist_optic_screen = (
                    self.find_best_camera_pose_preserving_aim_and_distance(
                        r_cam_optic_entering_loop,
                        v_cam_optic_cam_entering_loop,
                        original_orientation,
                        vxyz_corners_sfc,
                        mask_processed,
                        search_spiral,
                    )
                )
                loop_record["pose_rms"] = camera_view_alignment_rms_error
                loop_record["pose_dist_optic_screen"] = camera_view_dist_optic_screen
                lt.info(
                    f'In fit_surface_sf2gen3, final reproject-to-mask-edges alignment error={camera_view_alignment_rms_error}.'
                    f'In fit_surface_sf2gen3, final reproject-to-mask-edges dist_optic_screen={camera_view_dist_optic_screen}.'
                )
            else:
                # &&&& DELETE-SCAFFOLDING -- RENAME THESE VARIABLES, MOST LIKELY
                r_cam_optic_new = r_cam_optic_entering_loop
                v_cam_optic_cam_new = v_cam_optic_cam_entering_loop

            # ********
            # # # &&&& DELETE-SCAFFOLDING -- OBSOLETE CODE
            # r_cam_optic_new = r_cam_optic_adjusted
            # v_cam_optic_cam_new = v_cam_optic_cam_adjusted_2
            # # &&&& DELETE-SCAFFOLDING -- SHOULD THIS NAME BE INVERTED?
            # r_cam_optic_new = r_optic_cam_new.inv()
            # Orient optic
            # &&&& DELETE-SCAFFOLDING -- TEMPORARY, OR DOCUMENT
            # # &&&& DELETE-SCAFFOLDING -- ORIGINAL GEN 2 VERSION
            # if loop_idx == 1:  # loop_idx = 1 first pass through lopp.
            #     ori_new = copy.copy(original_orientation)
            #     ori_new.orient_optic_cam(r_cam_optic_new, v_cam_optic_cam_new)
            # &&&& DELETE-SCAFFOLDING -- TEMPORARY, OR DOCUMENT
            # &&&& DELETE-SCAFFOLDING -- "< 1000" IS TEMPORARY
            if loop_idx < 1000:  # loop_idx = 1 first pass through lopp.
                loop_record["loop_action_list"].append("orient_optic_cam")
                ori_new = copy.copy(original_orientation)
                ori_new.orient_optic_cam(r_cam_optic_new, v_cam_optic_cam_new)
            loop_record["r_cam_optic"] = ori_new.r_cam_optic
            loop_record["v_cam_optic_cam"] = ori_new.v_cam_optic_cam
            # Plot reprojected points over mask image
            # &&&& DELETE-SCAFFOLDING -- MOVE REPROJECTION INTO DEBUG FIGURE ROUTINE
            vxy_corners_sfc_reproj_new = self.debug.debug_geometry.camera.project(
                vxyz_corners_sfc, ori_new.r_cam_optic.inv(), ori_new.v_cam_optic_cam
            )
            # #$$$$$$$$$$
            # if self.debug.debug_active:
            #     ssdo.reproj_after_snap_snap(
            #         vxy_corners_sfc_reproj_snap, vxy_corners_sfc_reproj_new, loop_idx, self.debug
            #     )

            # 4b. Update cached values in surface to match new camera POSE'.
            #     From Surface2DParabolic.set_spatial_data()
            #         # Downsample and save measurement data
            #         self.u_active_pixel_pointing_optic = u_active_pixel_pointing_optic[:: self.downsample]
            #         self.v_screen_points_optic = v_screen_points_optic[:: self.downsample]
            #         # Save position data
            #         self.v_optic_cam_optic = v_optic_cam_optic
            #         self.u_measure_pixel_pointing_optic = u_measure_pixel_pointing_optic
            #         self.v_align_point_optic = v_align_point_optic
            #         self.v_optic_screen_optic = v_optic_screen_optic
            #         Convert pixel pointing directions to optic coordinates
            # The pixel pointing directions are rigidly connected to the camera.
            # So when the camera moves, these vectors change, when expressed in optic cordinates.
            u_active_pixel_pointing_optic_new = u_pixel_pointing_cam.rotate(ori_new.r_cam_optic)
            # The screen points are rigidly connected to the screen, which is rigidly connected
            # to the camera by an unchanging calibration transform.
            # So when the camera moves, the screen points move, when expressed in optic cordinates.
            v_screen_points_optic_new = ori_new.trans_screen_optic.apply(self.v_screen_points_screen)
            v_optic_cam_optic_new = ori_new.v_optic_cam_optic
            u_measure_pixel_pointing_optic_new = self.surface.u_measure_pixel_pointing_optic.rotate(ori_new.r_cam_optic)
            v_align_point_optic_new = self.v_align_point_optic  # Align point is defined in optic coords.
            v_optic_screen_optic_new = ori_new.v_optic_screen_optic
            # Update surface cache
            self.surface.u_active_pixel_pointing_optic = u_active_pixel_pointing_optic_new[:: self.surface.downsample]
            self.surface.v_screen_points_optic = v_screen_points_optic_new[:: self.surface.downsample]
            self.surface.v_optic_cam_optic = v_optic_cam_optic_new
            self.surface.u_measure_pixel_pointing_optic = u_measure_pixel_pointing_optic_new
            self.surface.v_align_point_optic = v_align_point_optic_new
            self.surface.v_optic_screen_optic = v_optic_screen_optic_new
            # 5. Using POSE' project rays from camera to COEFFS surface, finding intersection points INT.
            # Downsample measurement data
            u_active_pixel_pointing_optic_new_downsample = u_active_pixel_pointing_optic_new[:: self.surface.downsample]
            # Project camera pixel rays and intersect with fit surface
            loop_record["loop_action_list"].append("intersect camera rays")
            self.surface.v_surf_int_pts_optic = self.surface.intersect(
                u_active_pixel_pointing_optic_new_downsample, ori_new.v_optic_cam_optic
            )
            loop_record["n_intersect"] = self.surface.v_surf_int_pts_optic.len()
            # Check for invalid points
            # &&&& DELETE-SCAFFOLDING -- MOVE INTO INTERSECT() ROUTINE
            num_nans = np.isnan(self.surface.v_surf_int_pts_optic.data)
            if np.any(num_nans):
                warnings.warn(
                    f"{num_nans.sum():d} / {num_nans.size:d} values are NANs in surface intersection points in iteration: ({loop_idx:d}).",
                    stacklevel=2,
                )
            # Plot debug plot
            if self.debug.debug_active:
                ssdo.figure_intersection_surface_situation(
                    "After Calculate Intersections", vxyz_corners_sfc, None, self.surface, ori_new, loop_idx, self.debug
                )

            # &&&& DELETE-SCAFFOLDING -- BEGIN BLOCK OFF NON-POSITION

            if True:  # loop_idx > 2:  # &&&& DELETE-SCAFFOLDING -- FIX THIS
                loop_record["loop_action_list"].append("calculate_slopes")
                # 6. Use intersection points and reflection points RF to compute surface normals at INT points.
                self.surface.calculate_slopes()
                # Check for invalid points
                # &&&& DELETE-SCAFFOLDING -- MOVE INTO CALCULATE_SLOPES() ROUTINE
                num_nans = np.isnan(self.surface.slopes)
                if np.any(num_nans):
                    warnings.warn(
                        f"{num_nans.sum():d} / {num_nans.size:d} values are NANs in slope data in iteration: ({loop_idx:d}).",
                        stacklevel=2,
                    )

                # &&&& DELETE-SCAFFOLDING -- DOCUMENT NEW APPROACH: ALIGN BASED ON AVERAGE SLOPE.
                u_avg_fit_normal = self.surface.average_fit_normal()
                u_avg_measured_normal = self.surface.average_measured_normal()
                loop_record["u_avg_fit_normal"] = u_avg_fit_normal
                loop_record["u_avg_measured_normal"] = u_avg_measured_normal
                loop_record["u_avg_measured_minus_fit"] = u_avg_measured_normal.as_Vxyz() - u_avg_fit_normal.as_Vxyz()
                print("\nIn fit_surface_sf2gen3(), before alignment rotation:")
                print("   normal_design_at_align_point =", self.surface.normal_design_at_align_point().to_str())
                print("   normal_fit_at_align_point    =", self.surface.normal_fit_at_align_point().to_str())
                print("   u_avg_fit_normal             =", loop_record["u_avg_fit_normal"].to_str())
                print("   u_avg_measured_normal        =", loop_record["u_avg_measured_normal"].to_str())
                print("   Delta: measured minus fit    =", loop_record["u_avg_measured_minus_fit"].to_str())
                print("\n")

                # &&&& DELETE-SCAFFOLDING -- DOCUMENT NEW APPROACH: ALIGN BASED ON AVERAGE SLOPE.
                # # &&&& DELETE-SCAFFOLDING -- THRESHOLDS SET FOR A SPECIFIC MIRROR EXAMPLE
                if (loop_idx > 1) and (loop_idx <= 7):  # loop_idx = 1 first pass through loop.
                    loop_record["loop_action_list"].append("align fit and measured average slopes")
                    # Capture current camera rotation and translation.
                    r_cam_optic_new_copy = copy.deepcopy(ori_new.r_cam_optic)
                    v_cam_optic_cam_new_copy = copy.deepcopy(ori_new.v_cam_optic_cam)
                    # Calculate the rotation needed to align the normal vectors
                    # ORIGINAL: WRONG DIRECTION r_align_step = u_avg_fit_normal.align_to(u_avg_measured_normal)  # &&&& DELETE-SCAFFOLDING -- DELETE?
                    r_align_step = u_avg_measured_normal.align_to(u_avg_fit_normal)
                    loop_record["r_align_step"] = r_align_step
                    # # Rotate all points about alignment point.
                    # self.surface.rotate_all(r_align_step)
                    # Update orientation.
                    r_cam_optic_new_2 = r_align_step * r_cam_optic_new_copy
                    v_cam_optic_cam_new_2 = v_cam_optic_cam_new_copy
                    print("\nIn fit_surface_sf2gen3(), alignment rotation:")
                    print("   r_cam_optic_new_copy =", r_cam_optic_new_copy.as_euler('XYZ', degrees=True))
                    print("   r_align_step         =", r_align_step.as_euler('XYZ', degrees=True))
                    print("   r_cam_optic_new_2    =", r_cam_optic_new_2.as_euler('XYZ', degrees=True))
                    print("\n")
                    loop_record["loop_action_list"].append("orient_optic_cam_2")
                    ori_new = copy.deepcopy(original_orientation)
                    ori_new.orient_optic_cam(r_cam_optic_new_2, v_cam_optic_cam_new_2)
                    loop_record["r_cam_optic"] = ori_new.r_cam_optic
                    loop_record["v_cam_optic_cam"] = ori_new.v_cam_optic_cam
                    # $$$$ BEGIN REPEAT
                    # 4b. Update cached values in surface to match new camera POSE'.
                    #     From Surface2DParabolic.set_spatial_data()
                    #         # Downsample and save measurement data
                    #         self.u_active_pixel_pointing_optic = u_active_pixel_pointing_optic[:: self.downsample]
                    #         self.v_screen_points_optic = v_screen_points_optic[:: self.downsample]
                    #         # Save position data
                    #         self.v_optic_cam_optic = v_optic_cam_optic
                    #         self.u_measure_pixel_pointing_optic = u_measure_pixel_pointing_optic
                    #         self.v_align_point_optic = v_align_point_optic
                    #         self.v_optic_screen_optic = v_optic_screen_optic
                    #         Convert pixel pointing directions to optic coordinates
                    # The pixel pointing directions are rigidly connected to the camera.
                    # So when the camera moves, these vectors change, when expressed in optic cordinates.
                    u_active_pixel_pointing_optic_new = u_pixel_pointing_cam.rotate(ori_new.r_cam_optic)
                    # The screen points are rigidly connected to the screen, which is rigidly connected
                    # to the camera by an unchanging calibration transform.
                    # So when the camera moves, the screen points move, when expressed in optic cordinates.
                    v_screen_points_optic_new = ori_new.trans_screen_optic.apply(self.v_screen_points_screen)
                    v_optic_cam_optic_new = ori_new.v_optic_cam_optic
                    u_measure_pixel_pointing_optic_new = self.surface.u_measure_pixel_pointing_optic.rotate(
                        ori_new.r_cam_optic
                    )
                    v_align_point_optic_new = self.v_align_point_optic  # Align point is defined in optic coords.
                    v_optic_screen_optic_new = ori_new.v_optic_screen_optic
                    # Update surface cache
                    self.surface.u_active_pixel_pointing_optic = u_active_pixel_pointing_optic_new[
                        :: self.surface.downsample
                    ]
                    self.surface.v_screen_points_optic = v_screen_points_optic_new[:: self.surface.downsample]
                    self.surface.v_optic_cam_optic = v_optic_cam_optic_new
                    self.surface.u_measure_pixel_pointing_optic = u_measure_pixel_pointing_optic_new
                    self.surface.v_align_point_optic = v_align_point_optic_new
                    self.surface.v_optic_screen_optic = v_optic_screen_optic_new
                    # 5. Using POSE' project rays from camera to COEFFS surface, finding intersection points INT.
                    # Downsample measurement data
                    u_active_pixel_pointing_optic_new_downsample = u_active_pixel_pointing_optic_new[
                        :: self.surface.downsample
                    ]
                    # Project camera pixel rays and intersect with fit surface
                    loop_record["loop_action_list"].append("intersect camera rays for new pose")
                    self.surface.v_surf_int_pts_optic = self.surface.intersect(
                        u_active_pixel_pointing_optic_new_downsample, ori_new.v_optic_cam_optic
                    )
                    loop_record["n_intersect"] = self.surface.v_surf_int_pts_optic.len()
                    # Check for invalid points
                    # &&&& DELETE-SCAFFOLDING -- MOVE INTO INTERSECT() ROUTINE
                    num_nans = np.isnan(self.surface.v_surf_int_pts_optic.data)
                    if np.any(num_nans):
                        warnings.warn(
                            f"{num_nans.sum():d} / {num_nans.size:d} values are NANs in surface intersection points in iteration: ({loop_idx:d}).",
                            stacklevel=2,
                        )
                    # Plot debug plot
                    if self.debug.debug_active:
                        ssdo.figure_intersection_surface_situation(
                            "After Align Rotate, Then Calculate Intersections",
                            vxyz_corners_sfc,
                            None,
                            self.surface,
                            ori_new,
                            loop_idx,
                            self.debug,
                        )

                    # 6. Use intersection points and reflection points RF to compute surface normals at INT points.
                    loop_record["loop_action_list"].append("calculate_slopes_after_align")
                    self.surface.calculate_slopes()
                    # Check for invalid points
                    # &&&& DELETE-SCAFFOLDING -- MOVE INTO CALCULATE_SLOPES() ROUTINE
                    num_nans = np.isnan(self.surface.slopes)
                    if np.any(num_nans):
                        warnings.warn(
                            f"{num_nans.sum():d} / {num_nans.size:d} values are NANs in slope data in iteration: ({loop_idx:d}).",
                            stacklevel=2,
                        )
                    # Plot debug plot
                    if self.debug.debug_active:
                        ssdo.figure_intersection_surface_situation(
                            "After Align Rotate, Calculate Intersections, Calculate Slopes",
                            vxyz_corners_sfc,
                            None,
                            self.surface,
                            ori_new,
                            loop_idx,
                            self.debug,
                        )
                    # $$$$ END REPEAT
                    # Update normal summaries.
                    u_avg_fit_normal = self.surface.average_fit_normal()
                    u_avg_measured_normal = self.surface.average_measured_normal()
                    loop_record["u_avg_fit_normal"] = u_avg_fit_normal
                    loop_record["u_avg_measured_normal"] = u_avg_measured_normal
                    loop_record["u_avg_measured_minus_fit"] = (
                        u_avg_measured_normal.as_Vxyz() - u_avg_fit_normal.as_Vxyz()
                    )
                    print(
                        "\nIn fit_surface_sf2gen3(), after alignment rotation, then calculate intersections, then calculate slopes:"
                    )
                    print("   normal_design_at_align_point =", self.surface.normal_design_at_align_point().to_str())
                    print("   normal_fit_at_align_point    =", self.surface.normal_fit_at_align_point().to_str())
                    print("   u_avg_fit_normal             =", loop_record["u_avg_fit_normal"].to_str())
                    print("   u_avg_measured_normal        =", loop_record["u_avg_measured_normal"].to_str())
                    print("   Delta: measured minus fit    =", loop_record["u_avg_measured_minus_fit"].to_str())
                    print("\n")
                else:
                    loop_record["r_align_step"] = None

                # # # &&&& DELETE-SCAFFOLDING -- TEMPORARY?
                # # Plot debug plot
                # if self.debug.debug_active:
                #     ssdo.figure_intersection_surface_situation(
                #         "After Alignment Rotation", vxyz_corners_sfc, None, self.surface, ori_new, loop_idx, self.debug
                #     )

                # 7. Using surface normals at points, compute regression fit for slope coefficients.
                # 8. Convert fit slope coefficients to new surface COEFFS' = c0', c1x', c2x2', c3y', c4xy', c5y2'.
                # # &&&& DELETE-SCAFFOLDING -- THRESHOLDS SET FOR A SPECIFIC MIRROR EXAMPLE
                if (loop_idx > 7) and (loop_idx != 11):
                    loop_record["loop_action_list"].append("fit_slopes")
                    self.surface.fit_slopes()
                loop_record["surf_coefs"] = self.surface.surf_coefs
                loop_record["slope_coefs"] = self.surface.slope_coefs

                # Set the fine facet boundary z values to lie on the new fit surface.
                loop_record["loop_action_list"].append("facet corner z values to new surface")
                z_facet_corners_hires_4 = sf2.coef_to_points(vxyz_corners_sfc, self.surface.surf_coefs, 2)
                v_facet_corners_hires_4 = copy.deepcopy(vxyz_corners_sfc)
                v_facet_corners_hires_4.data[2, :] = z_facet_corners_hires_4
                # Check fine facet boundary z values.
                v_facet_corners_hires_4_minus_1 = v_facet_corners_hires_4 - vxyz_corners_sfc
                largest_change = abs(v_facet_corners_hires_4_minus_1.data[2, :]).max()
                # &&&& DELETE-SCAFFOLDING -- FIX CONTAINING ROUTINE NAME IN MESSAGE BELOW
                lt.info(
                    f"In fit_surface_sf2gen3(), maximum z difference between high-resolution corners before and after fit = {largest_change} m."
                )

                # High-resolution points reprojected (after snap to new fit surface).
                # &&&& DELETE-SCAFFOLDING -- MOVE REPROJECTION INTO DEBUG FIGURE ROUTINE
                loop_record["loop_action_list"].append("project updated facet corners to image")
                hires_pts_reproj_4 = self.debug.debug_geometry.camera.project(
                    v_facet_corners_hires_4, ori_new.r_cam_optic.inv(), ori_new.v_cam_optic_cam
                )
                # Plot reprojected points over mask image
                if self.debug.debug_active:
                    ssdo.reproj_after_fit(
                        vxy_corners_sfc_reproj_snap,
                        vxy_corners_sfc_reproj_new,
                        hires_pts_reproj_4,
                        loop_idx,
                        self.debug,
                    )
                # Plot debug plot
                if self.debug.debug_active:
                    ssdo.figure_intersection_surface_situation(
                        "After Slope Fit",
                        vxyz_corners_sfc,
                        v_facet_corners_hires_4,
                        self.surface,
                        ori_new,
                        loop_idx,
                        self.debug,
                    )

                # Summarize loop progress.
                if self.debug.debug_active:
                    # lt.info("\nIn fit_surface_sf2gen3(), loop_record_list:")
                    # lt.info(ssdo.fit_surface_loop_record_column_headings())
                    # lt.info(ssdo.fit_surface_loop_record_column_headings_units())
                    # lt.info(ssdo.fit_surface_loop_record_column_headings_separator())
                    # for loop_record in loop_record_list:
                    #     lt.info(ssdo.fit_surface_loop_record_str(loop_record))
                    # lt.info(ssdo.fit_surface_loop_record_column_headings_separator())

                    # lt.info("\nIn fit_surface_sf2gen3(), loop_record_list 2:")
                    # lt.info(ssdo.fit_surface_loop_record_column_headings_2())
                    # lt.info(ssdo.fit_surface_loop_record_column_headings_units_2())
                    # lt.info(ssdo.fit_surface_loop_record_column_headings_separator_2())
                    # for loop_record in loop_record_list:
                    #     lt.info(ssdo.fit_surface_loop_record_str_2(loop_record))
                    # lt.info(ssdo.fit_surface_loop_record_column_headings_separator_2())

                    # lt.info("\nIn fit_surface_sf2gen3(), loop_record_action_list (only selected):")
                    # lt.info(ssdo.fit_surface_loop_record_column_headings_action())
                    # lt.info(ssdo.fit_surface_loop_record_column_headings_separator_action())
                    # for loop_record in loop_record_list:
                    #     lt.info(ssdo.fit_surface_loop_record_str_action(loop_record, only_loop_select_actions=True))
                    # lt.info(ssdo.fit_surface_loop_record_column_headings_separator_action())

                    lt.info("\nIn fit_surface_sf2gen3(), loop_record_action_list:")
                    lt.info(ssdo.fit_surface_loop_record_column_headings_action())
                    lt.info(ssdo.fit_surface_loop_record_column_headings_separator_action())
                    for loop_record in loop_record_list:
                        lt.info(ssdo.fit_surface_loop_record_str_action(loop_record))
                    lt.info(ssdo.fit_surface_loop_record_column_headings_separator_action())

                    lt.info("\nIn fit_surface_sf2gen3(), Loop convergence parameters:")
                    lt.info(ssdo.fit_surface_loop_record_column_headings_5())
                    lt.info(ssdo.fit_surface_loop_record_column_headings_units_5())
                    lt.info(ssdo.fit_surface_loop_record_column_headings_separator_5())
                    for loop_record in loop_record_list:
                        lt.info(ssdo.fit_surface_loop_record_str_5(loop_record))
                    lt.info(ssdo.fit_surface_loop_record_column_headings_separator_5())

                    lt.info("\nIn fit_surface_sf2gen3(), Loop solution health parameters:")
                    lt.info(ssdo.fit_surface_loop_record_column_headings_6())
                    lt.info(ssdo.fit_surface_loop_record_column_headings_units_6())
                    lt.info(ssdo.fit_surface_loop_record_column_headings_separator_6())
                    for loop_record in loop_record_list:
                        lt.info(ssdo.fit_surface_loop_record_str_6(loop_record))
                    lt.info(ssdo.fit_surface_loop_record_column_headings_separator_6())

            # &&&& DELETE-SCAFFOLDING -- END BLOCK OFF NON-POSITION

            # Check loop termination.
            if loop_idx >= 20:  # 12:  # 7:  # 1:  # 20:  # &&&& DELETE-SCAFFOLDING -- NEEDS BETTER LOOP EXIT CONTROL.

                break
            else:
                # Update loop entry values.
                vxyz_corners_entering_loop = vxyz_corners_sfc
                r_cam_optic_entering_loop = ori_new.r_cam_optic
                v_cam_optic_cam_entering_loop = ori_new.v_cam_optic_cam

        # Update intersection points and slopes after final fit_slopes() call.
        # Convert final pixel pointing directions to optic coordinates
        u_active_pixel_pointing_optic_new = u_pixel_pointing_cam.rotate(ori_new.r_cam_optic)
        # Project final camera pixel rays and intersect with fit surface
        self.surface.v_surf_int_pts_optic = self.surface.intersect(
            u_active_pixel_pointing_optic_new, ori_new.v_optic_cam_optic
        )
        # &&&& DELETE-SCAFFOLDING -- USING self.v_screen_points_facet IS PROBABLY NOT RIGHT, BECAUSE I DON'T KNOW THAT IT HAS BEEN UPDATED TO NEW TRANSFORM
        slopes_facet_xy = sf2.calc_slopes(
            self.surface.v_surf_int_pts_optic, self.surface.v_optic_cam_optic, self.v_screen_points_facet
        )

        # Store loop results
        self._data.surf_coefs_facet = self.surface.surf_coefs
        self._data.slope_coefs_facet = self.surface.slope_coefs
        self._data.trans_alignment = TransformXYZ.identity()  # &&&& DELETE-SCAFFOLDING -- ELIMINATE THIS?  MULTI-FACET?
        self._data.v_surf_points_facet = self.surface.v_surf_int_pts_optic
        self._data.slopes_facet_xy = slopes_facet_xy

    # &&&& DELETE-SCAFFOLDING -- END NEW GEN 3 SOLVER ALGORITHM

    def adjust_camera_pose_preserving_aim_and_distance(
        self,
        camera_pose_dx_dy_dtheta: tuple[float, float, float],
        r_cam_optic_entering_loop: Rotation,
        v_cam_optic_cam_entering_loop: Vxyz,
        original_orientation: SpatialOrientation,
    ):
        # 4. Use refined image points to call solvePnP() and compute a refined camera POSE'.
        # # &&&& DELETE-SCAFFOLDING -- ORIGINAL VERSION
        # r_optic_cam_new, v_cam_optic_cam_new = sp.calc_rt_from_img_pts(
        #     vxy_corners_sfc_reproj_snap,
        #     vxyz_corners_sfc_matched,
        #     camera,
        #     initial_rotation=r_cam_optic_entering_loop,
        #     initial_vxyz=v_cam_optic_cam_entering_loop,
        # )
        # # # &&&& DELETE-SCAFFOLDING -- WORKS, BUT SENSE REVERSED.
        # r_optic_cam_entering_loop = r_cam_optic_entering_loop.inv()
        # r_correction = Rotation.from_euler('z', 10.0, degrees=True)  # &&&& DELETE-SCAFFOLDING -- TEMPORARY
        # r_optic_cam_adjusted = r_optic_cam_entering_loop * r_correction

        # Desired rotation about camera optical optical axis, therefore in camera coordinates.
        # r_cam_correction = Rotation.from_euler('z', 5.0, degrees=True)  # &&&& DELETE-SCAFFOLDING -- TEMPORARY
        u_facet_centroid_normal = self.debug.debug_geometry.facet_data.u_facet_centroid_normal
        r_optic_cam_entering_loop = r_cam_optic_entering_loop.inv()
        rotated_u_facet_centroid_normal = u_facet_centroid_normal.rotate(r_optic_cam_entering_loop)
        ux = rotated_u_facet_centroid_normal.x[0]
        uy = rotated_u_facet_centroid_normal.y[0]
        uz = rotated_u_facet_centroid_normal.z[0]
        axis = np.array([ux, uy, uz])
        # # &&&& DELETE-SCAFFOLDING -- OBSOLETE CODE BLOCK
        # # angle = np.radians(0.0)  # &&&& DELETE-SCAFFOLDING -- TEMPORARY MANUAL SETTING
        # if loop_idx == 1:  # &&&& DELETE-SCAFFOLDING -- TEMPORARY MANUAL SETTING
        #     # angle = np.radians(3.5)  # &&&& DELETE-SCAFFOLDING -- TEMPORARY MANUAL SETTING
        #     angle = camera_pose_dtheta  # &&&& DELETE-SCAFFOLDING -- TEMPORARY MANUAL SETTING
        # else:  # &&&& DELETE-SCAFFOLDING -- TEMPORARY MANUAL SETTING
        #     angle = np.radians(0.0)  # &&&& DELETE-SCAFFOLDING -- TEMPORARY MANUAL SETTING
        angle = camera_pose_dx_dy_dtheta[2]
        r_cam_correction = Rotation.from_rotvec(angle * axis)  # &&&& DELETE-SCAFFOLDING -- TEMPORARY
        r_optic_cam_adjusted = r_cam_correction * r_optic_cam_entering_loop
        r_cam_optic_adjusted = r_optic_cam_adjusted.inv()
        # r_cam_optic_adjusted = r_cam_optic_entering_loop * r_cam_correction
        # r_cam_optic_adjusted = r_cam_optic_entering_loop

        r_optic_cam_entering_loop = r_cam_optic_entering_loop.inv()
        r_optic_cam_adjusted = r_cam_optic_adjusted.inv()

        v_align_point_optic = self.v_align_point_optic
        v_align_point_cam_entering_loop = v_align_point_optic.rotate(r_optic_cam_entering_loop)
        v_align_point_cam_adjusted = v_align_point_optic.rotate(r_optic_cam_adjusted)
        v_align_point_shift_cam = v_align_point_cam_adjusted - v_align_point_cam_entering_loop

        # Correction directions:
        #   +x: reprojected facet vertices move right in mask image
        #   +y: reprojected facet vertices move down in mask image
        #   +z: reprojected facet vertice outline gets smaller and appears further away
        # # &&&& DELETE-SCAFFOLDING -- OBSOLETE CODE BLOCK
        # if loop_idx == 1:  # &&&& DELETE-SCAFFOLDING -- TEMPORARY MANUAL SETTING
        #     v_correction = Vxyz([0.0, 0.0, 0.0])  # meters  # &&&& DELETE-SCAFFOLDING -- TEMPORARY MANUAL SETTING
        # elif loop_idx == 2:  # &&&& DELETE-SCAFFOLDING -- TEMPORARY MANUAL SETTING
        #     # v_correction = Vxyz([0.022, 0.0, 0.0])  # meters  # &&&& DELETE-SCAFFOLDING -- TEMPORARY MANUAL SETTING
        #     v_correction = Vxyz(
        #         [camera_pose_dx, camera_pose_dy, 0.0]
        #     )  # meters  # &&&& DELETE-SCAFFOLDING -- TEMPORARY MANUAL SETTING
        # else:  # &&&& DELETE-SCAFFOLDING -- TEMPORARY MANUAL SETTING
        #     v_correction = Vxyz([0.0, 0.0, 0.0])  # meters  # &&&& DELETE-SCAFFOLDING -- TEMPORARY MANUAL SETTING
        v_correction = Vxyz([camera_pose_dx_dy_dtheta[0], camera_pose_dx_dy_dtheta[1], 0.0])  # meters
        v_cam_optic_cam_adjusted = v_cam_optic_cam_entering_loop + v_correction - v_align_point_shift_cam

        v_cam_optic_cam_adjusted_2 = sp.refine_v_distance(
            v_cam_optic_cam_adjusted,
            self.dist_optic_screen,
            original_orientation.v_cam_screen_cam,
            v_align_point_cam_adjusted,
        )
        # &&&& DELETE-SCAFFOLDING -- DOCUMENT IMPLICIT ASSUMPTION HERE THAT ALIGNMENT POINT AND MEASURE POINT ARE THE SAME
        v_optic_screen_entering_loop = original_orientation.v_cam_screen_cam - (
            v_cam_optic_cam_entering_loop + v_align_point_cam_entering_loop
        )
        dist_optic_screen_entering_loop = v_optic_screen_entering_loop.magnitude()[0]

        v_optic_screen_adjusted = original_orientation.v_cam_screen_cam - (
            v_cam_optic_cam_adjusted + v_align_point_cam_adjusted
        )
        dist_optic_screen_adjusted = v_optic_screen_adjusted.magnitude()[0]

        v_optic_screen_adjusted_2 = original_orientation.v_cam_screen_cam - (
            v_cam_optic_cam_adjusted_2 + v_align_point_cam_adjusted
        )
        dist_optic_screen_adjusted_2 = v_optic_screen_adjusted_2.magnitude()[0]

        # # Keep for detailed testing.
        # if self.debug.debug_active:
        #     lt.info('In adjust_camera_pose_preserving_aim_and_distance:')
        #     lt.info('  dist_optic_screen_entering_loop   = ' + str(dist_optic_screen_entering_loop))
        #     lt.info('  dist_optic_screen_adjusted        = ' + str(dist_optic_screen_adjusted))
        #     lt.info('  dist_optic_screen_adjusted_2      = ' + str(dist_optic_screen_adjusted_2))

        # &&&& DELETE-SCAFFOLDING -- CLEANUP NAMES, SEMANTICS
        # Pass result forward.
        # # &&&& DELETE-SCAFFOLDING -- DELETE ONCE TESTED
        # r_optic_cam_new = r_cam_optic_adjusted.inv()  # r_optic_cam_adjusted.inv()
        # v_cam_optic_cam_new = v_cam_optic_cam_adjusted  # &&&& DELETE-SCAFFOLDING -- DELETE THIS

        # # &&&& DELETE-SCAFFOLDING -- DELETE ONCE TESTED
        # # &&&& DELETE-SCAFFOLDING -- SHOULD THIS NAME BE INVERTED?
        # r_cam_optic_new = r_optic_cam_new.inv()

        # Return.
        # &&&& DELETE-SCAFFOLDING -- DELETE ONCE TESTED
        # return r_optic_cam_new, v_cam_optic_cam_adjusted_2
        return r_cam_optic_adjusted, v_cam_optic_cam_adjusted_2, dist_optic_screen_adjusted_2

    def compute_pose_alignment_error_for_proposed_camera_adjustment(
        self,
        camera_dx_dy_dtheta: tuple[float, float, float],
        # # &&&& DELETE-SCAFFOLDING -- DELETE ONCE TESTED
        # camera_dx_dy_dtheta_array: np.ndarray,
        r_cam_optic_input: Rotation,
        v_cam_optic_cam_input: Vxyz,
        orientation: SpatialOrientation,
        vxyz_corners_sfc: Vxyz,
        mask_processed: np.ndarray,
        search_spiral: list[tuple[int, int, float]],
        output_debug_figure: bool,
        debug_figure_status_str: str | None = None,
    ):
        r_cam_optic_adjusted, v_cam_optic_cam_adjusted, dist_optic_screen_adjusted = (
            self.adjust_camera_pose_preserving_aim_and_distance(
                camera_dx_dy_dtheta, r_cam_optic_input, v_cam_optic_cam_input, orientation
            )
        )

        # 2. Use camera model and current POSE estimate to project 3-d vertices onto image.
        vxy_corners_sfc_reproj = self.debug.debug_geometry.camera.project(
            vxyz_corners_sfc, r_cam_optic_adjusted.inv(), v_cam_optic_cam_adjusted
        )

        # 3. Refine projected image points by snapping onto edges in mask image.
        sfc_pt_reproj_pt_snapped_pt_list = ip.snap_points_to_nearest_edge(
            vxyz_corners_sfc, vxy_corners_sfc_reproj, mask_processed, search_spiral
        )

        # Compute alignment error.
        # This is the root-mean-square (RMS) distance of the input points (considered to be error),
        # compared to the mask edge points (considered to be truth, in this calculation).
        distance_squared_sum = 0.0
        n = 0
        for world_image_snapped in sfc_pt_reproj_pt_snapped_pt_list:
            # world_xyz = world_image_snapped[0]
            image_xy = world_image_snapped[1]
            snapped_xy = world_image_snapped[2]
            image_x = image_xy.x[0]
            image_y = image_xy.y[0]
            snapped_x = snapped_xy.x[0]
            snapped_y = snapped_xy.y[0]
            dx = snapped_x - image_x
            dy = snapped_y - image_y
            distance = np.sqrt((dx * dx) + (dy * dy))
            distance_squared_sum += distance * distance
            n += 1
        mean_distance_squared = distance_squared_sum / n
        rms = np.sqrt(mean_distance_squared)

        # Debugging output.
        # Plot reprojected points over mask image
        if self.debug.debug_active and output_debug_figure:
            # vxyz_corners_sfc_matched = Vxyz.from_list([a[0] for a in sfc_pt_reproj_pt_snapped_pt_list])
            vxy_corners_sfc_reproj_matched = Vxy.from_list([a[1] for a in sfc_pt_reproj_pt_snapped_pt_list])
            vxy_corners_sfc_reproj_snap = Vxy.from_list([a[2] for a in sfc_pt_reproj_pt_snapped_pt_list])
            ssdo.reproj_snap_3(
                vxy_corners_sfc_reproj_matched,
                vxy_corners_sfc_reproj_snap,
                status_str=debug_figure_status_str,
                camera_dx_dy_dtheta=camera_dx_dy_dtheta,
                rms=rms,
                debug=self.debug,
            )

        # Return.
        return rms, dist_optic_screen_adjusted

    def optimize_camera_pose_preserving_aim_and_distance(
        self,
        r_cam_optic_input: Rotation,
        v_cam_optic_cam_input: Vxyz,
        orientation: SpatialOrientation,
        vxyz_corners_sfc: Vxyz,
        mask_processed: np.ndarray,
        search_spiral: list[tuple[int, int, float]],
    ):

        # &&&& DELETE-SCAFFOLDING -- WHERE SHOULD THIS COMENT BLOCK GO?
        #
        # &&&& DELETE-SCAFFOLDING -- FLESH OUT AND FINISH THIS COMENT BLOCK
        #
        # &&&& DELETE-SCAFFOLDING -- DISCUSS SCIPY APPROACH, USING SCIPY.OPTIMIZE.MINIMIZE (SEE IMPORTS).  WHY IT FAILED DUE TO (A) UNDERLYING FUNCTION WAS NOT WELL-DEFINED OUTISDE OF SMALL WINDOW, AS ALL POINTS COULD DISAPPEAR, CAUSING A DIVIDE-BY-ZERO ERROR.  (B) UNDERLYING FUNCTION WAS NOT MONOTONIC -- AS POINTS EXITED IMAGE, DISTANCES COULD GET SMALLER. (C) SCIPY WOULD PRODUCE A LARGE STEP, E.G. DX=1.0 M, WHICH HIT THESE PROBLEMS. (D) NOTE THAT FUNCTION HAD DIFFERENT UNITS -- RMS IS IN PIXEL SPACE, BUT (DX,DY,DTHETA) ARE IN METERS AND RADIANS.
        #
        # &&&& DELETE-SCAFFOLDING -- DISCUSS OPENCV APPROACH, USING cv2.estimateAffinePartial2D (GOOGLE SEARCH "REGISTERING TWO IMAGES TRANSLATION AND ROTATION"). WHY IT WOULD FAIL -- IT SOLVES PROBLEM PURELY IN PIXEL SPACE, BUT (DX,DY,DTHETA) ARE IN 3-D SPACE, WITH DIFFERENT UNITS. THE RESULTING POSE REGISTRATION IN PIXEL SPACE WOULD NOT GIVE A RESULT THAT COULD IMMEDIATELY BE USED.  IT WOULD NEED TO BE CONVERTED TO (DX,DY,DTHETA), WHICH IS NOT STRAIGHTFORWARD.  FURTHER, THE SHAPE OF THE MOVING REPROJECTED IMAGE IS NOT CONSTANT.  DUE TO PARALLAX EFFECTS AND UNEQUAL CAMERA-TO-MIRROR DISTANCES, IT MORPHS SHAPE AS (DX,DY,DTHETA) CHANGES, BUT AN IMAGE REGISTRATION APPRAOCH DOES NOT RECOGNIZE THIS.
        #
        # &&&& DELETE-SCAFFOLDING -- DISCUSS CUSTOM APPROACH, USED HERE. WHY IT WAS PURSUED (SEE ABOVE), AND ITS OWN WEAKNESSES.  IN PARTICULAR, THE UNDERLYING FUNCTION'S NON-MONOTONE BEHAVIOR IS A HAZARD.  THE SMALL-STEP ALGORITHM MAY PROVE VULNERABLE TO THIS, AND THUS IT MAY STILL BE WORTHWHILE TO MAKE THE UNDERLYING FUNCITON MONTONE AND RELIABLE.
        #
        # &&&& DELETE-SCAFFOLDING -- NTOE STILL MIGHT BE A MUCH BETTER SOLUTION. LIST AS A SOFAST ISSUE.
        #
        # We seek the (dx,dy,dtheta) point which shifts the camera to minimize the error between
        # reprojected points and the mask edges.
        #
        # A Golden Section search is tempting (https://en.wikipedia.org/wiki/Golden-section_search),
        # because it is simple, reliable, and robust -- like binary search.  But this will break
        # down when multiple degrees of freedom are considered.  For example, consider a well-behaved
        # problem where a fairly symmetric paraboloid is the shape of the function to find the minimum.
        # Using a single-degree of freedom Golden Section search on x and y won't converge to the
        # minimum, but a secussesive series of alternating degrees of freedom could -- eventually.
        # Adding a third rotational degree of freedom further compliates things.  But reasoning
        # about this search becomes more difficult in challenging cases, such as a search space
        # with a shape that is effectievly similar to a parbolioid with strong astigmatism
        # that is rotated slightly.
        #
        # This is further complicated by the unusual nature of the search space gradients, since
        # the scale factor between pixels and 3-d world motions is an unkown factor which is
        # not even constant across points, due to differences in camera-to-object distance.
        # Thus we'll use a general-purpose minimizaiton approach, which considers all three
        # degrees of freedom (x,y,theta) simultaneously, and does not require a gradient
        # function.  Due to the underlying mathematical complexity, we'll prefer a simple,
        # more robust algorithmic method.
        #
        # # &&&& DELETE-SCAFFOLDING -- SCIPY APPROACH. DEPRECATED (SEE ABOVE).
        # #
        # # See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
        # #
        # # Calculate scale so that align-point to screen matches measurement
        # args = (r_cam_optic_input, v_cam_optic_cam_input, orientation, vxyz_corners_sfc, mask_processed, search_spiral)
        # out = minimize(
        #     self.compute_pose_alignment_error_for_proposed_camera_adjustment, np.array([0.0, 0.0, 0.0]), args=args
        # )
        # optimal_dx_dy_dtheta_array = out.x[0]
        #         # Calculate scale so that align-point to screen matches measurement
        #         args = (
        #             dist_optic_screen,
        #             v_align_point_optic,
        #             v_optic_cam_optic,
        #             v_optic_screen_optic,
        #             v_meas_pts_surf_int_optic,
        #         )
        #         out = minimize(sf2.dist_optic_screen_error, np.array([1.0]), args=args)
        #         scale = out.x[0]

        # # &&&& DELETE-SCAFFOLDING -- MANUAL APPROACH. OBSOLETE.
        # #
        # camera_pose_dx = 0.022
        # camera_pose_dy = 0.0
        # camera_pose_dtheta = np.radians(3.5)
        # # # &&&& DELETE-SCAFFOLDING -- DELETE ONCE TESTED
        # # camera_pose_dx_dy_dtheta_array = np.array([camera_pose_dx, camera_pose_dy, camera_pose_dtheta])
        # camera_pose_dx_dy_dtheta = (camera_pose_dx, camera_pose_dy, camera_pose_dtheta)
        # camera_view_alignment_rms_error = self.compute_pose_alignment_error_for_proposed_camera_adjustment(
        #     camera_pose_dx_dy_dtheta,
        #     r_cam_optic_input,
        #     v_cam_optic_cam_input,
        #     orientation,
        #     vxyz_corners_sfc,
        #     mask_processed,
        #     search_spiral,
        # )
        # return camera_pose_dx_dy_dtheta, camera_view_alignment_rms_error

        current_best_dx = 0.0  # m
        current_best_dy = 0.0  # m
        current_best_dtheta = 0.0  # radians
        current_best_dx_dy_dtheta = (current_best_dx, current_best_dy, current_best_dtheta)
        current_best_rms_error, current_best_dist_optic_screen = (
            self.compute_pose_alignment_error_for_proposed_camera_adjustment(
                current_best_dx_dy_dtheta,
                r_cam_optic_input,
                v_cam_optic_cam_input,
                orientation,
                vxyz_corners_sfc,
                mask_processed,
                search_spiral,
                output_debug_figure=True,  # Draw the initial condition (if debug=True).
            )
        )
        if self.debug.debug_active:
            lt.info(
                f'In optimize_camera_pose_preserving_aim_and_distance(), current_best_dx_dy_dtheta={current_best_dx_dy_dtheta}; current_best_rms_error={current_best_rms_error}pix; current_best_dist_optic_screen={current_best_dist_optic_screen:.4f}'
            )

        # Search for best translation of distance r.
        r = 0.01  # 0.002 # m
        beta_step = np.radians(30.0)
        while True:
            current_best_dx_dy_dtheta, current_best_rms_error, current_best_dist_optic_screen, better_solution_found = (
                self.find_best_camera_translation_of_distance_r(
                    current_best_dx_dy_dtheta,
                    current_best_rms_error,
                    current_best_dist_optic_screen,
                    r,
                    beta_step,
                    r_cam_optic_input,
                    v_cam_optic_cam_input,
                    orientation,
                    vxyz_corners_sfc,
                    mask_processed,
                    search_spiral,
                )
            )
            if self.debug.debug_active:
                lt.info(
                    f'In optimize_camera_pose_preserving_aim_and_distance(), after trans r={r}m, '
                    f'best_dx_dy_dth = ({current_best_dx_dy_dtheta[0]:.3f}, {current_best_dx_dy_dtheta[1]:.3f}, {np.degrees(current_best_dx_dy_dtheta[2]):.4f}, ), '
                    f'best_rms = {current_best_rms_error:.5f}, '
                    f'better_found = {better_solution_found}'
                )
            if not better_solution_found:
                break
        # Debugging output.
        if self.debug.debug_active:
            # Force output of debugging figure, which requires some recomputation.
            # Don't capture the returned values.
            self.compute_pose_alignment_error_for_proposed_camera_adjustment(
                current_best_dx_dy_dtheta,
                r_cam_optic_input,
                v_cam_optic_cam_input,
                orientation,
                vxyz_corners_sfc,
                mask_processed,
                search_spiral,
                output_debug_figure=True,
                debug_figure_status_str=f'r={r}m ',
            )

        # Search for best rotation of step dtheta_step.
        dtheta_step = np.radians(1.0)
        abs_max_delta_dtheta = np.radians(10.0)
        while True:
            current_best_dx_dy_dtheta, current_best_rms_error, current_best_dist_optic_screen, better_solution_found = (
                self.find_best_camera_rotation_of_dtheta_step(
                    current_best_dx_dy_dtheta,
                    current_best_rms_error,
                    current_best_dist_optic_screen,
                    dtheta_step,
                    abs_max_delta_dtheta,
                    r_cam_optic_input,
                    v_cam_optic_cam_input,
                    orientation,
                    vxyz_corners_sfc,
                    mask_processed,
                    search_spiral,
                )
            )
            if self.debug.debug_active:
                lt.info(
                    f'In optimize_camera_pose_preserving_aim_and_distance(), after rot dth_step={np.degrees(dtheta_step)}deg, '
                    f'best_dx_dy_dth = ({current_best_dx_dy_dtheta[0]:.3f}, {current_best_dx_dy_dtheta[1]:.3f}, {np.degrees(current_best_dx_dy_dtheta[2]):.4f}, ), '
                    f'best_rms = {current_best_rms_error:.5f}, '
                    f'better_found = {better_solution_found}'
                )
            if not better_solution_found:
                break
        # Debugging output.
        if self.debug.debug_active:
            # Force output of debugging figure, which requires some recomputation.
            # Don't capture the returned values.
            self.compute_pose_alignment_error_for_proposed_camera_adjustment(
                current_best_dx_dy_dtheta,
                r_cam_optic_input,
                v_cam_optic_cam_input,
                orientation,
                vxyz_corners_sfc,
                mask_processed,
                search_spiral,
                output_debug_figure=True,
                debug_figure_status_str=f'dth_step={np.degrees(dtheta_step)}deg ',
            )

        # Search for best translation of distance r.
        r = 0.005  # 0.002 # m
        beta_step = np.radians(30.0)
        while True:
            current_best_dx_dy_dtheta, current_best_rms_error, current_best_dist_optic_screen, better_solution_found = (
                self.find_best_camera_translation_of_distance_r(
                    current_best_dx_dy_dtheta,
                    current_best_rms_error,
                    current_best_dist_optic_screen,
                    r,
                    beta_step,
                    r_cam_optic_input,
                    v_cam_optic_cam_input,
                    orientation,
                    vxyz_corners_sfc,
                    mask_processed,
                    search_spiral,
                )
            )
            if self.debug.debug_active:
                lt.info(
                    f'In optimize_camera_pose_preserving_aim_and_distance(), after trans r={r}m, '
                    f'best_dx_dy_dth = ({current_best_dx_dy_dtheta[0]:.3f}, {current_best_dx_dy_dtheta[1]:.3f}, {np.degrees(current_best_dx_dy_dtheta[2]):.4f}, ), '
                    f'best_rms = {current_best_rms_error:.5f}, '
                    f'better_found = {better_solution_found}'
                )
            if not better_solution_found:
                break
        # Debugging output.
        if self.debug.debug_active:
            # Force output of debugging figure, which requires some recomputation.
            # Don't capture the returned values.
            self.compute_pose_alignment_error_for_proposed_camera_adjustment(
                current_best_dx_dy_dtheta,
                r_cam_optic_input,
                v_cam_optic_cam_input,
                orientation,
                vxyz_corners_sfc,
                mask_processed,
                search_spiral,
                output_debug_figure=True,
                debug_figure_status_str=f'r={r}m ',
            )

        # Search for best rotation of step dtheta_step.
        dtheta_step = np.radians(0.5)
        abs_max_delta_dtheta = np.radians(2.5)
        while True:
            current_best_dx_dy_dtheta, current_best_rms_error, current_best_dist_optic_screen, better_solution_found = (
                self.find_best_camera_rotation_of_dtheta_step(
                    current_best_dx_dy_dtheta,
                    current_best_rms_error,
                    current_best_dist_optic_screen,
                    dtheta_step,
                    abs_max_delta_dtheta,
                    r_cam_optic_input,
                    v_cam_optic_cam_input,
                    orientation,
                    vxyz_corners_sfc,
                    mask_processed,
                    search_spiral,
                )
            )
            if self.debug.debug_active:
                lt.info(
                    f'In optimize_camera_pose_preserving_aim_and_distance(), after rot dth_step={np.degrees(dtheta_step)}deg, '
                    f'best_dx_dy_dth = ({current_best_dx_dy_dtheta[0]:.3f}, {current_best_dx_dy_dtheta[1]:.3f}, {np.degrees(current_best_dx_dy_dtheta[2]):.4f}, ), '
                    f'best_rms = {current_best_rms_error:.5f}, '
                    f'better_found = {better_solution_found}'
                )
            if not better_solution_found:
                break
        # Debugging output.
        if self.debug.debug_active:
            # Force output of debugging figure, which requires some recomputation.
            # Don't capture the returned values.
            self.compute_pose_alignment_error_for_proposed_camera_adjustment(
                current_best_dx_dy_dtheta,
                r_cam_optic_input,
                v_cam_optic_cam_input,
                orientation,
                vxyz_corners_sfc,
                mask_processed,
                search_spiral,
                output_debug_figure=True,
                debug_figure_status_str=f'dth_step={np.degrees(dtheta_step)}deg ',
            )

        # Search for best translation of distance r.
        r = 0.0025  # 0.002 # m
        beta_step = np.radians(30.0)
        while True:
            current_best_dx_dy_dtheta, current_best_rms_error, current_best_dist_optic_screen, better_solution_found = (
                self.find_best_camera_translation_of_distance_r(
                    current_best_dx_dy_dtheta,
                    current_best_rms_error,
                    current_best_dist_optic_screen,
                    r,
                    beta_step,
                    r_cam_optic_input,
                    v_cam_optic_cam_input,
                    orientation,
                    vxyz_corners_sfc,
                    mask_processed,
                    search_spiral,
                )
            )
            if self.debug.debug_active:
                lt.info(
                    f'In optimize_camera_pose_preserving_aim_and_distance(), after trans r={r}m, '
                    f'best_dx_dy_dth = ({current_best_dx_dy_dtheta[0]:.3f}, {current_best_dx_dy_dtheta[1]:.3f}, {np.degrees(current_best_dx_dy_dtheta[2]):.4f}, ), '
                    f'best_rms = {current_best_rms_error:.5f}, '
                    f'better_found = {better_solution_found}'
                )
            if not better_solution_found:
                break
        # Debugging output.
        if self.debug.debug_active:
            # Force output of debugging figure, which requires some recomputation.
            # Don't capture the returned values.
            self.compute_pose_alignment_error_for_proposed_camera_adjustment(
                current_best_dx_dy_dtheta,
                r_cam_optic_input,
                v_cam_optic_cam_input,
                orientation,
                vxyz_corners_sfc,
                mask_processed,
                search_spiral,
                output_debug_figure=True,
                debug_figure_status_str=f'r={r}m ',
            )

        # Search for best rotation of step dtheta_step.
        dtheta_step = np.radians(0.25)
        abs_max_delta_dtheta = np.radians(1.25)
        while True:
            current_best_dx_dy_dtheta, current_best_rms_error, current_best_dist_optic_screen, better_solution_found = (
                self.find_best_camera_rotation_of_dtheta_step(
                    current_best_dx_dy_dtheta,
                    current_best_rms_error,
                    current_best_dist_optic_screen,
                    dtheta_step,
                    abs_max_delta_dtheta,
                    r_cam_optic_input,
                    v_cam_optic_cam_input,
                    orientation,
                    vxyz_corners_sfc,
                    mask_processed,
                    search_spiral,
                )
            )
            if self.debug.debug_active:
                lt.info(
                    f'In optimize_camera_pose_preserving_aim_and_distance(), after rot dth_step={np.degrees(dtheta_step)}deg, '
                    f'best_dx_dy_dth = ({current_best_dx_dy_dtheta[0]:.3f}, {current_best_dx_dy_dtheta[1]:.3f}, {np.degrees(current_best_dx_dy_dtheta[2]):.4f}, ), '
                    f'best_rms = {current_best_rms_error:.5f}, '
                    f'better_found = {better_solution_found}'
                )
            if not better_solution_found:
                break
        # Debugging output.
        if self.debug.debug_active:
            # Force output of debugging figure, which requires some recomputation.
            # Don't capture the returned values.
            self.compute_pose_alignment_error_for_proposed_camera_adjustment(
                current_best_dx_dy_dtheta,
                r_cam_optic_input,
                v_cam_optic_cam_input,
                orientation,
                vxyz_corners_sfc,
                mask_processed,
                search_spiral,
                output_debug_figure=True,
                debug_figure_status_str=f'dth_step={np.degrees(dtheta_step)}deg ',
            )

        # Search for best translation of distance r.
        r = 0.001  # 0.002 # m
        beta_step = np.radians(30.0)
        while True:
            current_best_dx_dy_dtheta, current_best_rms_error, current_best_dist_optic_screen, better_solution_found = (
                self.find_best_camera_translation_of_distance_r(
                    current_best_dx_dy_dtheta,
                    current_best_rms_error,
                    current_best_dist_optic_screen,
                    r,
                    beta_step,
                    r_cam_optic_input,
                    v_cam_optic_cam_input,
                    orientation,
                    vxyz_corners_sfc,
                    mask_processed,
                    search_spiral,
                )
            )
            if self.debug.debug_active:
                lt.info(
                    f'In optimize_camera_pose_preserving_aim_and_distance(), after trans r={r}m, '
                    f'best_dx_dy_dth = ({current_best_dx_dy_dtheta[0]:.3f}, {current_best_dx_dy_dtheta[1]:.3f}, {np.degrees(current_best_dx_dy_dtheta[2]):.4f}, ), '
                    f'best_rms = {current_best_rms_error:.5f}, '
                    f'better_found = {better_solution_found}'
                )
            if not better_solution_found:
                break
        # Debugging output.
        if self.debug.debug_active:
            # Force output of debugging figure, which requires some recomputation.
            # Don't capture the returned values.
            self.compute_pose_alignment_error_for_proposed_camera_adjustment(
                current_best_dx_dy_dtheta,
                r_cam_optic_input,
                v_cam_optic_cam_input,
                orientation,
                vxyz_corners_sfc,
                mask_processed,
                search_spiral,
                output_debug_figure=True,
                debug_figure_status_str=f'r={r}m ',
            )

        # Search for best rotation of step dtheta_step.
        dtheta_step = np.radians(0.1)
        abs_max_delta_dtheta = np.radians(0.5)
        while True:
            current_best_dx_dy_dtheta, current_best_rms_error, current_best_dist_optic_screen, better_solution_found = (
                self.find_best_camera_rotation_of_dtheta_step(
                    current_best_dx_dy_dtheta,
                    current_best_rms_error,
                    current_best_dist_optic_screen,
                    dtheta_step,
                    abs_max_delta_dtheta,
                    r_cam_optic_input,
                    v_cam_optic_cam_input,
                    orientation,
                    vxyz_corners_sfc,
                    mask_processed,
                    search_spiral,
                )
            )
            if self.debug.debug_active:
                lt.info(
                    f'In optimize_camera_pose_preserving_aim_and_distance(), after rot dth_step={np.degrees(dtheta_step)}deg, '
                    f'best_dx_dy_dth = ({current_best_dx_dy_dtheta[0]:.3f}, {current_best_dx_dy_dtheta[1]:.3f}, {np.degrees(current_best_dx_dy_dtheta[2]):.4f}, ), '
                    f'best_rms = {current_best_rms_error:.5f}, '
                    f'better_found = {better_solution_found}'
                )
            if not better_solution_found:
                break
        # Debugging output.
        if self.debug.debug_active:
            # Force output of debugging figure, which requires some recomputation.
            # Don't capture the returned values.
            self.compute_pose_alignment_error_for_proposed_camera_adjustment(
                current_best_dx_dy_dtheta,
                r_cam_optic_input,
                v_cam_optic_cam_input,
                orientation,
                vxyz_corners_sfc,
                mask_processed,
                search_spiral,
                output_debug_figure=True,
                debug_figure_status_str=f'dth_step={np.degrees(dtheta_step)}deg ',
            )

        # Return.
        if self.debug.debug_active:
            lt.info(
                f'In optimize_camera_pose_preserving_aim_and_distance(), final best_dx_dy_dtheta={current_best_dx_dy_dtheta}; final best_rms_error={current_best_rms_error:.4f}pix'
            )
        return current_best_dx_dy_dtheta, current_best_rms_error, current_best_dist_optic_screen

    def find_best_camera_translation_of_distance_r(
        self,
        current_best_dx_dy_dtheta: tuple[float, float, float],
        current_best_rms_error: float,
        current_best_dist_optic_screen: float,
        r: float,
        beta_step: float,
        r_cam_optic_input: Rotation,
        v_cam_optic_cam_input: Vxyz,
        orientation: SpatialOrientation,
        vxyz_corners_sfc: Vxyz,
        mask_processed: np.ndarray,
        search_spiral: list[tuple[int, int, float]],
    ):
        current_best_dx = current_best_dx_dy_dtheta[0]
        current_best_dy = current_best_dx_dy_dtheta[1]
        current_best_dtheta = current_best_dx_dy_dtheta[2]
        beta = 0.0  # radians
        beta_limit = 2.0 * np.pi
        better_solution_found = False
        while beta < beta_limit:
            delta_dx = r * np.cos(beta)
            delta_dy = r * np.sin(beta)
            delta_dtheta = 0.0
            this_dx = current_best_dx + delta_dx
            this_dy = current_best_dy + delta_dy
            this_dtheta = current_best_dtheta + delta_dtheta
            this_dx_dy_dtheta = (this_dx, this_dy, this_dtheta)
            this_rms_error, this_dist_optic_screen_adjusted = (
                self.compute_pose_alignment_error_for_proposed_camera_adjustment(
                    this_dx_dy_dtheta,
                    r_cam_optic_input,
                    v_cam_optic_cam_input,
                    orientation,
                    vxyz_corners_sfc,
                    mask_processed,
                    search_spiral,
                    output_debug_figure=False,  # Change to True for in-loop progress figures.
                )
            )
            # # Keep for detailed testing.
            # if self.debug.debug_active:
            #     lt.info(
            #         f'In find_best_camera_translation_of_distance_r(), r={r}m; beta={np.degrees(beta):.2f}; this_dx_dy_dtheta={this_dx_dy_dtheta}; this_rms={this_rms_error}pix'
            #     )
            if this_rms_error < current_best_rms_error:
                if better_solution_found == False:
                    # We have a better solution, and we haven't seen another one during this loop.
                    better_dx_dy_dtheta = this_dx_dy_dtheta
                    better_rms_error = this_rms_error
                    better_dist_optic_screen_adjusted = this_dist_optic_screen_adjusted
                    better_solution_found = True
                elif this_rms_error < better_rms_error:
                    # We have a better solution than the better solution previously found in this loop.
                    better_dx_dy_dtheta = this_dx_dy_dtheta
                    better_rms_error = this_rms_error
                    better_dist_optic_screen_adjusted = this_dist_optic_screen_adjusted
                else:
                    # No action required.
                    pass
            # Increment loop variable.
            beta += beta_step
        # If a better solution was found, update our current best.
        # Note that we didn't do this sooner, because the current best was used
        # to compute the relative positions.
        if better_solution_found == True:
            current_best_dx_dy_dtheta = better_dx_dy_dtheta
            current_best_rms_error = better_rms_error
            current_best_dist_optic_screen = better_dist_optic_screen_adjusted

        # Return.
        return current_best_dx_dy_dtheta, current_best_rms_error, current_best_dist_optic_screen, better_solution_found

    def find_best_camera_rotation_of_dtheta_step(
        self,
        current_best_dx_dy_dtheta: tuple[float, float, float],
        current_best_rms_error: float,
        current_best_dist_optic_screen: float,
        dtheta_step: float,
        abs_max_delta_dtheta: float,
        r_cam_optic_input: Rotation,
        v_cam_optic_cam_input: Vxyz,
        orientation: SpatialOrientation,
        vxyz_corners_sfc: Vxyz,
        mask_processed: np.ndarray,
        search_spiral: list[tuple[int, int, float]],
    ):
        current_best_dx = current_best_dx_dy_dtheta[0]
        current_best_dy = current_best_dx_dy_dtheta[1]
        current_best_dtheta = current_best_dx_dy_dtheta[2]

        # Search in positive direction.
        better_solution_found = False
        delta_dtheta = 0.0
        while delta_dtheta < abs_max_delta_dtheta:
            delta_dx = 0.0
            delta_dy = 0.0
            # We started at 0.0, which has already been checked.
            # So increment the loop variable at the top of the loop.
            delta_dtheta += dtheta_step
            this_dx = current_best_dx + delta_dx
            this_dy = current_best_dy + delta_dy
            this_dtheta = current_best_dtheta + delta_dtheta
            this_dx_dy_dtheta = (this_dx, this_dy, this_dtheta)
            this_rms_error, this_dist_optic_screen_adjusted = (
                self.compute_pose_alignment_error_for_proposed_camera_adjustment(
                    this_dx_dy_dtheta,
                    r_cam_optic_input,
                    v_cam_optic_cam_input,
                    orientation,
                    vxyz_corners_sfc,
                    mask_processed,
                    search_spiral,
                    output_debug_figure=False,  # Change to True for in-loop progress figures.
                )
            )
            # # Keep for detailed testing.
            # if self.debug.debug_active:
            #     lt.info(
            #         f'In find_best_camera_rotation_of_dtheta_step(), delta_dth={np.degrees(delta_dtheta):.2f}; this_dx_dy_dth={this_dx_dy_dtheta}; this_rms={this_rms_error}pix'
            #     )
            if this_rms_error < current_best_rms_error:
                if better_solution_found is False:
                    # We have a better solution, and we haven't seen another one during this loop.
                    better_dx_dy_dtheta = this_dx_dy_dtheta
                    better_rms_error = this_rms_error
                    better_dist_optic_screen_adjusted = this_dist_optic_screen_adjusted
                    better_solution_found = True
                elif this_rms_error < better_rms_error:
                    # We have a better solution than the better solution previously found in this loop.
                    better_dx_dy_dtheta = this_dx_dy_dtheta
                    better_rms_error = this_rms_error
                    better_dist_optic_screen_adjusted = this_dist_optic_screen_adjusted
                else:
                    # Our rms error function has some noise, due to the disappearance and
                    # appearance of matching points.  So even if we have found a better
                    # solution, we'll keep going until we hit the search window limit.
                    pass
        # If a better solution was found, update our current best.
        # Note that we didn't do this sooner, because the current best was used
        # to compute the relative positions.
        if better_solution_found is True:
            current_best_dx_dy_dtheta = better_dx_dy_dtheta
            current_best_rms_error = better_rms_error
            current_best_dist_optic_screen = better_dist_optic_screen_adjusted

        # Search in negative direction.
        delta_dtheta = 0.0
        while delta_dtheta > -abs_max_delta_dtheta:
            delta_dx = 0.0
            delta_dy = 0.0
            # We started at 0.0, which has already been checked.
            # So increment the loop variable at the top of the loop.
            delta_dtheta -= dtheta_step
            this_dx = current_best_dx + delta_dx
            this_dy = current_best_dy + delta_dy
            this_dtheta = current_best_dtheta + delta_dtheta
            this_dx_dy_dtheta = (this_dx, this_dy, this_dtheta)
            this_rms_error, this_dist_optic_screen_adjusted = (
                self.compute_pose_alignment_error_for_proposed_camera_adjustment(
                    this_dx_dy_dtheta,
                    r_cam_optic_input,
                    v_cam_optic_cam_input,
                    orientation,
                    vxyz_corners_sfc,
                    mask_processed,
                    search_spiral,
                    output_debug_figure=False,  # Change to True for in-loop progress figures.
                )
            )
            # # Keep for detailed testing.
            # if self.debug.debug_active:
            #     lt.info(
            #         f'In find_best_camera_rotation_of_dtheta_step(), delta_dth={np.degrees(delta_dtheta):.2f}; this_dx_dy_dth={this_dx_dy_dtheta}; this_rms={this_rms_error}pix'
            #     )
            if this_rms_error < current_best_rms_error:
                if better_solution_found is False:
                    # We have a better solution, and we haven't seen another one during this loop.
                    better_dx_dy_dtheta = this_dx_dy_dtheta
                    better_rms_error = this_rms_error
                    better_dist_optic_screen_adjusted = this_dist_optic_screen_adjusted
                    better_solution_found = True
                elif this_rms_error < better_rms_error:
                    # We have a better solution than the better solution previously found in this loop.
                    better_dx_dy_dtheta = this_dx_dy_dtheta
                    better_rms_error = this_rms_error
                    better_dist_optic_screen_adjusted = this_dist_optic_screen_adjusted
                else:
                    # Our rms error function has some noise, due to the disappearance and
                    # appearance of matching points.  So even if we have found a better
                    # solution, we'll keep going until we hit the search window limit.
                    pass
        # If a better solution was found, update our current best.
        # Note that we didn't do this sooner, because the current best was used
        # to compute the relative positions.
        if better_solution_found == True:
            current_best_dx_dy_dtheta = better_dx_dy_dtheta
            current_best_rms_error = better_rms_error
            current_best_dist_optic_screen = better_dist_optic_screen_adjusted

        # Return.
        return current_best_dx_dy_dtheta, current_best_rms_error, current_best_dist_optic_screen, better_solution_found

    def find_best_camera_pose_preserving_aim_and_distance(
        self,
        r_cam_optic_entering_loop: Rotation,
        v_cam_optic_cam_entering_loop: Vxyz,
        original_orientation: SpatialOrientation,
        vxyz_corners_sfc: Vxyz,
        mask_processed: np.ndarray,
        search_spiral: list[tuple[int, int, float]],
    ):
        optimized_dx_dy_dtheta, camera_view_alignment_rms_error, camera_view_dist_optic_screen = (
            self.optimize_camera_pose_preserving_aim_and_distance(
                r_cam_optic_entering_loop,
                v_cam_optic_cam_entering_loop,
                original_orientation,
                vxyz_corners_sfc,
                mask_processed,
                search_spiral,
            )
        )

        r_cam_optic_adjusted, v_cam_optic_cam_adjusted, dist_optic_screen_adjusted = (
            self.adjust_camera_pose_preserving_aim_and_distance(
                optimized_dx_dy_dtheta, r_cam_optic_entering_loop, v_cam_optic_cam_entering_loop, original_orientation
            )
        )
        # Check that these are the same.
        if camera_view_dist_optic_screen != dist_optic_screen_adjusted:
            lt.error(
                "In find_best_camera_pose_preserving_aim_and_distance(), camera_view_dist_optic_screen={camera_view_dist_optic_screen} and dist_optic_screen_adjusted={dist_optic_screen_adjusted} are not equal."
            )

        # Return.
        return (
            r_cam_optic_adjusted,
            v_cam_optic_cam_adjusted,
            camera_view_alignment_rms_error,
            camera_view_dist_optic_screen,
        )
