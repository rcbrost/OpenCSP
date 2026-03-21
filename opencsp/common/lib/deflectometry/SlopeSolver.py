import copy
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

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
        self.v_screen_points_facet = v_screen_points_facet
        self.v_optic_screen_optic = v_optic_screen_optic
        self.v_align_point_optic = v_align_point_optic
        self.dist_optic_screen = dist_optic_screen
        self.debug = debug

        # Load initialization data in surface fit object
        self.surface.set_spatial_data(
            u_active_pixel_pointing_optic,
            v_screen_points_facet,
            v_optic_cam_optic,
            u_measure_pixel_pointing_optic,
            v_align_point_optic,
            v_optic_screen_optic,
        )

        self._data = SlopeSolverData()

    def get_data(self) -> SlopeSolverData:
        """Returns data output object"""
        return self._data

    def fit_surface(self) -> None:
        """
        Performs the initial fine-tuning alignment of the facet, screen, and
        camera. Fits a surface to the calculated slope data.

        """
        # Gather inputs
        v_optic_cam_optic = self.v_optic_cam_optic
        u_measure_pixel_pointing_optic = self.u_measure_pixel_pointing_optic
        v_optic_screen_optic = self.v_optic_screen_optic
        v_align_point_optic = self.v_align_point_optic
        dist_optic_screen = self.dist_optic_screen

        # Instantiate alignment transform
        trans_align = TransformXYZ.from_zero_zero()

        # &&&& DELETE-SCAFFOLDING -- TEMPORARY PASS-THROUGH HACK
        for idx1 in range(4):
            # for idx1 in range(1):  # &&&& DELETE-SCAFFOLDING -- TEMPORARY PASS-THROUGH HACK
            # &&&& DELETE-SCAFFOLDING -- TEMPORARY PASS-THROUGH HACK
            for idx2 in range(3):
                # for idx2 in range(1):  # &&&& DELETE-SCAFFOLDING -- TEMPORARY PASS-THROUGH HACK
                # Calculate surface intersection points
                self.surface.calculate_surface_intersect_points()

                # Check for invalid points
                num_nans = np.isnan(self.surface.v_surf_int_pts_optic.data)
                if np.any(num_nans):
                    warnings.warn(
                        f"{num_nans.sum():d} / {num_nans.size:d} values are NANs in surface intersection points in iteration: ({idx1:d}, {idx2:d}).",
                        stacklevel=2,
                    )

                # Calculate measurement point slopes
                self.surface.calculate_slopes()

                # Check for invalid points
                num_nans = np.isnan(self.surface.slopes)
                if np.any(num_nans):
                    warnings.warn(
                        f"{num_nans.sum():d} / {num_nans.size:d} values are NANs in slope data in iteration: ({idx1:d}, {idx2:d}).",
                        stacklevel=2,
                    )

                # # Plot debug plot
                # if self.debug.debug_active:
                #     self._plot_debug_plots("Before Slope Fit", idx1, idx2)
                #     # self._plot_debug_plots("Before Slope Fit", idx1, idx2, az_el_roll_deg=(0, 0, 0))
                #     # self._plot_debug_plots("Before Slope Fit", idx1, idx2, az_el_roll_deg=(-90, 0, 0))
                #     # self._plot_debug_plots("Before Slope Fit", idx1, idx2, az_el_roll_deg=(0, 90, 90))

                # Update slope fit
                self.surface.fit_slopes()

                # Plot debug plot
                if self.debug.debug_active:
                    self._plot_debug_plots("After Slope Fit", idx1, idx2)
                    self._plot_debug_plots("After Slope Fit", idx1, idx2, az_el_roll_deg=(0, 0, 0))
                    self._plot_debug_plots("After Slope Fit", idx1, idx2, az_el_roll_deg=(-90, 0, 0))
                    self._plot_debug_plots("After Slope Fit", idx1, idx2, az_el_roll_deg=(0, 90, 90))

            # Calculate measure point intersection point with existing fitting function
            v_meas_pts_surf_int_optic = self.surface.intersect(u_measure_pixel_pointing_optic, v_optic_cam_optic)

            # Calculate design normal at alignment point
            n_design = self.surface.normal_design_at_align_point()

            # Calculate measured normal at alignment point
            n_meas = self.surface.normal_fit_at_align_point()

            # Calculate the rotation needed to align the normal vectors
            r_align_step = n_meas.align_to(n_design)

            # Rotate all points about alignment point
            self.surface.rotate_all(r_align_step)

            # # Plot debug plot
            # if self.debug.debug_active:
            #     self._plot_debug_plots("After Rotate All", idx1, idx2)
            #     self._plot_debug_plots("After Rotate All", idx1, idx2, az_el_roll_deg=(0, 0, 0))
            #     self._plot_debug_plots("After Rotate All", idx1, idx2, az_el_roll_deg=(-90, 0, 0))
            #     self._plot_debug_plots("After Rotate All", idx1, idx2, az_el_roll_deg=(0, 90, 90))

            # Calculate scale so that align-point to screen matches measurement
            args = (
                dist_optic_screen,
                v_align_point_optic,
                v_optic_cam_optic,
                v_optic_screen_optic,
                v_meas_pts_surf_int_optic,
            )
            out = minimize(sf2.dist_optic_screen_error, np.array([1.0]), args=args)
            scale = out.x[0]
            v_align_optic_step = (v_optic_cam_optic - v_align_point_optic) * (scale - 1)

            # Shift all points along align-point to camera axis
            self.surface.shift_all(v_align_optic_step)

            # # Plot debug plot
            # if self.debug.debug_active:
            #     self._plot_debug_plots("After Shift All", idx1, idx2)
            #     # self._plot_debug_plots("After Shift All", idx1, idx2, az_el_roll_deg=(0, 0, 0))
            #     # self._plot_debug_plots("After Shift All", idx1, idx2, az_el_roll_deg=(-90, 0, 0))
            #     # self._plot_debug_plots("After Shift All", idx1, idx2, az_el_roll_deg=(0, 90, 90))

            # Calculate alignment transform
            trans_step = TransformXYZ.from_R_V(r_align_step, v_align_optic_step)
            trans_align = trans_step * trans_align

        # Store alignment parameters
        self._data.surf_coefs_facet = self.surface.surf_coefs
        self._data.slope_coefs_facet = self.surface.slope_coefs
        # &&&& DELETE-SCAFFOLDING -- TEMPORARY PASS-THROUGH HACK
        self._data.trans_alignment = trans_align
        # self._data.trans_alignment = TransformXYZ.identity()  # &&&& DELETE-SCAFFOLDING -- TEMPORARY PASS-THROUGH HACK

    # &&&& DELETE-SCAFFOLDING -- BEGIN NEW SOLVER ALGORITHM

    def fit_surface_2(self, original_orientation: SpatialOrientation) -> None:
        """
        # &&&& DELETE-SCAFFOLDING -- FIX THIS DOCSTRING
        Performs the initial fine-tuning alignment of the facet, screen, and
        camera. Fits a surface to the calculated slope data.

        """
        # &&&& DELETE-SCAFFOLDING -- MOVE THIS LINE TO A BETTER LOCATION.  OR, BETTER YET, PASS IN TO MAKE EXPLICIT?
        v_facet_corners_hires_0 = self.debug.debug_geometry.v_facet_corners_hires

        # Set the fine facet boundary z values to lie on the fit surface.
        z_facet_corners_hires_1 = sf2.coef_to_points(v_facet_corners_hires_0, self.surface.surf_coefs, 2)
        v_facet_corners_hires_1 = copy.deepcopy(v_facet_corners_hires_0)
        v_facet_corners_hires_1.data[2, :] = z_facet_corners_hires_1
        # Check fine facet boundary z values.
        v_facet_corners_hires_1_minus_0 = v_facet_corners_hires_1 - v_facet_corners_hires_0
        worst_error = abs(v_facet_corners_hires_1_minus_0.data[2, :]).max()
        if worst_error > 1e-5:
            lt.error_and_raise(
                ValueError,
                # &&&& DELETE-SCAFFOLDING -- FIX CONTAINING ROUTINE NAME IN MESSAGE BELOW
                f"ERROR: In fit_surface_2(), error check of high-resolution z corners yielded a z discrepancy of {worst_error} meters.",
            )

        # Instantiate alignment transform
        trans_align = TransformXYZ.from_zero_zero()

        # &&&& DELETE-SCAFFOLDING -- PASS THIS IN
        mask_processed = self.debug.debug_geometry.mask_processed

        # Construct search spiral for snapping boundary points to nearest edge.
        search_spiral = ip.construct_search_spiral(max_radius=50)
        # &&&& DELETE-SCAFFOLDING -- TEMPORARY
        print('In fit_surface_2(), spiral complete.  Spiral length=', len(search_spiral))

        # &&&& DELETE-SCAFFOLDING -- TEMPORARY PASS-THROUGH HACK
        # for idx1 in range(4):
        for idx1 in range(1):  # &&&& DELETE-SCAFFOLDING -- TEMPORARY PASS-THROUGH HACK
            # &&&& DELETE-SCAFFOLDING -- TEMPORARY PASS-THROUGH HACK
            # for idx2 in range(3):
            for idx2 in range(1):  # &&&& DELETE-SCAFFOLDING -- TEMPORARY PASS-THROUGH HACK
                # &&&& DELETE-SCAFFOLDING -- RELOCATE, COLLAPSE
                # Points reprojected using camera pose from solvePnP() and then refined distance.
                pts_reproj = self.debug.debug_geometry.camera.project(
                    self.debug.debug_geometry.v_facet_corners,
                    self.debug.debug_geometry.r_cam_optic_refine_1.inv(),
                    self.debug.debug_geometry.v_cam_optic_cam_refine_2,
                )

                # High-resolution points reprojected.
                hires_pts_reproj_1 = self.debug.debug_geometry.camera.project(
                    v_facet_corners_hires_1,
                    self.debug.debug_geometry.r_cam_optic_refine_1.inv(),
                    self.debug.debug_geometry.v_cam_optic_cam_refine_2,
                )

                # Snap reprojected high-resolution points to nearest image edge.
                reproj_pt_snapped_pt_pair_list = ip.snap_points_to_nearest_edge(
                    image_points=hires_pts_reproj_1, mask=mask_processed, search_spiral=search_spiral
                )
                matched_hires_pts_reproj_1 = Vxy.from_list([a[0] for a in reproj_pt_snapped_pt_pair_list])
                hires_pts_reproj_snap_1 = Vxy.from_list([a[1] for a in reproj_pt_snapped_pt_pair_list])

                # Plot reprojected points over mask image
                if self.debug.debug_active:
                    ssdo.reproj_snap(pts_reproj, matched_hires_pts_reproj_1, hires_pts_reproj_snap_1, debug=self.debug)

                # Compute refined camera pose using high-resolution points.
                r_optic_cam_refine_3, v_cam_optic_cam_refine_3 = sp.calc_rt_from_img_pts(
                    matched_hires_pts_reproj_1,
                    v_facet_corners_hires_1,
                    self.debug.debug_geometry.camera,
                    initial_rotation=self.debug.debug_geometry.r_cam_optic_refine_1,
                    initial_vxyz=self.debug.debug_geometry.v_cam_optic_cam_refine_2,
                )

                # &&&& DELETE-SCAFFOLDING -- SHOULD THIS NAME BE INVERTED?
                r_cam_optic_refine_3 = r_optic_cam_refine_3.inv()

                # High-resolution points reprojected (again).
                hires_pts_reproj_3 = self.debug.debug_geometry.camera.project(
                    v_facet_corners_hires_1, r_cam_optic_refine_3.inv(), v_cam_optic_cam_refine_3
                )

                # Plot reprojected points over mask image
                if self.debug.debug_active:
                    ssdo.reproj_after_snap_snap(hires_pts_reproj_snap_1, hires_pts_reproj_3, self.debug)

                # Calculate surface intersection points
                # Calculate pixel intersection points with existing fitting function
                ori_3 = copy.copy(original_orientation)
                # Orient optic
                ori_3.orient_optic_cam(r_cam_optic_refine_3, v_cam_optic_cam_refine_3)

                # Calculate pixel pointing directions (camera coordinates)
                camera = self.debug.debug_geometry.camera  # &&&& DELETE-SCAFFOLDING -- PASS THIS IN
                u_pixel_pointing_cam_3 = ip.calculate_active_pixels_vectors(mask_processed, camera)
                # Convert to optic coordinates
                u_active_pixel_pointing_optic_3_pre = u_pixel_pointing_cam_3.rotate(ori_3.r_cam_optic)
                # Downsample measurement data
                downsample_3 = self.surface.downsample
                u_active_pixel_pointing_optic_3 = u_active_pixel_pointing_optic_3_pre[::downsample_3]
                v_optic_cam_optic_refine_3 = ori_3.v_optic_cam_optic
                # Project camera pixel rays and intersect with fit surface
                self.surface.v_surf_int_pts_optic = self.surface.intersect(
                    u_active_pixel_pointing_optic_3, v_optic_cam_optic_refine_3
                )

                # &&&& DELETE-SCAFFOLDING -- NORMAL COMPUTATION BELOW
                # =====================================================================================================================================
                # &&&& DELETE-SCAFFOLDING -- NORMAL COMPUTATION BELOW

                # Plot debug plot
                if self.debug.debug_active:
                    ssdo.figure_intersection_surface_situation(
                        "After Calculate Intersections",
                        v_facet_corners_hires_1,
                        None,
                        self.surface,
                        idx1,
                        idx2,
                        self.debug,
                    )

                # Check for invalid points
                num_nans = np.isnan(self.surface.v_surf_int_pts_optic.data)
                if np.any(num_nans):
                    warnings.warn(
                        f"{num_nans.sum():d} / {num_nans.size:d} values are NANs in surface intersection points in iteration: ({idx1:d}, {idx2:d}).",
                        stacklevel=2,
                    )

                # Calculate measurement point slopes
                self.surface.calculate_slopes()

                # Check for invalid points
                num_nans = np.isnan(self.surface.slopes)
                if np.any(num_nans):
                    warnings.warn(
                        f"{num_nans.sum():d} / {num_nans.size:d} values are NANs in slope data in iteration: ({idx1:d}, {idx2:d}).",
                        stacklevel=2,
                    )

                # Update slope fit
                self.surface.fit_slopes()

                # Set the fine facet boundary z values to lie on the new fit surface.
                z_facet_corners_hires_4 = sf2.coef_to_points(v_facet_corners_hires_1, self.surface.surf_coefs, 2)
                v_facet_corners_hires_4 = copy.deepcopy(v_facet_corners_hires_1)
                v_facet_corners_hires_4.data[2, :] = z_facet_corners_hires_4
                # Check fine facet boundary z values.
                v_facet_corners_hires_4_minus_1 = v_facet_corners_hires_4 - v_facet_corners_hires_1
                largest_change = abs(v_facet_corners_hires_4_minus_1.data[2, :]).max()
                # &&&& DELETE-SCAFFOLDING -- FIX CONTAINING ROUTINE NAME IN MESSAGE BELOW
                lt.info(
                    f"In fit_surface_2(), maximum z difference between high-resolution corners before and after fit = {largest_change} m."
                )

                # High-resolution points reprojected (after snap to new fit surface).
                hires_pts_reproj_4 = self.debug.debug_geometry.camera.project(
                    v_facet_corners_hires_4, r_cam_optic_refine_3.inv(), v_cam_optic_cam_refine_3
                )

                # Plot reprojected points over mask image
                if self.debug.debug_active:
                    ssdo.reproj_after_fit(hires_pts_reproj_snap_1, hires_pts_reproj_3, hires_pts_reproj_4, self.debug)

                # Plot debug plot
                if self.debug.debug_active:
                    ssdo.figure_intersection_surface_situation(
                        "After Slope Fit",
                        v_facet_corners_hires_1,
                        v_facet_corners_hires_4,
                        self.surface,
                        idx1,
                        idx2,
                        self.debug,
                    )

        # Store alignment parameters
        self._data.surf_coefs_facet = self.surface.surf_coefs
        self._data.slope_coefs_facet = self.surface.slope_coefs
        # &&&& DELETE-SCAFFOLDING -- TEMPORARY PASS-THROUGH HACK
        self._data.trans_alignment = trans_align
        # self._data.trans_alignment = TransformXYZ.identity()  # &&&& DELETE-SCAFFOLDING -- TEMPORARY PASS-THROUGH HACK

    # &&&& DELETE-SCAFFOLDING -- END NEW SOLVER ALGORITHM

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
            raise ValueError("Initial alignment needs to be completed before final slope fitting (self.fit_surface).")

        # Apply alignment transforms about alignment point
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


# # &&&& DELETE-SCAFFOLDING -- DELETE THE FOLLOWING COMMENT BLOCK -- OBSOLETE
#     # &&&& DELETE-SCAFFOLDING -- CONSOLIDATE WITH ABOVE AND REDUCE CODE REPLICATION
#     def _plot_debug_plots_2(
#         self,
#         title: str,
#         v_facet_corners_hires: Vxyz,
#         idx1: int,
#         idx2: int,
#         view_spec: dict = vs.view_spec_3d(),
#         az_el_roll_deg: tuple[float, float, float] = None,
#     ):
#         # &&&& DELETE-SCAFFOLDING -- DELETE THE FOLLOWING COMMENT BLOCK -- OBSOLETE
#         # # Create figure and axes
#         # if self.debug.slope_solver_single_plot and isinstance(self.debug.slope_solver_figures, list):
#         #     # Create first figure if needed
#         #     fig = plt.figure()
#         #     axes = fig.add_subplot(projection="3d")
#         #     self.debug.slope_solver_figures = fig
#         #     # Plot facet corners
#         #     facet_outline = self.debug.optic_data.v_facet_corners.data
#         #     axes.scatter(*facet_outline, color="k")
#         #     # Format
#         #     axes.set_title("Slope Solver: " + title)
#         # elif self.debug.slope_solver_single_plot:
#         #     # Get axes for single plot
#         #     axes = self.debug.slope_solver_figures.gca()
#         # else:
#         #     # Create a new figure
#         #     fig = plt.figure(figsize=(12, 9))
#         #     axes = fig.add_subplot(projection="3d", proj_type='ortho')
#         #     self.debug.slope_solver_figures.append(fig)
#         #     # Plot facet corners
#         #     facet_outline = self.debug.optic_data.v_facet_corners.data
#         #     axes.scatter(
#         #         *facet_outline, color="lightgreen", label="Facet Vertices"
#         #     )  # &&&& DELETE-SCAFFOLDING -- COLOR WAS "k"
#         #     # Plot high-resolution facet corners
#         #     facet_outline = v_facet_corners_hires.data
#         #     axes.scatter(*facet_outline, s=5, color="blue", label="Facet Vertices (High Resolution)")
#         #     # Format
#         #     axes.set_title(f"Slope Solver ({idx1:d}, {idx2:d}): " + title)

#         # Create a new figure
#         full_title = f"Slope Solver ({idx1:d}, {idx2:d}): " + title
#         fig_rec = sdfs.start_debug_3d_figure(figure_title=full_title, view_spec=view_spec, figsize=(12, 9))

#         # # Plot original facet corners
#         # Use draw_line(), because we only want one legend entry, not a legend entry for every point.
#         self.debug.optic_data.v_facet_corners.draw_line(
#             fig_rec, style=rcps.marker(marker='o', color='lightgreen', markersize=5), label="Facet Vertices"
#         )

#         # Plot high-resolution facet corners.
#         # Use draw_line(), because we only want one legend entry, not a legend entry for every point.
#         v_facet_corners_hires.draw_line(
#             fig_rec, style=rcps.marker(marker='.', color='b', markersize=4), label="Facet Vertices (High Resolution)"
#         )

#         # Set view direction, if desired.
#         if az_el_roll_deg is not None:
#             if not fig_rec.view.is_3d():
#                 lt.error_and_raise(
#                     ValueError, "In SlopeSolver._plot_debug_plots_2(), asked to set view direction for a non-3d plot."
#                 )
#             azimuth_deg = az_el_roll_deg[0]
#             elevation_deg = az_el_roll_deg[1]
#             roll_deg = az_el_roll_deg[2]
#             lt.info(
#                 'In SlopeSolver._plot_debug_plots_2(), setting view (azimuth, elevation, roll) to '
#                 + str((azimuth_deg, elevation_deg, roll_deg))
#                 + ' degrees.'
#             )
#             fig_rec.view.axis.view_init(azim=azimuth_deg, elev=elevation_deg, roll=roll_deg)

#         # Plot intersection points.
#         # The trisurf plot is only supported for 3-d axes.
#         if fig_rec.view.is_3d():
#             self.surface.plot_intersection_points(
#                 fig_rec.view.axis,
#                 self.debug.slope_solver_point_downsample,
#                 self.debug.slope_solver_camera_rays_length,
#                 self.debug.slope_solver_plot_camera_screen_points,
#             )

#         # Add legend.
#         plt.legend()

#         # Save and close.
#         full_title_for_file = (
#             full_title.replace(' ', '_').replace(':', '').replace('(', '').replace(')', '').replace(',', '')
#         )
#         sdfs.finish_debug_3d_figure(full_title_for_file, 'solver', fig_rec, self.debug.debug_geometry)

# # &&&& DELETE-SCAFFOLDING -- DELETE THE FOLLOWING COMMENT BLOCK -- OBSOLETE
#     # &&&& DELETE-SCAFFOLDING -- CONSOLIDATE WITH ABOVE AND REDUCE CODE REPLICATION
#     def _plot_debug_plots_3(
#         self,
#         title: str,
#         v_facet_corners_hires_1: Vxyz,
#         v_facet_corners_hires_2: Vxyz,
#         idx1: int,
#         idx2: int,
#         az_el_roll_deg: tuple[float, float, float] = None,
#     ):
#         # Create figure and axes
#         if self.debug.slope_solver_single_plot and isinstance(self.debug.slope_solver_figures, list):
#             # Create first figure if needed
#             fig = plt.figure()
#             axes = fig.add_subplot(projection="3d")
#             self.debug.slope_solver_figures = fig
#             # Plot facet corners
#             facet_outline = self.debug.optic_data.v_facet_corners.data
#             axes.scatter(*facet_outline, color="k")
#             # Format
#             axes.set_title("Slope Solver: " + title)
#         elif self.debug.slope_solver_single_plot:
#             # Get axes for single plot
#             axes = self.debug.slope_solver_figures.gca()
#         else:
#             # Create a new figure
#             fig = plt.figure(figsize=(12, 9))
#             axes = fig.add_subplot(projection="3d", proj_type='ortho')
#             self.debug.slope_solver_figures.append(fig)
#             # Plot facet corners
#             facet_outline = self.debug.optic_data.v_facet_corners.data
#             axes.scatter(
#                 *facet_outline, color="lightgreen", label="Facet Vertices"
#             )  # &&&& DELETE-SCAFFOLDING -- COLOR WAS "k"
#             # Plot high-resolution facet corners
#             facet_outline = v_facet_corners_hires_1.data
#             axes.scatter(*facet_outline, s=6, color="blue", label="Hires Vertices (Before Fit)")
#             # Plot high-resolution facet corners
#             facet_outline = v_facet_corners_hires_2.data
#             axes.scatter(*facet_outline, s=3, color="red", label="Hires Vertices (After Fit)")
#             # Format
#             axes.set_title(f"Slope Solver ({idx1:d}, {idx2:d}): " + title)

#         # Set view direction, if desired.
#         if az_el_roll_deg is not None:
#             azimuth_deg = az_el_roll_deg[0]
#             elevation_deg = az_el_roll_deg[1]
#             roll_deg = az_el_roll_deg[2]
#             axes.view_init(azim=azimuth_deg, elev=elevation_deg, roll=roll_deg)

#         # Plot intersection points
#         self.surface.plot_intersection_points(
#             axes,
#             self.debug.slope_solver_point_downsample,
#             self.debug.slope_solver_camera_rays_length,
#             self.debug.slope_solver_plot_camera_screen_points,
#         )

#         # Add legend
#         plt.legend()  # &&&& DELETE-SCAFFOLDING -- NEW.  KEEP?
