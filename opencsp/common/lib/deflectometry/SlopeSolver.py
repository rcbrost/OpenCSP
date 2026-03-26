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

    # &&&& DELETE-SCAFFOLDING -- BEGIN NEW SOLVER ALGORITHM

    def fit_surface_2(self, original_orientation: SpatialOrientation) -> None:
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
        print('In fit_surface_2(), spiral complete.  Spiral length=', len(search_spiral))

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
            "surf_coefs": self.surface.surf_coefs,
            "slope_coefs": self.surface.slope_coefs,
            "r_cam_optic": r_cam_optic_entering_loop,
            "v_cam_optic_cam": v_cam_optic_cam_entering_loop,
            "vxyz_corners": vxyz_corners_entering_loop,
            "vxyz_corner_change": (vxyz_corners_entering_loop - vxyz_corners_entering_loop),  # Zero change.
            "n_intersect": 0,
            "u_avg_fit_normal": Uxyz([0, 0, 1]),
            "u_avg_measured_normal": Uxyz([0, 0, 1]),
            "u_avg_measured_minus_fit": Uxyz([0, 0, 1]),
        }
        loop_record_list = [loop_record_0]

        # Main loop.
        first_time = True
        while True:
            loop_idx += 1
            loop_record = {"loop_idx": loop_idx}
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
            vxyz_corners_sfc_matched = Vxyz.from_list([a[0] for a in sfc_pt_reproj_pt_snapped_pt_list])
            vxy_corners_sfc_reproj_matched = Vxy.from_list([a[1] for a in sfc_pt_reproj_pt_snapped_pt_list])
            vxy_corners_sfc_reproj_snap = Vxy.from_list([a[2] for a in sfc_pt_reproj_pt_snapped_pt_list])
            # Plot reprojected points over mask image
            if self.debug.debug_active:
                ssdo.reproj_snap_2(
                    vxy_corners_sfc_reproj_matched, vxy_corners_sfc_reproj_snap, loop_idx=loop_idx, debug=self.debug
                )

            # 4. Use refined image points to call solvePnP() and compute a refined camera POSE'.
            r_optic_cam_new, v_cam_optic_cam_new = sp.calc_rt_from_img_pts(
                vxy_corners_sfc_reproj_snap,
                vxyz_corners_sfc_matched,
                camera,
                initial_rotation=r_cam_optic_entering_loop,
                initial_vxyz=v_cam_optic_cam_entering_loop,
            )
            # &&&& DELETE-SCAFFOLDING -- SHOULD THIS NAME BE INVERTED?
            r_cam_optic_new = r_optic_cam_new.inv()
            # Orient optic
            # &&&& DELETE-SCAFFOLDING -- TEMPORARY, OR DOCUMENT
            if first_time:
                ori_new = copy.copy(original_orientation)
                ori_new.orient_optic_cam(r_cam_optic_new, v_cam_optic_cam_new)
                first_time = False
            loop_record["r_cam_optic"] = ori_new.r_cam_optic
            loop_record["v_cam_optic_cam"] = ori_new.v_cam_optic_cam
            # Plot reprojected points over mask image
            # &&&& DELETE-SCAFFOLDING -- MOVE REPROJECTION INTO DEBUG FIGURE ROUTINE
            vxy_corners_sfc_reproj_new = self.debug.debug_geometry.camera.project(
                vxyz_corners_sfc, ori_new.r_cam_optic.inv(), ori_new.v_cam_optic_cam
            )
            if self.debug.debug_active:
                ssdo.reproj_after_snap_snap(
                    vxy_corners_sfc_reproj_snap, vxy_corners_sfc_reproj_new, loop_idx, self.debug
                )

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
            u_measure_pixel_pointing_optic_new = "NOT UPDATED"
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
            print("\n\nIn fit_surface_2():")
            print("   normal_design_at_align_point=", self.surface.normal_design_at_align_point().to_str())
            print("   normal_fit_at_align_point=", self.surface.normal_fit_at_align_point().to_str())
            print("   u_avg_fit_normal=", loop_record["u_avg_fit_normal"])
            print("   u_avg_measured_normal=", loop_record["u_avg_measured_normal"])
            print("   Delta: measured minus fit=", loop_record["u_avg_measured_minus_fit"])
            print("\n\n")

            # 7. Using surface normals at points, compute regression fit for slope coefficients.
            # 8. Convert fit slope coefficients to new surface COEFFS' = c0', c1x', c2x2', c3y', c4xy', c5y2'.
            # &&&& DELETE-SCAFFOLDING -- TEMPORARY TURN OFF
            self.surface.fit_slopes()
            loop_record["surf_coefs"] = self.surface.surf_coefs
            loop_record["slope_coefs"] = self.surface.slope_coefs

            # Set the fine facet boundary z values to lie on the new fit surface.
            z_facet_corners_hires_4 = sf2.coef_to_points(vxyz_corners_sfc, self.surface.surf_coefs, 2)
            v_facet_corners_hires_4 = copy.deepcopy(vxyz_corners_sfc)
            v_facet_corners_hires_4.data[2, :] = z_facet_corners_hires_4
            # Check fine facet boundary z values.
            v_facet_corners_hires_4_minus_1 = v_facet_corners_hires_4 - vxyz_corners_sfc
            largest_change = abs(v_facet_corners_hires_4_minus_1.data[2, :]).max()
            # &&&& DELETE-SCAFFOLDING -- FIX CONTAINING ROUTINE NAME IN MESSAGE BELOW
            lt.info(
                f"In fit_surface_2(), maximum z difference between high-resolution corners before and after fit = {largest_change} m."
            )

            # High-resolution points reprojected (after snap to new fit surface).
            # &&&& DELETE-SCAFFOLDING -- MOVE REPROJECTION INTO DEBUG FIGURE ROUTINE
            hires_pts_reproj_4 = self.debug.debug_geometry.camera.project(
                v_facet_corners_hires_4, ori_new.r_cam_optic.inv(), ori_new.v_cam_optic_cam
            )
            # Plot reprojected points over mask image
            if self.debug.debug_active:
                ssdo.reproj_after_fit(
                    vxy_corners_sfc_reproj_snap, vxy_corners_sfc_reproj_new, hires_pts_reproj_4, loop_idx, self.debug
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
                lt.info("\nIn fit_surface_2(), loop_record_list:")
                lt.info(ssdo.fit_surface_loop_record_column_headings())
                lt.info(ssdo.fit_surface_loop_record_column_headings_units())
                lt.info(ssdo.fit_surface_loop_record_column_headings_separator())
                for loop_record in loop_record_list:
                    lt.info(ssdo.fit_surface_loop_record_str(loop_record))
                lt.info(ssdo.fit_surface_loop_record_column_headings_separator())

                lt.info("\nIn fit_surface_2(), loop_record_list 2:")
                lt.info(ssdo.fit_surface_loop_record_column_headings_2())
                lt.info(ssdo.fit_surface_loop_record_column_headings_units_2())
                lt.info(ssdo.fit_surface_loop_record_column_headings_separator_2())
                for loop_record in loop_record_list:
                    lt.info(ssdo.fit_surface_loop_record_str_2(loop_record))
                lt.info(ssdo.fit_surface_loop_record_column_headings_separator_2())

            # Check loop termination.
            if loop_idx >= 20:  # 7:  # 1:  # 20:  # &&&& DELETE-SCAFFOLDING -- NEEDS BETTER LOOP EXIT CONTROL.

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

    # &&&& DELETE-SCAFFOLDING -- END NEW SOLVER ALGORITHM
