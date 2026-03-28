import numpy as np
from scipy.spatial.transform import Rotation

import opencsp.common.lib.deflectometry.slope_fitting_2d as sf2
from opencsp.common.lib.deflectometry.Surface2DAbstract import Surface2DAbstract
from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.tool.hdf5_tools import save_hdf5_datasets, load_hdf5_datasets


class Surface2DParabolic(Surface2DAbstract):
    """Representation of 2D fit parabolic surface."""

    def __init__(self, initial_focal_lengths_xy: tuple[float, float], robust_least_squares: bool, downsample: int):
        """
        Representation of 2D fit parabolic surface.

        Parameters
        ----------
        initial_focal_lengths_xy : tuple[float, float]
            Focal lengths used as starting point for optimization algorithm.
        robust_least_squares : bool
            To use robust least squares solver or not.
        downsample : int
            Amount to downsample the data by in X and Y when performing the
            alignment initialization.

        """
        super().__init__()

        # Save initial parabola shape data
        # In the following, the coefficients A, B, C, D, E, F are for the general paraboloid equation:
        #
        #     z = A + Bx + Cx^2 + Dy + Exy + Fy^2
        #
        # with derivatives:
        #
        #     dz/dx = B + 2Cx + Ey
        #
        #     dz/dy = D + Ex + 2Fy
        #
        # For further details, see B. J. Smith, R. C. Brost, and B. G. Bean, "OpenCSP
        # Deflectometry Technical Description,"" Document Version 1.0, Sandia National
        # Laboratories Technical Report SAND2024-10934, August 2024.
        #
        self.slope_fit_poly_order = 1
        self.initial_focal_lengths_xy = initial_focal_lengths_xy
        A = 0
        B = 0
        C = 1 / (4 * initial_focal_lengths_xy[0])
        D = 0
        E = 0
        F = 1 / (4 * initial_focal_lengths_xy[1])
        self.surf_coefs = np.array([A, B, C, D, E, F], dtype=float)
        # &&&& DELETE-SCAFFOLDING -- FOR OTHER SURFACE TYPES (PLANO, ETC), ENSURE THAT THEY ALSO SET THEIR SLOPE COEFFICIENTS.
        self.slope_coefs = np.array([[B, (2 * C), E], [D, E, (2 * F)]])
        # # Original version
        # self.surf_coefs = np.array(
        #     [0, 0, 1 / (4 * initial_focal_lengths_xy[0]), 0, 0, 1 / (4 * initial_focal_lengths_xy[1])], dtype=float
        # )
        # self.slope_coefs = np.zeros((2, 3))

        # Save fitting data
        self.robust_least_squares = robust_least_squares
        self.downsample = downsample

    def set_spatial_data(
        self,
        u_active_pixel_pointing_optic: Uxyz,
        v_screen_points_optic: Vxyz,
        v_optic_cam_optic: Vxyz,
        u_measure_pixel_pointing_optic: Uxyz,
        v_align_point_optic: Vxyz,
        v_optic_screen_optic: Vxyz,
    ) -> None:
        """
        Saves all spatial orientation information in object.

        Parameters
        ----------
        u_active_pixel_pointing_optic : Uxyz
            Active pixel pointing directions in optic coordinates.
        v_screen_points_optic : Vxyz
            Positions of screen intersection points in optic coordinates.
        v_optic_cam_optic : Vxyz
            Optic to camera vector in optic coordinates.
        u_measure_pixel_pointing_optic : Uxyz
            Measure pixel pointing direction in optic cooridinates.
        v_align_point_optic : Vxyz
            Position of align point in optic coordinates.
        v_optic_screen_optic : Vxyz
            Optic to screen vector in optic coordinates.

        """
        # Downsample and save measurement data
        self.u_active_pixel_pointing_optic = u_active_pixel_pointing_optic[:: self.downsample]
        self.v_screen_points_optic = v_screen_points_optic[:: self.downsample]

        # Save position data
        self.v_optic_cam_optic = v_optic_cam_optic
        self.u_measure_pixel_pointing_optic = u_measure_pixel_pointing_optic
        self.v_align_point_optic = v_align_point_optic
        self.v_optic_screen_optic = v_optic_screen_optic

        if self.robust_least_squares:
            self.num_pts = len(self.u_active_pixel_pointing_optic)
            self.weights = np.ones(self.num_pts)

    def intersect(self, u_pixel_pointing: Uxyz, v_origin: Vxyz) -> Vxyz:
        """
        Intersects incoming rays with parabolic surface.

        Parameters
        ----------
        u_pixel_pointing : Uxyz
            Unit vector, pixel pointing directions of camera pixels in optic
            coordinates.
        v_origin : Vxyz
            Location of origin point of rays in optic coordinates.

        Returns
        -------
        Vxyz
            Camera ray intersection points with optic surface in optic coordinates.

        """
        q = u_pixel_pointing.x
        r = u_pixel_pointing.y
        s = u_pixel_pointing.z

        Xc = v_origin.x[0]
        Yc = v_origin.y[0]
        Zc = v_origin.z[0]

        F = self.surf_coefs[0]  # constant
        D = self.surf_coefs[1]  # X
        A = self.surf_coefs[2]  # X^2
        E = self.surf_coefs[3]  # Y
        C = self.surf_coefs[4]  # XY
        B = self.surf_coefs[5]  # Y^2

        # Mask unit vectors pointing straight down (handled below)
        mask = np.logical_and(np.abs(q) < 1e-6, np.abs(r) < 1e-6)

        # Solve quadratic formula for ray intersections with parabola
        a = (A * q**2) + (B * r**2) + (C * q * r)
        b = (2 * A * Xc * q) + (2 * B * Yc * r) + (C * Xc * r) + (C * Yc * q) + (D * q) + (E * r) - s
        c = (A * Xc**2) + (B * Yc**2) + (C * Xc * Yc) + (D * Xc) + (E * Yc) + F - Zc

        a[mask] = np.nan
        b[mask] = np.nan

        scale_1 = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
        scale_2 = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)

        mean_1 = scale_1.mean()
        mean_2 = scale_2.mean()

        if mean_1 < 0 and mean_2 < 0:
            raise ValueError("Camera ray intersection points with parabolic surface are found to be behind camera.")
        elif mean_1 > 0 and mean_2 > 0:  # Default to use scale 1
            scale = scale_1
        elif mean_2 > 0:
            scale = scale_2
        else:
            scale = scale_1

        # Calculate unit vectors pointing straight down
        if mask.sum() > 0:
            # Calculate z intersection point
            z_pt_norm = A * Xc**2 + B * Yc**2 + C * Xc * Yc + D * Xc + E * Yc + F
            # Calculate distance from origin to intersection point
            z_scale = Zc - z_pt_norm
            # Update scales
            scale[mask] = z_scale

        # Calculate intersection points
        int_pts = v_origin + u_pixel_pointing.as_Vxyz() * scale[np.newaxis, :]  # optic coordinates
        return int_pts

    def normal_design_at_align_point(self) -> Uxyz:
        """
        Returns the surface normal of the design surface at the align point.

        Returns
        -------
        Vxyz
            Surface normal vector.

        """
        return self.normal_design_at_point(self.v_align_point_optic)

    def normal_design_at_point(self, p_xyz: Vxyz) -> Uxyz:
        """
        Returns the surface normal of the design surface at the point.

        Parameters
        -------
        p_xyz : Vxyz
            Point to compute surface normal.

        Returns
        -------
        Vxyz
            Surface normal vector.

        """
        # &&&& DELETE-SCAFFOLDING -- THIS FORMULA IS IN ERROR, BECAUSE IT DOES NOT CONSIDER THAT THE DESIGN PARAOLOID MIGHT HAVE ROTATED ASTIGMATISM
        dzdx_design = p_xyz.x[0] / (2 * self.initial_focal_lengths_xy[0])
        dzdy_design = p_xyz.y[0] / (2 * self.initial_focal_lengths_xy[1])
        normal = Uxyz([-dzdx_design, -dzdy_design, 1])  # &&&& DELETE-SCAFFOLDING -- UNNECESSARY COPY
        return normal  # &&&& DELETE-SCAFFOLDING -- UNNECESSARY COPY, FOR BREAK POINT ANALYSIS

    def normal_fit_at_align_point(self) -> Uxyz:
        """
        Returns the surface normal of the fit surface at the align point.

        Returns
        -------
        Vxyz
            Surface normal vector.

        """
        return self.normal_fit_at_point(self.v_align_point_optic)

    def normal_fit_at_point(self, p_xyz: Vxyz) -> Uxyz:
        """
        Returns the surface normal of the fit surface at the align point.

        Returns
        -------
        Vxyz
            Surface normal vector.

        """
        # &&&& DELETE-SCAFFOLDING -- ERROR IN BELOW.  NOTE COEFFICIENT [1,2] SHOULD BE [0,2].
        # ORIGINAL VERSION:
        # dzdx_meas = -(
        #     self.slope_coefs[0, 1] * self.v_align_point_optic.x[0]
        #     + self.slope_coefs[0, 0]
        #     + self.slope_coefs[1, 2] * self.v_align_point_optic.x[0]
        # )
        #
        # &&&& DELETE-SCAFFOLDING -- ANOTHER ERROR IN BELOW.  MINUS SIGN SHOULD APPEAR IN SURFACE NORMAL EQUATION, NOT SLOPE.
        # REVISED #1 VERSION:
        # dzdx_meas = -(
        #     self.slope_coefs[0, 1] * self.v_align_point_optic.x[0]
        #     + self.slope_coefs[0, 0]
        #     + self.slope_coefs[0, 2] * self.v_align_point_optic.x[0]
        # )
        # dzdy_meas = -(
        #     self.slope_coefs[1, 2] * self.v_align_point_optic.y[0]
        #     + self.slope_coefs[1, 0]
        #     + self.slope_coefs[1, 1] * self.v_align_point_optic.y[0]
        # )
        #
        # &&&& DELETE-SCAFFOLDING -- ANOTHER ERROR IN BELOW.  NOTE BOTH COEFFICIENTS MULTIPLIED BY X.
        # REVISED #2 VERSION:
        # dzdx_meas = self.slope_coefs[0, 1] * self.v_align_point_optic.x[0]
        #     + self.slope_coefs[0, 0]
        #     + self.slope_coefs[0, 2] * self.v_align_point_optic.x[0]
        #
        # &&&& DELETE-SCAFFOLDING -- ANOTHER ERROR IN BELOW.  NOTE BOTH COEFFICIENTS MULTIPLIED BY Y.
        # REVISED #2 VERSION:
        # dzdy_meas = self.slope_coefs[1, 2] * self.v_align_point_optic.y[0]
        #     + self.slope_coefs[1, 0]
        #     + self.slope_coefs[1, 1] * self.v_align_point_optic.y[0]

        # From routine fit_slopes() below:
        #
        #     slope_coefs_x[0] = B
        #     slope_coefs_x[1] = 2C
        #     slope_coefs_x[2] = E
        #
        #     slope_coefs_y[0] = D
        #     slope_coefs_y[1] = E
        #     slope_coefs_y[2] = 2F
        #
        B = self.slope_coefs[0, 0]
        C = self.slope_coefs[0, 1] / 2
        E = self.slope_coefs[0, 2]

        D = self.slope_coefs[1, 0]
        E = self.slope_coefs[1, 1]
        F = self.slope_coefs[1, 2] / 2

        dzdx_meas = B + (2 * C * p_xyz.x[0]) + (E * p_xyz.y[0])
        dzdy_meas = D + (E * p_xyz.x[0]) + (2 * F * p_xyz.y[0])
        normal = Uxyz((-dzdx_meas, -dzdy_meas, 1))  # &&&& DELETE-SCAFFOLDING -- UNNECESSARY COPY
        return normal  # &&&& DELETE-SCAFFOLDING -- UNNECESSARY COPY, FOR BREAK POINT ANALYSIS

    def average_fit_normal(self) -> Uxyz:
        # &&&& DELETE-SCAFFOLDING -- VECTORIZE THIS
        # &&&& DELETE-SCAFFOLDING -- MOVE INTO BASE CLASS
        # &&&& DELETE-SCAFFOLDING -- PROVIDE DOCSTRING
        # &&&& DELETE-SCAFFOLDING -- NOTE INTERSECTION POINTS MUST BE SET.  PROTECT WITH ERROR CHECK.
        n_pts = self.v_surf_int_pts_optic.len()
        x_sum = 0
        y_sum = 0
        z_sum = 0
        for idx in range(n_pts):
            x = self.v_surf_int_pts_optic.x[idx]
            y = self.v_surf_int_pts_optic.y[idx]
            z = self.v_surf_int_pts_optic.z[idx]
            uxyz_normal = self.normal_fit_at_point(Vxyz([x, y, z]))
            x_sum += uxyz_normal.x[0]
            y_sum += uxyz_normal.y[0]
            z_sum += uxyz_normal.z[0]
        x_avg = x_sum / n_pts
        y_avg = y_sum / n_pts
        z_avg = z_sum / n_pts
        # Note that the average of the coordinates of many units vectors is generally
        # not a unit vector.  But the Uxyz constructor will normalize it.
        uxyz_avg_normal = Uxyz([x_avg, y_avg, z_avg])  # &&&& DELETE-SCAFFOLDING -- UNNECESSARY COPY
        return uxyz_avg_normal  # &&&& DELETE-SCAFFOLDING -- UNNECESSARY COPY, FOR BREAK POINT ANALYSIS

    def average_measured_normal(self) -> Uxyz:
        # &&&& DELETE-SCAFFOLDING -- VECTORIZE THIS
        # &&&& DELETE-SCAFFOLDING -- MOVE INTO BASE CLASS
        # &&&& DELETE-SCAFFOLDING -- PROVIDE DOCSTRING
        # &&&& DELETE-SCAFFOLDING -- NOTE SLOPES MUST BE SET.  PROTECT WITH ERROR CHECK.
        n_pts = self.slopes.shape[1]
        x_sum = 0
        y_sum = 0
        z_sum = 0
        for idx in range(n_pts):
            dzdx = self.slopes[0, idx]
            dzdy = self.slopes[1, idx]
            uxyz_normal = Uxyz([-dzdx, -dzdy, 1])  # Uxyz normalizes its input
            x_sum += uxyz_normal.x[0]
            y_sum += uxyz_normal.y[0]
            z_sum += uxyz_normal.z[0]
        x_avg = x_sum / n_pts
        y_avg = y_sum / n_pts
        z_avg = z_sum / n_pts
        # Note that the average of the coordinates of many units vectors is generally
        # not a unit vector.  But the Uxyz constructor will normalize it.
        uxyz_avg_normal = Uxyz([x_avg, y_avg, z_avg])  # &&&& DELETE-SCAFFOLDING -- UNNECESSARY COPY
        return uxyz_avg_normal  # &&&& DELETE-SCAFFOLDING -- UNNECESSARY COPY, FOR BREAK POINT ANALYSIS

    def calculate_surface_intersect_points(self) -> None:
        """
        Calculates pixel ray intersection points with surface.

        """
        # Calculate pixel intersection points with existing fitting function
        self.v_surf_int_pts_optic = self.intersect(self.u_active_pixel_pointing_optic, self.v_optic_cam_optic)

    def calculate_slopes(self) -> tuple[Vxyz, np.ndarray]:
        """
        Calculate slopes of each measurement point.

        """
        self.slopes = sf2.calc_slopes(self.v_surf_int_pts_optic, self.v_optic_cam_optic, self.v_screen_points_optic)

    def fit_slopes(self) -> None:
        """
        Fits slopes to surface.

        """
        # Fit Nth order surfaces to slope distributions in X and Y
        if self.robust_least_squares:
            slope_coefs_x, weights_x = sf2.fit_slope_robust_ls(
                self.slope_fit_poly_order, self.slopes[0], self.weights.copy(), self.v_surf_int_pts_optic
            )
            slope_coefs_y, weights_y = sf2.fit_slope_robust_ls(
                self.slope_fit_poly_order, self.slopes[1], self.weights.copy(), self.v_surf_int_pts_optic
            )
            self.weights = np.array((weights_x, weights_y)).min(0)
        else:
            slope_coefs_x = sf2.fit_slope_ls(self.slope_fit_poly_order, self.slopes[0], self.v_surf_int_pts_optic)
            slope_coefs_y = sf2.fit_slope_ls(self.slope_fit_poly_order, self.slopes[1], self.v_surf_int_pts_optic)

        # Average to create surface shape coefficients.
        #
        # In the following, the coefficients A, B, C, D, E, F are for the general paraboloid equation:
        #
        #     z = A + Bx + Cx^2 + Dy + Exy + Fy^2
        #         c0  c1x  c2x2   c3y  c4xy  c5y2  <-- Surface equation coefficients
        #
        # with derivatives:
        #
        #     dz/dx = B + 2Cx + Ey
        #
        #     dz/dy = D + Ex + 2Fy
        #
        # For further details, see B. J. Smith, R. C. Brost, and B. G. Bean, "OpenCSP
        # Deflectometry Technical Description,"" Document Version 1.0, Sandia National
        # Laboratories Technical Report SAND2024-10934, August 2024.
        #
        # The fit routines return these coefficients from the linear least-squares fit in the
        # order [k0,k1,k2], corresponding to dz = k0 + k1*x + k2*y.  These are packed into
        # the self.surf_coeffs data member, with the following correspondence:
        #
        #     slope_coefs_x[0] = B
        #     slope_coefs_x[1] = 2C
        #     slope_coefs_x[2] = E
        #
        #     slope_coefs_y[0] = D
        #     slope_coefs_y[1] = E
        #     slope_coefs_y[2] = 2F
        #
        # These mappings allow us to save the coefficients below.

        # Save slope coefficients.
        self.slope_coefs = np.array((slope_coefs_x, slope_coefs_y))

        # Save surface coefficients.
        self.surf_coefs = np.array(
            [
                0,  # A
                slope_coefs_x[0],  # B
                slope_coefs_x[1] / 2,  # C
                slope_coefs_y[0],  # D
                (slope_coefs_x[2] + slope_coefs_y[1]) / 2,  # E
                slope_coefs_y[2] / 2,  # F
            ]
        )

        # Calculate z coordinate
        z_pt = self.v_align_point_optic.z[0] - sf2.coef_to_points(self.v_align_point_optic, self.surf_coefs, 2)
        self.surf_coefs[0] = z_pt

    def rotate_all(self, r_align_step: Rotation) -> None:
        """
        Rotates all spatial vectors about align point by given rotation.

        Parameters
        ----------
        r_align_step : Rotation
            Rotation object to rotate all vectors by.

        """
        self.v_optic_cam_optic = self.v_optic_cam_optic.rotate_about(r_align_step, self.v_align_point_optic)
        self.v_screen_points_optic = self.v_screen_points_optic.rotate_about(r_align_step, self.v_align_point_optic)
        self.u_active_pixel_pointing_optic = self.u_active_pixel_pointing_optic.rotate(r_align_step)
        self.u_measure_pixel_pointing_optic = self.u_measure_pixel_pointing_optic.rotate(r_align_step)
        self.v_optic_screen_optic = self.v_optic_screen_optic.rotate_about(r_align_step, self.v_align_point_optic)

    def shift_all(self, v_align_optic_step: Vxyz) -> None:
        """
        Shifts all spatial vectors by given step size.

        Parameters
        ----------
        v_align_optic_step : Vxyz
            Shift vector to apply to all data.

        """
        self.v_optic_cam_optic += v_align_optic_step
        self.v_screen_points_optic += v_align_optic_step
        self.v_optic_screen_optic += v_align_optic_step

    def save_to_hdf(self, file: str, prefix: str = ""):
        """Saves data to given file. Data is stored as: PREFIX + ParamsSurface/Field_1

        Parameters
        ----------
        file : str
            HDF file to save to
        prefix : str, optional
            Prefix to append to folder path within HDF file (folders must be separated by "/").
            Default is empty string ''.
        """
        data = [self.initial_focal_lengths_xy, self.robust_least_squares, self.downsample, "parabolic"]
        datasets = [
            prefix + "ParamsSurface/initial_focal_lengths_xy",
            prefix + "ParamsSurface/robust_least_squares",
            prefix + "ParamsSurface/downsample",
            prefix + "ParamsSurface/surface_type",
        ]
        save_hdf5_datasets(data, datasets, file)

    @classmethod
    def load_from_hdf(cls, file: str, prefix: str = ""):
        """Loads data from given file. Assumes data is stored as: PREFIX + ParamsSurface/Field_1

        Parameters
        ----------
        file : str
            HDF file to load from
        prefix : str, optional
            Prefix to append to folder path within HDF file (folders must be separated by "/").
            Default is empty string ''.
        """
        # Check surface type
        data = load_hdf5_datasets([prefix + "ParamsSurface/surface_type"], file)
        if data["surface_type"] != "parabolic":
            raise ValueError(f'Surface2DParabolic cannot load surface type, {data["surface_type"]:s}')

        # Load
        datasets = [
            prefix + "ParamsSurface/initial_focal_lengths_xy",
            prefix + "ParamsSurface/robust_least_squares",
            prefix + "ParamsSurface/downsample",
        ]
        data = load_hdf5_datasets(datasets, file)
        return cls(**data)
