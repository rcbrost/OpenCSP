from abc import abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.tool.hdf5_tools import HDF5_IO_Abstract


class Surface2DAbstract(HDF5_IO_Abstract):
    """Representation of 2d surface for SOFAST processing"""

    def __init__(self):
        # Instantiate variables
        self.v_surf_int_pts_optic: Vxyz
        self.slopes: np.ndarray
        self.u_active_pixel_pointing_optic: Vxyz
        self.v_screen_points_optic: Vxyz
        self.v_optic_cam_optic: Vxyz
        self.u_measure_pixel_pointing_optic: Uxyz
        self.v_optic_screen_optic: Vxyz
        self.v_align_point_optic: Vxyz
        self.weights: np.ndarray
        self.num_pts: int
        self.surf_coefs: np.ndarray
        self.slope_coefs: np.ndarray

    @abstractmethod
    def set_spatial_data(
        self,
        u_active_pixel_pointing_optic: Uxyz,
        v_screen_points_optic: Vxyz,
        v_optic_cam_optic: Vxyz,
        u_measure_pixel_pointing_optic: Uxyz,
        v_align_point_optic: Vxyz,
        v_optic_screen_optic: Vxyz,
    ) -> None:
        """Saves spatial data in class"""

    @abstractmethod
    def intersect(self, u_pixel_pointing: Uxyz, v_origin: Vxyz) -> Vxyz:
        """Intersects rays with the surface"""

    @abstractmethod
    def normal_design_at_align_point(self) -> Vxyz:
        """Normal vector of design surface at alignment point"""

    @abstractmethod
    def normal_design_at_point(self, p_xyz: Vxyz) -> Vxyz:
        """Normal vector of design surface at given point"""

    @abstractmethod
    def normal_fit_at_align_point(self) -> Vxyz:
        """Normal vector of fit surface at alignment point"""

    @abstractmethod
    def normal_fit_at_point(self, p_xyz: Vxyz) -> Vxyz:
        """Normal vector of fit surface at given point"""

    @abstractmethod
    def calculate_surface_intersect_points(self) -> Vxyz:
        """Calculates surface intersection points"""

    @abstractmethod
    def calculate_slopes(self) -> tuple[Vxyz, np.ndarray]:
        """Calculate slopes for all measurement points"""

    @abstractmethod
    def fit_slopes(self) -> None:
        """Fits slopes to using coefficients"""

    @abstractmethod
    def rotate_all(self, r_align_step: Rotation) -> None:
        """Rotates all data vectors"""

    @abstractmethod
    def shift_all(self, v_align_optic_step: Vxyz) -> None:
        """Shifts all data vectors"""

    def plot_intersection_points(self, axes: plt.Axes, downsample: int = 50) -> None:
        """Plots calculated intersection points with surface and align point.

        Parameters
        ----------
        axes : plt.Axes
            Matplotlib axes.
        downsample : int, optional
            Ray downsample factor, by default 50

        Returns
        -------
        Matplotlib axes
        """
        # Plot intersection points surface
        axes.plot_trisurf(
            *self.v_surf_int_pts_optic[::downsample].data, edgecolor="none", alpha=0.5, linewidth=0, antialiased=False
        )

    def plot_screen_points(self, axes: plt.Axes, downsample: int = 50) -> None:
        """Plots calculated screen reflection points.

        Parameters
        ----------
        axes : plt.Axes
            Matplotlib axes.
        downsample : int, optional
            Ray downsample factor, by default 50

        Returns
        -------
        Matplotlib axes
        """
        # Plot intersection points surface
        axes.plot_trisurf(
            *self.v_screen_points_optic[::downsample].data, edgecolor="none", alpha=0.5, linewidth=0, antialiased=False
        )
