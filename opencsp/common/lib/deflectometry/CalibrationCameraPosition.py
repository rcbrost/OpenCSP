import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from scipy.spatial.transform import Rotation

from opencsp.common.lib.photogrammetry.photogrammetry import find_aruco_marker
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz
import opencsp.common.lib.tool.log_tools as lt


class CalibrationCameraPosition:
    """Class that calculates the relative pose of a camera viewing several Aruco markers.
    Must have a list of xyz marker corner points and corner IDs. There can be more IDs in
    this list than is viewed by the camera.

    Calculates
    ----------
    - rvec : rotation vector, screen to camera rotation vector
    - tvec : translation vector, camera to screen (in screen coordinates) translation vector

    Attributes
    ----------
    make_figures : bool
        To create output summary figures
    """

    def __init__(
        self,
        camera: Camera,
        reconstruction_pts_xyz_corners: Vxyz,
        reconstruction_ids_corners: ndarray,
        cal_image: ndarray,
    ) -> "CalibrationCameraPosition":
        """Instantiates class

        Parameters
        ----------
        camera : Camera
            Camera object
        pts_xyz_corners : Vxyz
            Aruco corner xyz positions
        ids_corners : ndarray
            Corner IDs corresponding to pts_xyz_corners
        cal_image : ndarray
            Calibration image captured by camera
        """
        # Initialize attributes data types
        self.camera = camera
        self.reconstruction_pts_xyz_corners = reconstruction_pts_xyz_corners
        self.reconstruction_ids_corners = reconstruction_ids_corners
        self.image = cal_image

        self.ids_corners: ndarray
        self.image: ndarray
        self.found_ids_marker: ndarray[int]
        self.found_pts_xy_marker_corners_list: list[ndarray]
        self.found_reconstruction_ids_marker: ndarray[int]
        self.found_reconstruction_pts_xy_marker_corners_list: list[ndarray]
        self.pts_xyz_active_corner_locations: Vxyz
        self.rot_screen_cam: Rotation
        self.v_cam_screen_cam: Vxyz
        self.v_cam_screen_screen: Vxyz
        self.errors_reprojection_xy: Vxy
        self.pts_xy_marker_corners_reprojected: Vxy

        # Save figures
        self.make_figures = False
        self.figures: list[plt.Figure] = []

    def find_markers(self) -> None:
        """Finds marker corner locations in image"""
        self.found_ids_marker, self.found_pts_xy_marker_corners_list = find_aruco_marker(self.image)
        lt.info("Markers found: " + str(self.found_ids_marker))

    def discard_new_markers(self) -> None:
        """
        The scene reconstruction analysis of a set of photogrammetry camera images found some collection of Aruco markers, each with an associated unique ID.  Thus there is a set of expected marker IDs.  We will call this set ER, for "Expected from Reconstruction."

        Meanwhile, the spatial orientation image captured by the SOFAST camera will generally not see all of these markers, because some may be outside the field of view, partially occluded, etc.

        Thus we need to search the SOFAST camera image for markers.  We will call the resulting set of markers FS, for "Found in SOFAST image."

        Typically we expect FS to be a subset of ER.

        However, new Aruco markers might appear in the SOFAST camera image that were not seen in reconstruction.  These may occurs because an Aruco marker was not found during reconstruction (due to occlusion, etc, or becuase of image processing "surprises."  In ny case, we need to disregard tany new Aruco markers found in the SOFAST camera image.  That is, we ignore all markers in (FS - ER).

        """
        # self.found_reconstruction_ids_marker = self.found_ids_marker
        # self.found_reconstruction_pts_xy_marker_corners_list = self.found_pts_xy_marker_corners_list

        reconstruction_ids_corners_list = self.reconstruction_ids_corners.tolist()
        self.found_reconstruction_ids_marker = []
        self.found_reconstruction_pts_xy_marker_corners_list = []
        for found_marker_id, found_pts_xy_marker_corners in zip(
            self.found_ids_marker, self.found_pts_xy_marker_corners_list
        ):
            # See if the found marker was identified by the scene reconstruction.
            marker_corner_zero_id = found_marker_id * 4
            if marker_corner_zero_id in reconstruction_ids_corners_list:
                self.found_reconstruction_ids_marker.append(found_marker_id)
                self.found_reconstruction_pts_xy_marker_corners_list.append(found_pts_xy_marker_corners)

    def collect_corner_xyz_locations(self) -> None:
        """Collects corner locations of viewed markers"""
        # Extract object points
        self.pts_xyz_active_corner_locations = Vxyz.empty()
        reconstruction_ids_corners_list = self.reconstruction_ids_corners.tolist()
        # Markers that are both found in the SOFAST image and also identified by
        # the scene reconstruction are active.
        for active_marker_id in self.found_reconstruction_ids_marker:
            # Get index of current marker
            index = reconstruction_ids_corners_list.index(active_marker_id * 4)
            # Extract calibrated corner locations (4 corners per marker)
            self.pts_xyz_active_corner_locations = self.pts_xyz_active_corner_locations.concatenate(
                self.reconstruction_pts_xyz_corners[index : index + 4]
            )

    def calculate_camera_pose(self) -> None:
        """Calculates the camera pose"""
        # Concatenate image points
        pts_img = np.vstack(self.found_reconstruction_pts_xy_marker_corners_list)

        # Calculate rvec/tvec
        ret, rvec, tvec = cv.solvePnP(
            self.pts_xyz_active_corner_locations.data.T, pts_img, self.camera.intrinsic_mat, self.camera.distortion_coef
        )
        if not ret:
            raise ValueError("Camera calibration was not successful.")

        rvec: ndarray = rvec.squeeze()
        tvec: ndarray = tvec.squeeze()

        self.rot_screen_cam = Rotation.from_rotvec(rvec)
        self.v_cam_screen_cam = Vxyz(tvec)
        self.v_cam_screen_screen = self.v_cam_screen_cam.rotate(self.rot_screen_cam.inv())

        lt.info("Camera pose calculated:")
        lt.info(f"rvec: {self.rot_screen_cam.as_rotvec()}")
        lt.info(f"tvec: {self.v_cam_screen_screen.data.squeeze()}")

    def get_data(self) -> tuple[ndarray, ndarray]:
        """Returns rvec and tvec orienting camera to screen coordinates

        Returns
        -------
        rot_screen_cam : ndarray
            (3,) vector, screen to camera rotation
        v_cam_screen_screen : ndarray
            (3,) vector, camera to screen translation in screen coordinates
        """
        rvec = self.rot_screen_cam.as_rotvec()
        tvec = self.v_cam_screen_screen.data.squeeze()
        return rvec, tvec

    def save_data_as_csv(self, file: str) -> None:
        """Saves rvec/tvec as csv file"""
        rvec, tvec = self.get_data()
        data = np.vstack((rvec, tvec))
        np.savetxt(file, data, delimiter=",", fmt="%.8f")

        lt.info(f"Saved camera rvec and tvec to: {os.path.abspath(file):s}")

    def calculate_reprojection_error(self) -> None:
        """Calculates reprojection error"""
        # Project points
        self.pts_xy_marker_corners_reprojected = self.camera.project(
            self.pts_xyz_active_corner_locations, self.rot_screen_cam, self.v_cam_screen_cam
        )

        # Calculate errors
        self.errors_reprojection_xy = self.pts_xy_marker_corners_reprojected - Vxy(
            np.vstack(self.found_reconstruction_pts_xy_marker_corners_list).T
        )

        errors_mag: ndarray = np.sqrt((self.errors_reprojection_xy.data.T**2).sum(axis=1))
        lt.info("Camera pose reprojection errors:")
        lt.info(f"Mean error: {errors_mag.mean():.3f} pixels")
        lt.info(f"STDEV of errors (N={errors_mag.size}): {errors_mag.std():.4f} pixels")

    def plot_found_corners(self) -> None:
        """Plots camera image and found corners"""
        fig = plt.figure("CalibrationCameraPosition_Found_Markers")
        self.figures.append(fig)
        ax = fig.gca()

        ax.imshow(self.image, cmap="gray")
        for id_, pts in zip(self.found_ids_marker, self.found_pts_xy_marker_corners_list):
            plt.scatter(*pts.T, marker="+")
            plt.text(*pts.mean(0).T + np.array([60, 0]), id_, backgroundcolor="white")

    def plot_found_reconstruction_corners(self) -> None:
        """Plots camera image and found corners"""
        fig = plt.figure("CalibrationCameraPosition_Found_Reconstruction_Markers")
        self.figures.append(fig)
        ax = fig.gca()

        ax.imshow(self.image, cmap="gray")
        for id_, pts in zip(self.found_reconstruction_ids_marker, self.found_reconstruction_pts_xy_marker_corners_list):
            plt.scatter(*pts.T, marker="+")
            plt.text(*pts.mean(0).T + np.array([60, 0]), id_, backgroundcolor="white")

    def plot_reprojection_error(self) -> None:
        """Plots reprojection error"""
        fig = plt.figure("CalibrationCameraPosition_Reprojection_Error")
        self.figures.append(fig)
        ax = fig.gca()

        pts_img = np.vstack(self.found_reconstruction_pts_xy_marker_corners_list)

        ax.imshow(self.image, cmap="gray")
        ax.scatter(*pts_img.T, edgecolor="green", facecolor="none", label="Image Points")
        ax.scatter(*self.pts_xy_marker_corners_reprojected.data, marker=".", color="blue", label="Reprojected")
        dx = self.errors_reprojection_xy.x
        dy = -self.errors_reprojection_xy.y
        ax.quiver(*self.pts_xy_marker_corners_reprojected.data, dx, dy, label="Error", color="red")
        ax.legend()
        ax.axis("off")

    def run_calibration(self) -> None:
        """Runs calibration sequence"""

        # Run calibration
        self.find_markers()
        self.discard_new_markers()
        self.collect_corner_xyz_locations()
        self.calculate_camera_pose()
        self.calculate_reprojection_error()

        # Plot figures
        if self.make_figures:
            self.plot_found_corners()
            self.plot_found_reconstruction_corners()
            self.plot_reprojection_error()
