"""Library of functions used to process the geometry of a deflectometry setup."""

import copy
import os.path

import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from numpy import ndarray
from scipy.spatial.transform import Rotation

from opencsp.common.lib.camera.Camera import Camera
import opencsp.app.sofast.lib.calculation_data_classes as cdc
from opencsp.app.sofast.lib.DebugOpticsGeometry import DebugOpticsGeometry
from opencsp.app.sofast.lib.DefinitionEnsemble import DefinitionEnsemble
from opencsp.app.sofast.lib.DefinitionFacet import DefinitionFacet
from opencsp.app.sofast.lib.ParamsOpticGeometry import ParamsOpticGeometry
from opencsp.app.sofast.lib.ParamsMaskCalculation import ParamsMaskCalculation
import opencsp.app.sofast.lib.process_optics_geometry_debug_output as pogdo
import opencsp.app.sofast.lib.sofast_debug_figure_support as sdfs
import opencsp.app.sofast.lib.image_processing as ip
from opencsp.app.sofast.lib.SpatialOrientation import SpatialOrientation
import opencsp.app.sofast.lib.spatial_processing as sp
from opencsp.common.lib.geometry.LoopXY import LoopXY
import opencsp.common.lib.geometry.TransformXYZ as txyz
from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render.view_spec as vs
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlFigure as rcfg
import opencsp.common.lib.render_control.RenderControlMirror as rcm
import opencsp.common.lib.render_control.RenderControlMirrorProjected as rcmp
import opencsp.common.lib.render_control.RenderControlMirrorEmbedded as rcme
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.render_control.RenderControlSofastSetup as rcss
import opencsp.common.lib.tool.log_tools as lt


def process_singlefacet_geometry(
    facet_data: DefinitionFacet,
    mask_raw: ndarray,
    v_measure_point_facet: Vxyz,
    dist_optic_screen: float,
    orientation: SpatialOrientation,
    camera: Camera,
    params: ParamsOpticGeometry = ParamsOpticGeometry(),
    debug: DebugOpticsGeometry = DebugOpticsGeometry(),
) -> tuple[
    cdc.CalculationDataGeometryGeneral,
    cdc.CalculationImageProcessingGeneral,
    list[cdc.CalculationDataGeometryFacet],
    list[cdc.CalculationImageProcessingFacet],
    cdc.CalculationError,
]:
    """Processes optic geometry for single facet

    Parameters
    ----------
    facet_data : DefinitionFacet
        DefinitionFacet object
    mask_raw : ndarray
        Raw calculated mask
    v_measure_point_facet : Vxyz
        Measure point location on facet, meters
    dist_optic_screen : float
        Optic to screen distance, meters
    orientation : SpatialOrientation
        SpatialOrientation object
    camera : Camera
        Camera object
    params : ParamsOpticGeometry, optional
        ParamsOpticGeometry object, by default ParamsOpticGeometry()
    debug : DebugOpticsGeometry, optional
        DebugOpticsGeometry object, by default DebugOpticsGeometry()

    Returns
    -------
    data_geometry_general: calculation_data_classes.CalculationDataGeometryGeneral
        Positional optic geometry calculations general to entire measurement; not facet specific.
    data_image_processing_general: calculation_data_classes.CalculationImageProcessingGeneral
        Image processing calculations general to entire measurement; not facet specific.
    data_geometry_facet: list[calculation_data_classes.CalculationDataGeometryFacet]
        List of positional optic geometry calculations specific to each facet. Order is
        same as input facet definitions.
    data_image_processing_facet: list[calculation_data_classes.CalculationImageProcessingFacet]
        List of image processing calculations specific to each facet. Order is same as input facet
        definitions.
    data_error: calculation_data_classes.CalculationError
        Geometric/positional errors and reprojection errors associated with solving for facet location.
    """
    # &&&& DELETE-SCAFFOLDING -- PASS THESE IN
    mask_insert_background_clutter = True
    mask_clutter_box = (250, 1750, 50, 390)  # (x_min, x_max, y_min, y_max)
    mask_clutter_circle = (250, 500, 400)  # (x_center, y_center, radius)
    mask_remove_background = True
    mask_ROI_loop = LoopXY.from_vertices(
        Vxy.from_list([(500, 1100), (1400, 1100), (1400, 450), (900, 450), (500, 900)])
    )
    mask_dilate_erode = True
    dilate_erode_kernel_size = 8
    mask_keep_largest_area = True
    mask_add_border = True

    if debug.debug_active:
        lt.debug("process_optics_geometry debug on.")
    else:
        lt.debug("process_optics_geometry debug off.")

    # Create data classes
    data_geometry_general = cdc.CalculationDataGeometryGeneral()
    data_image_processing_general = cdc.CalculationImageProcessingGeneral()
    data_geometry_facet = cdc.CalculationDataGeometryFacet()
    data_image_processing_facet = cdc.CalculationImageProcessingFacet()
    data_error = cdc.CalculationError()

    # Make copy of orientation
    ori = copy.copy(orientation)

    # Get optic data
    v_facet_corners: Vxyz = facet_data.v_facet_corners  # Corners of facet in facet coordinates
    v_facet_centroid: Vxyz = facet_data.v_facet_centroid  # Centroid of facet in facet coordinates
    # Surface normal at centroid in facet coordinates
    u_facet_centroid_normal: Uxyz = facet_data.u_facet_centroid_normal

    # Draw SOFAST setup, wthout a mirror.
    if True:  # debug.debug_active:  # &&&& DELETE-SCAFFOLDING -- TEMPORARY
        figure_title = "SOFAST Setup, Without Mirror"
        pogdo.figure_sofast_setup_without_mirror(figure_title, camera, facet_data, orientation, debug)

    # Save mask raw
    data_image_processing_general.mask_raw = mask_raw

    # Plot mask
    if debug.debug_active:
        figure_title = "Raw Mask"
        fig_rec = sdfs.start_debug_image_figure(figure_title)
        fig_rec.view.imshow(mask_raw, cmap="gray")
        sdfs.finish_debug_image_figure(figure_title, 'geometry', fig_rec, debug)

    mask_copy_for_cleanup = np.copy(mask_raw)

    # Introduce background clutter, for test/demonstration purposes.
    if mask_insert_background_clutter:
        # OpenCV doesn't support rectangle or circle for Boolean arrays.
        # We multiply by 128 to make visualizations recognizable.
        # We multiply by 128 instead of 255 to make visualizations different from Boolean masks.
        mask_copy_for_cleanup_uint8 = mask_copy_for_cleanup.astype(np.uint8) * 128
        # Draw a white rectangle
        x_min = mask_clutter_box[0]
        x_max = mask_clutter_box[1]
        y_min = mask_clutter_box[2]
        y_max = mask_clutter_box[3]
        cv.rectangle(mask_copy_for_cleanup_uint8, (x_min, y_min), (x_max, y_max), 255, -1)
        # Draw a white circle
        x_center = mask_clutter_circle[0]
        y_center = mask_clutter_circle[1]
        radius = mask_clutter_circle[2]
        cv.circle(mask_copy_for_cleanup_uint8, (x_center, y_center), radius, 255, -1)
        # Convert back to Boolean.
        mask_copy_for_cleanup = mask_copy_for_cleanup_uint8.astype('bool')
        if debug.debug_active:
            figure_title = "After Inserting Background Clutter"
            fig_rec = sdfs.start_debug_image_figure(figure_title)
            fig_rec.view.imshow(mask_copy_for_cleanup, cmap="gray")
            sdfs.finish_debug_image_figure(figure_title, 'geometry', fig_rec, debug)

    # Remove mask background clutter.
    # These can lead to incorrect region focus, gross error in facet centroid identification, etc.
    if mask_remove_background:
        # Plot ROI specifying removal.
        if debug.debug_active:
            figure_title = "User-Specified Region of Interest (ROI)"
            fig_rec = sdfs.start_debug_image_figure(figure_title)
            fig_rec.view.imshow(mask_copy_for_cleanup, cmap="gray")
            fig_rec.view.draw_pq_list(mask_ROI_loop.as_xy_list(), close=True, style=rcps.outline(color='red'))
            sdfs.finish_debug_image_figure(figure_title, 'geometry', fig_rec, debug)
        # Execute removal.
        vx = np.arange(mask_copy_for_cleanup.shape[1])
        vy = np.arange(mask_copy_for_cleanup.shape[0])
        mask_ROI = mask_ROI_loop.as_mask(vx, vy)
        mask_copy_after_ROI = np.logical_and(mask_copy_for_cleanup, mask_ROI)
        # Plot removal result.
        if debug.debug_active:
            figure_title = "After Background Removal"
            fig_rec = sdfs.start_debug_image_figure(figure_title)
            fig_rec.view.imshow(mask_copy_after_ROI, cmap="gray")
            sdfs.finish_debug_image_figure(figure_title, 'geometry', fig_rec, debug)
    else:
        mask_copy_after_ROI = mask_copy_for_cleanup.copy()

    # Remove mask internal features.
    # These can corrupt centroid finding and loop refinement.
    # We do this by first dilating the white in the image, and then eroding it.
    #
    # This could be accomplished with a single OpenCV "close" operation:
    #    closing = cv.morphologyEx(mask_clean, cv.MORPH_CLOSE, kernel)
    # But we split into two operations so that we can draw figures illustrating
    # the construction.
    #
    # For a detailed explanation of these operations, see:
    #    https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
    #
    if mask_dilate_erode:
        # OpenCV doesn't support dilate or erode for Boolean arrays.
        # We multiply by 128 to make visualizations recognizable.
        # We multiply by 128 instead of 255 to make visualizations different from Boolean masks.
        mask_copy_after_ROI_uint8 = mask_copy_after_ROI.astype(np.uint8) * 128
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (dilate_erode_kernel_size, dilate_erode_kernel_size))
        mask_copy_after_ROI_dilate_uint8 = cv.dilate(mask_copy_after_ROI_uint8, kernel)
        mask_copy_after_ROI_dilate_erode_uint8 = cv.erode(mask_copy_after_ROI_dilate_uint8, kernel, iterations=1)
        # Convert back to Boolean.
        mask_copy_after_ROI_dilate = mask_copy_after_ROI_dilate_uint8.astype('bool')
        mask_copy_after_ROI_dilate_erode = mask_copy_after_ROI_dilate_erode_uint8.astype('bool')
        if debug.debug_active:
            # Dilated
            figure_title = "Dilated Mask"
            fig_rec = sdfs.start_debug_image_figure(figure_title)
            fig_rec.view.imshow(mask_copy_after_ROI_dilate, cmap="gray")
            sdfs.finish_debug_image_figure(figure_title, 'geometry', fig_rec, debug)
            # Dilated then eroded
            figure_title = "Dilated Then Eroded Mask"
            fig_rec = sdfs.start_debug_image_figure(figure_title)
            fig_rec.view.imshow(mask_copy_after_ROI_dilate_erode, cmap="gray")
            sdfs.finish_debug_image_figure(figure_title, 'geometry', fig_rec, debug)
        else:
            mask_copy_after_ROI_dilate_erode = mask_copy_after_ROI.copy()

    # Select largest mask area.
    # If enabled, keep only the largest mask area
    if mask_keep_largest_area:
        mask_copy_after_ROI_dilate_erode_largest = ip.keep_largest_mask_area(mask_copy_after_ROI_dilate_erode)
        if debug.debug_active:
            figure_title = "After Keeping Largest Area"
            fig_rec = sdfs.start_debug_image_figure(figure_title)
            fig_rec.view.imshow(mask_copy_after_ROI_dilate_erode_largest, cmap="gray")
            sdfs.finish_debug_image_figure(figure_title, 'geometry', fig_rec, debug)
    else:
        mask_copy_after_ROI_dilate_erode_largest = mask_copy_after_ROI_dilate_erode.copy()

    # Show edges of mask, before adding border.
    if debug.debug_active:
        # Find edges of mask
        v_edges_image_before_border = ip.edges_from_mask(mask_copy_after_ROI_dilate_erode_largest)
        figure_title = "Edges of Current Mask"
        fig_rec = sdfs.start_debug_image_figure(figure_title)
        fig_rec.view.imshow(mask_copy_after_ROI_dilate_erode_largest, cmap="gray")
        fig_rec.view.axis.scatter(*v_edges_image_before_border.data, marker=".", c='red', s=0.8)
        sdfs.finish_debug_image_figure(figure_title, 'geometry', fig_rec, debug)

    # Add black border, so that edges will be generated for any places where
    # mirror touches the boundary of the field of view.
    #
    # This is needed because otherwise the loop refinement will be degraded or fail.
    # Note that ideally it is better to get a differnt data set that avoids this,
    # but that may be inconvenient or even impractical due to measurement constraints.
    #
    mask_copy_after_ROI_dilate_erode_largest_border = mask_copy_after_ROI_dilate_erode_largest.copy()
    if mask_add_border:
        n_rows = mask_copy_after_ROI_dilate_erode_largest_border.shape[0]
        n_cols = mask_copy_after_ROI_dilate_erode_largest_border.shape[1]
        mask_copy_after_ROI_dilate_erode_largest_border[0, :] = 0
        mask_copy_after_ROI_dilate_erode_largest_border[(n_rows - 1), :] = 0
        mask_copy_after_ROI_dilate_erode_largest_border[:, 0] = 0
        mask_copy_after_ROI_dilate_erode_largest_border[:, (n_cols - 1)] = 0
        if debug.debug_active:
            figure_title = "After Adding Black Border"
            fig_rec = sdfs.start_debug_image_figure(figure_title)
            fig_rec.view.imshow(mask_copy_after_ROI_dilate_erode_largest_border, cmap="gray")
            sdfs.finish_debug_image_figure(figure_title, 'geometry', fig_rec, debug)

    # Construct final cleaned-up mask.
    mask_clean = mask_copy_after_ROI_dilate_erode_largest_border.copy()
    data_image_processing_facet.mask_clean = mask_clean

    # Find edges of mask
    v_edges_image = ip.edges_from_mask(mask_clean)
    data_image_processing_general.v_edges_image = v_edges_image

    # Plot mask edges
    if debug.debug_active:
        figure_title = "Mask Edges After Adding Border"
        fig_rec = sdfs.start_debug_image_figure(figure_title)
        fig_rec.view.imshow(mask_clean, cmap="gray")
        fig_rec.view.axis.scatter(*v_edges_image.data, marker=".", c='red', s=0.8)
        sdfs.finish_debug_image_figure(figure_title, 'geometry', fig_rec, debug)

    # Plot cleaned-up mask
    if debug.debug_active:
        figure_title = "Cleaned-Up Mask"
        fig_rec = sdfs.start_debug_image_figure(figure_title)
        fig_rec.view.imshow(mask_clean, cmap="gray")
        sdfs.finish_debug_image_figure(figure_title, 'geometry', fig_rec, debug)

    # Find centroid of processed mask
    v_mask_centroid_image = ip.centroid_mask(mask_clean)
    data_image_processing_general.v_mask_centroid_image = v_mask_centroid_image

    # Plot centroid
    if debug.debug_active:
        figure_title = "Mask Centroid"
        fig_rec = sdfs.start_debug_image_figure(figure_title)
        fig_rec.view.imshow(mask_clean, cmap="gray")
        fig_rec.view.axis.scatter(*v_mask_centroid_image.data, marker="x", c='red')
        sdfs.finish_debug_image_figure(figure_title, 'geometry', fig_rec, debug)

    # Plot mirror pose before translation fit.
    if True:  # debug.debug_active:  # &&&& DELETE-SCAFFOLDING -- TEMPORARY
        figure_title = "Mirror Location, Before Translation or Rotation"
        pogdo.figure_setup_before_mirror_translation_fit(figure_title, camera, facet_data, orientation, debug)

    # Plot optic corners, before translation or rotation.
    if debug.debug_active:
        v_optic_corners_image_0 = camera.project(v_facet_corners, Rotation.identity(), Vxyz([0, 0, 0]))
        figure_title = "Expected Optic Corners, Before Translation or Rotation"
        fig_rec = sdfs.start_debug_image_figure(figure_title)
        fig_rec.view.imshow(mask_clean, cmap="gray")
        sdfs.plot_labeled_points(v_optic_corners_image_0)
        sdfs.finish_debug_image_figure(figure_title, 'geometry', fig_rec, debug)

    # Find expected 3-d position of optic centroid, in camera coordinates.
    # &&&& DELETE-SCAFFOLDING -- RENAME "v_cam_optic_centroid_cam_exp" TO "v_cam_optic_origin_cam_exp"
    v_cam_optic_centroid_cam_exp = sp.t_from_distance(
        v_mask_centroid_image, dist_optic_screen, camera, ori.v_cam_screen_cam
    )
    data_geometry_general.v_cam_optic_centroid_cam_exp = v_cam_optic_centroid_cam_exp

    # Plot expected centroid
    if debug.debug_active:
        figure_title = "Expected Optic Centroid"
        fig_rec = sdfs.start_debug_image_figure(figure_title)
        fig_rec.view.imshow(mask_clean, cmap="gray")
        fig_rec.view.axis.scatter(*v_mask_centroid_image.data, marker="x", c='red', s=45, label='Mask Centroid')
        expected_centroid = camera.project(v_cam_optic_centroid_cam_exp, Rotation.identity(), Vxyz((0, 0, 0)))
        fig_rec.view.axis.scatter(*expected_centroid.data, marker=".", c='cyan', s=35, label='Expected Centroid')
        fig_rec.view.axis.legend()
        sdfs.finish_debug_image_figure(figure_title, 'geometry', fig_rec, debug)

    # Plot mirror pose with translation only fit.
    if True:  # debug.debug_active:  # &&&& DELETE-SCAFFOLDING -- TEMPORARY
        figure_title = "Mirror Location, Translated Only, No Rotation"
        pogdo.figure_setup_mirror_translation_only_fit(
            figure_title, camera, facet_data, orientation, debug, v_cam_optic_centroid_cam_exp, v_facet_centroid
        )

    # Plot optic corners, with translation only fit.
    if debug.debug_active:
        figure_title = "Expected Optic Corners, Translated Only, No Rotation"
        fig_rec = sdfs.start_debug_image_figure(figure_title)
        fig_rec.view.imshow(mask_clean, cmap="gray")
        # Centroid measured in image.
        fig_rec.view.axis.scatter(*v_mask_centroid_image.data, marker="x", c='red', s=55, label='Mask Centroid')
        # 3-d position of centroid in camera coordinates, projected back into image.
        expected_centroid_1 = camera.project(v_cam_optic_centroid_cam_exp, Rotation.identity(), Vxyz((0, 0, 0)))
        fig_rec.view.axis.scatter(*expected_centroid_1.data, marker=".", c='cyan', s=45, label='Expected Centroid')
        # The variable "v_cam_optic_centroid_cam_exp_1" is the estimated 3-d positon of the optic centroid, in camera coordinates.
        # It also may be viewed as a vector from the front nodal point of the camera to the optic centroid.
        # If we add this vector to all the facet corners, we will obtain an estimate of their position in 3-d space.
        #
        # That's not quite right, because adding the vector will place the facet corners relative of the facet centroid.
        # But if the facet centroid is not at the origin of the facet coordinate system (the general case), then these
        # will be offset by the vector from the facet origin to the facet centroid.  So our translation should be the
        # vector from the camera front nodal point to the facet centroid, minus the vector from the facet origin to
        # the facet centroid.
        #
        # Then we can project and render them using the same method as used for the centroid.
        # Note this is for illustration purposes only; this result is not useful for the computation.
        #
        # Vector from camera front nodal point to 3-d optic centroid position, and then to the optic origin,
        # if there were no rotation.
        # In other words, position of optic origin in 3-d space, in camera coordinates.
        translation = v_cam_optic_centroid_cam_exp - v_facet_centroid
        # Computed 3-d position of facet corners in camera coordinates, assuming no rotation.
        v_facet_corners_cam = v_facet_corners + translation
        # 3-d facet corner positions, projected back into image.
        v_optic_corners_image_1 = camera.project(v_facet_corners_cam, Rotation.identity(), Vxyz((0, 0, 0)))
        sdfs.plot_labeled_points(v_optic_corners_image_1)
        # Treat centroid the same way, for cross-check.
        # Re-computed 3-d position of facet centroid in camera coordinates, assuming no rotation.
        v_facet_centroid_cam = v_facet_centroid + translation
        # Re-computed 3-d facet centroid position, projected back into image.
        expected_centroid_1b = camera.project(v_facet_centroid_cam, Rotation.identity(), Vxyz((0, 0, 0)))
        fig_rec.view.axis.scatter(
            *expected_centroid_1b.data, marker="+", c='blue', s=35, label='Centroid from Translation'
        )
        fig_rec.view.axis.legend()
        sdfs.finish_debug_image_figure(figure_title, 'geometry', fig_rec, debug)

    # Find expected orientation of optic, assuming that reflection at centroid shows the cross-hair center during alignment.
    # This produces the rotation for the optic centroid, not the optic coordinate system.
    r_cam_optic_exp_A = sp.r_from_position(v_cam_optic_centroid_cam_exp, ori.v_cam_screen_cam)

    # Plot mirror pose with translation and rotation considered, but not centroid surface normal.
    if True:  # debug.debug_active:  # &&&& DELETE-SCAFFOLDING -- TEMPORARY
        figure_title = "Mirror Location, Translated and Rotated, without Centroid Normal"
        pogdo.figure_setup_mirror_fit_translation_rotation_but_not_normal(
            figure_title, camera, facet_data, orientation, debug
        )

    # Plot optic corners, rotated but without consideration of surface normal at centroid.
    if debug.debug_active:
        figure_title = "Expected Corners, Translated and Rotated, without Centroid Normal"
        fig_rec = sdfs.start_debug_image_figure(figure_title)
        fig_rec.view.imshow(mask_clean, cmap="gray")
        # Centroid measured in image.
        fig_rec.view.axis.scatter(*v_mask_centroid_image.data, marker="x", c='red', s=65, label='Mask Centroid')
        # Original computation of 3-d position of centroid in camera coordinates, projected back into image.
        expected_centroid_2 = camera.project(v_cam_optic_centroid_cam_exp, Rotation.identity(), Vxyz((0, 0, 0)))
        fig_rec.view.axis.scatter(*expected_centroid_2.data, marker=".", c='cyan', s=55, label='Expected Centroid')

        # In other words, position of optic origin in 3-d space, in camera coordinates.
        translation_2 = v_cam_optic_centroid_cam_exp - v_facet_centroid.rotate(r_cam_optic_exp_A)
        # Computed 3-d position of facet corners in camera coordinates.
        v_facet_corners_cam_2 = v_facet_corners.rotate(r_cam_optic_exp_A) + translation_2
        # 3-d facet corner positions, projected back into image.
        v_optic_corners_image_2 = camera.project(v_facet_corners_cam_2, Rotation.identity(), Vxyz((0, 0, 0)))
        sdfs.plot_labeled_points(v_optic_corners_image_2, legend_label='Points Using Identity Camera')
        # Treat centroid the same way, for cross-check.
        # Re-computed 3-d position of facet centroid in camera coordinates.
        v_facet_centroid_cam_2 = v_facet_centroid.rotate(r_cam_optic_exp_A) + translation_2
        # Re-computed 3-d facet centroid position, projected back into image.
        expected_centroid_2b = camera.project(v_facet_centroid_cam_2, Rotation.identity(), Vxyz((0, 0, 0)))
        fig_rec.view.axis.scatter(
            *expected_centroid_2b.data, marker="+", c='blue', s=35, label='Centroid Using Identity Camera'
        )

        # Position of optic origin in 3-d space, in camera coordinates, including rotation.
        v_cam_optic_cam_2b = v_cam_optic_centroid_cam_exp - v_facet_centroid.rotate(r_cam_optic_exp_A)
        # Expected positions of optic corners in the image, using the full [rotation, translation] camera transform,
        # but so far only considering part A of the rotation analysis -- not yet taking into account the optic
        # surface normal at the centroid.
        v_optic_corners_image_2b = camera.project(v_facet_corners, r_cam_optic_exp_A, v_cam_optic_cam_2b)
        sdfs.plot_labeled_points(
            v_optic_corners_image_2b, marker_size=10, point_color='k', legend_label='Points Using Camera Pose'
        )
        # Treat centroid the same way, for cross-check.
        # Re-computed 3-d facet centroid position in the image, using the full [rotation, translation] camera transform,
        # so far only considering part A of the rotation analysis.
        expected_centroid_2b = camera.project(v_facet_centroid, r_cam_optic_exp_A, v_cam_optic_cam_2b)
        fig_rec.view.axis.scatter(
            *expected_centroid_2b.data, marker="+", c='magenta', s=10, label='Centroid Using Camera Pose'
        )
        fig_rec.view.axis.legend()
        sdfs.finish_debug_image_figure(figure_title, 'geometry', fig_rec, debug)

    # Add consideration of surface normal at optic centroid.
    # Surface normal at facet origin
    z_axis = Uxyz([0.0, 0.0, 1.0])
    rotate_to_surface_normal = u_facet_centroid_normal.align_to(z_axis)
    r_cam_optic_exp = r_cam_optic_exp_A * rotate_to_surface_normal
    data_geometry_general.r_optic_cam_exp = r_cam_optic_exp.inv()

    # Find expected position of optic origin
    v_cam_optic_cam_exp = v_cam_optic_centroid_cam_exp - v_facet_centroid.rotate(r_cam_optic_exp.inv())
    data_geometry_general.v_cam_optic_cam_exp = v_cam_optic_cam_exp

    # Find expected optic vertices
    v_optic_corners_image_exp = camera.project(v_facet_corners, r_cam_optic_exp.inv(), v_cam_optic_cam_exp)

    # Plot expected optic corners, rotated including consideration of surface normal at centroid.
    if debug.debug_active:
        figure_title = "Expected Corners, Translated and Rotated, including Centroid Normal"
        fig_rec = sdfs.start_debug_image_figure(figure_title)
        fig_rec.view.imshow(mask_clean, cmap="gray")
        # Centroid measured in image.
        fig_rec.view.axis.scatter(*v_mask_centroid_image.data, marker="x", c='red', s=65, label='Mask Centroid')
        # Original computation of 3-d position of centroid in camera coordinates, projected back into image.
        expected_centroid = camera.project(v_cam_optic_centroid_cam_exp, Rotation.identity(), Vxyz((0, 0, 0)))
        fig_rec.view.axis.scatter(*expected_centroid.data, marker=".", c='cyan', s=55, label='Expected Centroid')

        # Position of optic origin in 3-d space, in camera coordinates.
        translation_3 = v_cam_optic_centroid_cam_exp - v_facet_centroid.rotate(r_cam_optic_exp.inv())
        # Computed 3-d position of facet corners in camera coordinates.
        v_facet_corners_cam_3 = v_facet_corners.rotate(r_cam_optic_exp.inv()) + translation_3
        # 3-d facet corner positions, projected back into image.
        v_optic_corners_image_3 = camera.project(v_facet_corners_cam_3, Rotation.identity(), Vxyz((0, 0, 0)))
        sdfs.plot_labeled_points(v_optic_corners_image_3, legend_label='Points Using Identity Camera')
        # Treat centroid the same way, for cross-check.
        # Re-computed 3-d position of facet centroid in camera coordinates.
        v_facet_centroid_cam_3 = v_facet_centroid.rotate(r_cam_optic_exp.inv()) + translation_3
        # Re-computed 3-d facet centroid position, projected back into image.
        expected_centroid_b = camera.project(v_facet_centroid_cam_3, Rotation.identity(), Vxyz((0, 0, 0)))
        fig_rec.view.axis.scatter(
            *expected_centroid_b.data, marker="+", c='blue', s=35, label='Centroid Using Identity Camera'
        )

        # Position of optic origin in 3-d space, in camera coordinates, including rotation.
        # Expected positions of optic corners in the image, using the full [rotation, translation] camera transform.
        sdfs.plot_labeled_points(
            v_optic_corners_image_exp, marker_size=10, point_color='k', legend_label='Points Using Camera Pose'
        )
        # Treat centroid the same way, for cross-check.
        # Re-computed 3-d facet centroid position in the image, using the full [rotation, translation] camera transform.
        expected_centroid_3b = camera.project(v_facet_centroid, r_cam_optic_exp.inv(), v_cam_optic_cam_exp)
        fig_rec.view.axis.scatter(
            *expected_centroid_3b.data, marker="+", c='magenta', s=10, label='Centroid Using Camera Pose'
        )
        fig_rec.view.axis.legend()
        sdfs.finish_debug_image_figure(figure_title, 'geometry', fig_rec, debug)

    # Construct expected optic loop in pixels
    loop_optic_image_exp = LoopXY.from_vertices(v_optic_corners_image_exp)
    data_image_processing_general.loop_optic_image_exp = loop_optic_image_exp

    # Plot expected optic loop
    if debug.debug_active:
        figure_title = "Expected Optic Loop"
        fig_rec = sdfs.start_debug_image_figure(figure_title)
        fig_rec.view.imshow(mask_clean, cmap="gray")
        # Centroid measured in image.
        fig_rec.view.axis.scatter(*v_mask_centroid_image.data, marker="x", c='red', s=65, label='Mask Centroid')
        # Expected positions of optic corners in the image.
        sdfs.plot_labeled_points(v_optic_corners_image_exp, legend_label='Points Using Camera Pose')
        fig_rec.view.draw_pq_list(
            loop_optic_image_exp.as_xy_list(), close=True, style=rcps.default(marker='arrow', color='green')
        )
        # Expected position of optic centroid in the image.
        expected_centroid_4b = camera.project(v_facet_centroid, r_cam_optic_exp.inv(), v_cam_optic_cam_exp)
        fig_rec.view.axis.scatter(
            *expected_centroid_4b.data, marker="+", c='k', s=20, label='Centroid Using Camera Pose'
        )
        fig_rec.view.axis.legend()
        sdfs.finish_debug_image_figure(figure_title, 'geometry', fig_rec, debug)

    # Refine locations of optic corners with mask
    try:
        prs = [params.perimeter_refine_axial_search_dist, params.perimeter_refine_perpendicular_search_dist]
        loop_facet_image_refine = ip.refine_mask_perimeter(debug, mask_clean, loop_optic_image_exp, v_edges_image, *prs)
        data_image_processing_facet.loop_facet_image_refine = loop_facet_image_refine
    except ValueError as er:
        lt.critical(repr(er))
        lt.error_and_raise(ValueError, "SOFAST failed to find the corners of the optic.")

    # Plot refined optic corners
    if debug.debug_active:
        figure_title = "Refined Optic Loop, Clean Mask"
        fig_rec = sdfs.start_debug_image_figure(figure_title)
        fig_rec.view.imshow(mask_clean, cmap="gray")
        # Centroid measured in image.
        fig_rec.view.axis.scatter(*v_mask_centroid_image.data, marker="x", c='red', s=65, label='Clean Mask Centroid')
        # Refined positions of optic corners in the image.
        sdfs.plot_labeled_points(loop_facet_image_refine.vertices, legend_label='Refined Points Using Camera Pose')
        fig_rec.view.draw_pq_list(
            loop_facet_image_refine.as_xy_list(), close=True, style=rcps.default(marker='arrow', color='magenta')
        )
        # Expected position of optic centroid in the image.
        expected_centroid_4b = camera.project(v_facet_centroid, r_cam_optic_exp.inv(), v_cam_optic_cam_exp)
        fig_rec.view.axis.scatter(
            *expected_centroid_4b.data, marker="+", c='k', s=20, label='Centroid Using Camera Pose'
        )
        fig_rec.view.axis.legend()
        sdfs.finish_debug_image_figure(figure_title, 'geometry', fig_rec, debug)

    # Plot refined optic corners, against original mask
    if debug.debug_active:
        figure_title = "Refined Optic Loop, Original Mask"
        fig_rec = sdfs.start_debug_image_figure(figure_title)
        fig_rec.view.imshow(mask_raw, cmap="gray")
        # Centroid measured in image.
        fig_rec.view.axis.scatter(*v_mask_centroid_image.data, marker="x", c='red', s=65, label='Clean Mask Centroid')
        # Expected position of optic centroid in the image.
        expected_centroid_4b = camera.project(v_facet_centroid, r_cam_optic_exp.inv(), v_cam_optic_cam_exp)
        fig_rec.view.axis.scatter(
            *expected_centroid_4b.data, marker="+", c='k', s=20, label='Centroid Using Camera Pose'
        )
        # Refined positions of optic corners in the image.
        sdfs.plot_labeled_points(loop_facet_image_refine.vertices, legend_label='Refined Points Using Camera Pose')
        fig_rec.view.draw_pq_list(
            loop_facet_image_refine.as_xy_list(), close=True, style=rcps.default(marker='arrow', color='magenta')
        )
        fig_rec.view.axis.legend()
        sdfs.finish_debug_image_figure(figure_title, 'geometry', fig_rec, debug)

    # Create fitted mask
    vx = np.arange(mask_clean.shape[1])
    vy = np.arange(mask_clean.shape[0])
    mask_fitted = loop_facet_image_refine.as_mask(vx, vy)
    data_image_processing_facet.mask_fitted = mask_fitted  # &&&& DELETE-SCAFFOLDING -- HOW IS THIS USED?

    # Remove non-active pixels from mask
    mask_processed = np.logical_and(mask_fitted, mask_clean)
    data_image_processing_facet.mask_processed = mask_processed  # &&&& DELETE-SCAFFOLDING -- HOW IS THIS USED?

    # Plot the four masks for comparison.
    if debug.debug_active:
        # Raw
        figure_title = "Raw Mask"
        fig_rec = sdfs.start_debug_image_figure(figure_title)
        fig_rec.view.imshow(mask_raw, cmap="gray")
        sdfs.finish_debug_image_figure(figure_title, 'geometry', fig_rec, debug)
        # Cleaned up
        figure_title = "Cleaned-Up Mask"
        fig_rec = sdfs.start_debug_image_figure(figure_title)
        fig_rec.view.imshow(mask_clean, cmap="gray")
        sdfs.finish_debug_image_figure(figure_title, 'geometry', fig_rec, debug)
        # Fitted
        figure_title = "Fitted Mask"
        fig_rec = sdfs.start_debug_image_figure(figure_title)
        fig_rec.view.imshow(mask_fitted, cmap="gray")
        sdfs.finish_debug_image_figure(figure_title, 'geometry', fig_rec, debug)
        # Processed
        figure_title = "Processed Mask"
        fig_rec = sdfs.start_debug_image_figure(figure_title)
        fig_rec.view.imshow(mask_processed, cmap="gray")
        sdfs.finish_debug_image_figure(figure_title, 'geometry', fig_rec, debug)

    # Calculate R/T from found corners
    r_optic_cam_refine_1, v_cam_optic_cam_refine_1 = sp.calc_rt_from_img_pts(
        loop_facet_image_refine.vertices, v_facet_corners, camera
    )
    r_cam_optic_refine_1 = r_optic_cam_refine_1.inv()
    data_geometry_general.r_optic_cam_refine_1 = r_optic_cam_refine_1
    data_geometry_general.v_cam_optic_cam_refine_1 = v_cam_optic_cam_refine_1

    # Plot reprojected points 1
    if debug.debug_active:
        fig = plt.figure()
        debug.figures.append(fig)
        plt.imshow(mask_raw)
        pts_reproj = camera.project(facet_data.v_facet_corners, r_cam_optic_refine_1.inv(), v_cam_optic_cam_refine_1)
        sdfs.plot_labeled_points(pts_reproj)
        plt.title("Reprojected Points 1")

    # Plot reprojected points in comparison to refined optic corners
    if debug.debug_active:
        figure_title = "Refined Loop Points vs. Reprojected Points 1"
        fig_rec = sdfs.start_debug_image_figure(figure_title)
        fig_rec.view.imshow(mask_raw, cmap="gray")
        # Centroid measured in image.
        fig_rec.view.axis.scatter(*v_mask_centroid_image.data, marker="x", c='red', s=65, label='Clean Mask Centroid')
        # Expected position of optic centroid in the image.
        expected_centroid_4b = camera.project(v_facet_centroid, r_cam_optic_exp.inv(), v_cam_optic_cam_exp)
        fig_rec.view.axis.scatter(
            *expected_centroid_4b.data, marker="+", c='k', s=20, label='Centroid Using Camera Pose'
        )
        # Refined positions of optic corners in the image.
        sdfs.plot_labeled_points(loop_facet_image_refine.vertices, legend_label='Refined Points Using Camera Pose')
        fig_rec.view.draw_pq_list(
            loop_facet_image_refine.as_xy_list(), close=True, style=rcps.default(marker='arrow', color='magenta')
        )
        # Points reprojected using camera pose from solvePnP()
        pts_reproj = camera.project(facet_data.v_facet_corners, r_cam_optic_refine_1.inv(), v_cam_optic_cam_refine_1)
        sdfs.plot_labeled_points(
            pts_reproj, marker_size=30, point_color='b', label_color='b', legend_label='Reprojected by solvePnP()'
        )
        fig_rec.view.axis.legend()
        sdfs.finish_debug_image_figure(figure_title, 'geometry', fig_rec, debug)

    # Calculate refined measure point vector in optic coordinates
    v_measure_point_optic_cam_refine_1 = v_measure_point_facet.rotate(r_optic_cam_refine_1)

    if debug.debug_active:
        lt.info('In process_singlefacet_geometry():')
        lt.info('  dist_optic_screen        = ' + str(dist_optic_screen))
        lt.info('  ori.v_cam_screen_cam     = ' + str(ori.v_cam_screen_cam))
        lt.info('  v_measure_point_facet    = ' + str(v_measure_point_facet))
        lt.info('  v_facet_centroid         = ' + str(v_facet_centroid))
        lt.info('  v_cam_optic_cam_exp      = ' + str(v_cam_optic_cam_exp))
        lt.info('  v_cam_optic_cam_refine_1 = ' + str(v_cam_optic_cam_refine_1))
        lt.info('  r_cam_optic_exp          = ' + str(r_cam_optic_exp.as_euler('XYZ', degrees=True)))
        lt.info('  r_optic_cam_refine_1     = ' + str(r_optic_cam_refine_1.as_euler('XYZ', degrees=True)))

    # Refine V with measured optic to display distance
    v_cam_optic_cam_refine_2 = sp.refine_v_distance(
        v_cam_optic_cam_refine_1, dist_optic_screen, ori.v_cam_screen_cam, v_measure_point_optic_cam_refine_1
    )
    data_geometry_general.v_cam_optic_cam_refine_2 = v_cam_optic_cam_refine_2

    # Plot reprojected points 2
    if debug.debug_active:
        fig = plt.figure()
        debug.figures.append(fig)
        plt.imshow(mask_raw)
        pts_reproj = camera.project(facet_data.v_facet_corners, r_cam_optic_refine_1.inv(), v_cam_optic_cam_refine_2)
        sdfs.plot_labeled_points(pts_reproj)
        plt.title("Reprojected Points 2")

    # Orient optic
    ori.orient_optic_cam(r_cam_optic_refine_1, v_cam_optic_cam_refine_2)

    # Calculate measure point pointing direction
    u_cam_measure_point_facet = Uxyz((ori.v_cam_optic_optic + v_measure_point_facet).data)
    data_geometry_facet.u_cam_measure_point_facet = u_cam_measure_point_facet

    # Calculate errors from using only facet corners
    error_dist_optic_screen_1 = sp.distance_error(
        ori.v_cam_screen_cam, v_cam_optic_cam_refine_1 + v_measure_point_optic_cam_refine_1, dist_optic_screen
    )
    data_error.error_dist_optic_screen_1 = error_dist_optic_screen_1
    error_reprojection_1 = sp.reprojection_error(
        camera, v_facet_corners, loop_facet_image_refine.vertices, r_optic_cam_refine_1, v_cam_optic_cam_refine_1
    )
    data_error.error_reprojection_1 = error_reprojection_1

    # Calculate errors after refining with measured distance
    error_dist_optic_screen_2 = sp.distance_error(
        ori.v_cam_screen_cam, v_cam_optic_cam_refine_2 + v_measure_point_optic_cam_refine_1, dist_optic_screen
    )
    data_error.error_dist_optic_screen_2 = error_dist_optic_screen_2
    error_reprojection_2 = sp.reprojection_error(
        camera, v_facet_corners, loop_facet_image_refine.vertices, r_optic_cam_refine_1, v_cam_optic_cam_refine_2
    )
    data_error.error_reprojection_2 = error_reprojection_2

    # Save other data
    data_geometry_facet.measure_point_screen_distance = dist_optic_screen
    data_geometry_facet.spatial_orientation = ori
    data_geometry_facet.v_align_point_facet = v_facet_centroid

    return (
        data_geometry_general,
        data_image_processing_general,
        [data_geometry_facet],
        [data_image_processing_facet],
        data_error,
    )


def process_undefined_geometry(
    mask_raw: ndarray,
    mask_keep_largest_area: bool,
    dist_optic_screen: float,
    orientation: SpatialOrientation,
    camera: Camera,
    debug: DebugOpticsGeometry = DebugOpticsGeometry(),
) -> tuple[
    cdc.CalculationDataGeometryGeneral,
    cdc.CalculationImageProcessingGeneral,
    list[cdc.CalculationDataGeometryFacet],
    list[cdc.CalculationImageProcessingFacet],
    cdc.CalculationError,
]:
    """Processes optic geometry for undefined deflectometry measurement

    Parameters
    ----------
    mask_raw : ndarray
        Raw calculated mask
    mask_keep_largest_area : bool
        To apply the "keep largest area" mask operation
    dist_optic_screen : float
        Optic centroid to screen distance, meters
    orientation : SpatialOrientation
        SpatialOrientation object
    camera : Camera
        Camera object
    debug : DebugOpticsGeometry, optional
        DebugOpticsGeometry object, by default DebugOpticsGeometry()

    Returns
    -------
    data_geometry_general: calculation_data_classes.CalculationDataGeometryGeneral
        Positional optic geometry calculations general to entire measurement; not facet specific.
    data_image_processing_general: calculation_data_classes.CalculationImageProcessingGeneral
        Image processing calculations general to entire measurement; not facet specific.
    data_geometry_facet: list[calculation_data_classes.CalculationDataGeometryFacet]
        List of positional optic geometry calculations specific to each facet. Order is
        same as input facet definitions.
    data_image_processing_facet: list[calculation_data_classes.CalculationImageProcessingFacet]
        List of image processing calcualtions specific to each facet. Order is same as input facet
        definitions.
    data_error: calculation_data_classes.CalculationError
        Geometric/positional errors and reprojection errors associated with solving for facet location.
    """
    if debug.debug_active:
        lt.debug("process_optics_geometry debug on, but is not yet supported for undefined mirrors.")

    # Define data classes
    data_geometry_general = cdc.CalculationDataGeometryGeneral()
    data_image_processing_general = cdc.CalculationImageProcessingGeneral()
    data_geometry_facet = cdc.CalculationDataGeometryFacet()
    data_image_processing_facet = cdc.CalculationImageProcessingFacet()
    data_error = None

    # Save mask raw
    data_image_processing_general.mask_raw = mask_raw

    # If enabled, keep only the largest mask area
    if mask_keep_largest_area:
        mask_raw_proc = ip.keep_largest_mask_area(mask_raw)
        mask_processed = np.logical_and(mask_raw, mask_raw_proc)
    else:
        mask_processed = mask_raw.copy()

    data_image_processing_facet.mask_processed = mask_processed

    # Find centroid of processed mask
    v_mask_centroid_image = ip.centroid_mask(mask_processed)
    data_image_processing_general.v_mask_centroid_image = v_mask_centroid_image

    # Find position of optic centroid in space
    v_cam_optic_cam = sp.t_from_distance(v_mask_centroid_image, dist_optic_screen, camera, orientation.v_cam_screen_cam)
    data_geometry_general.v_cam_optic_cam_exp = v_cam_optic_cam

    # Find orientation of optic
    r_cam_optic = sp.r_from_position(v_cam_optic_cam, orientation.v_cam_screen_cam)
    data_geometry_general.r_optic_cam_exp = r_cam_optic.inv()

    # Orient optic
    spatial_orientation = SpatialOrientation(orientation.r_cam_screen, orientation.v_cam_screen_cam)
    spatial_orientation.orient_optic_cam(r_cam_optic, v_cam_optic_cam)

    # Calculate measure point pointing direction
    u_cam_measure_point_facet = Uxyz(spatial_orientation.v_cam_optic_optic.data)

    # Save processed optic data
    data_geometry_facet.u_cam_measure_point_facet = u_cam_measure_point_facet
    data_geometry_facet.measure_point_screen_distance = dist_optic_screen
    data_geometry_facet.spatial_orientation = spatial_orientation
    data_geometry_facet.v_align_point_facet = Vxyz((0, 0, 0))

    return (
        data_geometry_general,
        data_image_processing_general,
        [data_geometry_facet],
        [data_image_processing_facet],
        data_error,
    )


def process_multifacet_geometry(
    facet_data: list[DefinitionFacet],
    ensemble_data: DefinitionEnsemble,
    mask_raw: ndarray,
    v_meas_pt_ensemble: Vxyz,
    orientation: SpatialOrientation,
    camera: Camera,
    dist_optic_screen: float,
    params_geometry: ParamsOpticGeometry = ParamsOpticGeometry(),
    params_mask: ParamsMaskCalculation = ParamsMaskCalculation(),
    debug: DebugOpticsGeometry = DebugOpticsGeometry(),
) -> tuple[
    cdc.CalculationDataGeometryGeneral,
    cdc.CalculationImageProcessingGeneral,
    list[cdc.CalculationDataGeometryFacet],
    list[cdc.CalculationImageProcessingFacet],
    cdc.CalculationError,
]:
    """Processes optic geometry for multifacet deflectometry measurement

    Parameters
    ----------
    facet_data : DefinitionFacet
        Facet definition object
    ensemble_data : DefinitionEnsemble
        Ensemble definition object
    mask_raw : ndarray
        Raw calculated mask, shape (m, n) array of booleans
    v_meas_pt_ensemble : Vxyz
        Measure point lcoation on ensemble, meters
    orientation : SpatialOrientation
        SpatialOrientation object
    camera : Camera
        Camera object
    dist_optic_screen : float
        Optic to screen distance, meters
    params_geometry : ParamsOpticGeometry, optional
        ParamsOpticGeometry object, by default ParamsOpticGeometry()
    params_mask : ParamsMaskCalculation, optional
        ParamsMaskCalculation object, by default ParamsMaskCalculation()
    debug : DebugOpticsGeometry, optional
        DebugOpticsGeometry object, by default DebugOpticsGeometry()

    Returns
    -------
    data_geometry_general: calculation_data_classes.CalculationDataGeometryGeneral
        Positional optic geometry calculations general to entire measurement; not facet specific.
    data_image_processing_general: calculation_data_classes.CalculationImageProcessingGeneral
        Image processing calculations general to entire measurement; not facet specific.
    data_geometry_facet: list[calculation_data_classes.CalculationDataGeometryFacet]
        List of positional optic geometry calculations specific to each facet. Order is
        same as input facet definitions.
    data_image_processing_facet: list[calculation_data_classes.CalculationImageProcessingFacet]
        List of image processing calcualtions specific to each facet. Order is same as input facet
        definitions.
    data_error: calculation_data_classes.CalculationError
        Geometric/positional errors and reprojection errors associated with solving for facet location.
    """
    if debug.debug_active:
        lt.debug("process_optics_geometry debug on.")

    # Get facet data
    v_facet_corners_facet: list = [
        f.v_facet_corners for f in facet_data
    ]  # Location of facet corners in facet coordinates
    v_facet_centroid_facet: list = [
        f.v_facet_centroid for f in facet_data
    ]  # Location of facet centroids in facet coordinates

    # Get ensemble data
    v_facet_locs_ensemble = (
        ensemble_data.v_facet_locations
    )  # Locations of facet origins relative to ensemble origin in ensemble coordinates
    r_facet_ensemble = ensemble_data.r_facet_ensemble  # Facet to ensemble rotation
    ensemble_corns_indices = ensemble_data.ensemble_perimeter  # [(facet_idx, facet_corner_idx), ...], integers
    v_centroid_ensemble = ensemble_data.v_centroid_ensemble  # Centroid of ensemble in ensemble coordinates

    # Get number of facets
    num_facets = len(v_facet_locs_ensemble)

    # Define data classes
    data_geometry_general = cdc.CalculationDataGeometryGeneral()
    data_image_processing_general = cdc.CalculationImageProcessingGeneral()
    data_geometry_facet = [cdc.CalculationDataGeometryFacet() for _ in range(num_facets)]
    data_image_processing_facet = [cdc.CalculationImageProcessingFacet() for _ in range(num_facets)]
    data_error = cdc.CalculationError()

    # Convert facet corners to ensemble coordinates
    v_ensemble_facet_corns = []
    for idx in range(num_facets):
        v_ensemble_facet_corns.append(
            v_facet_locs_ensemble[idx] + v_facet_corners_facet[idx].rotate(r_facet_ensemble[idx])
        )

    # Calculate ensemble corners in ensemble coordinates
    v_ensemble_corns_ensemble = []
    for idx_facet, idx_corn in ensemble_corns_indices:
        r_facet_ensemble_cur = r_facet_ensemble[idx_facet]
        v_ensemble_corns_ensemble.append(
            (
                v_facet_locs_ensemble[idx_facet]
                + v_facet_corners_facet[idx_facet][idx_corn].rotate(r_facet_ensemble_cur)
            ).data
        )
    v_ensemble_corns_ensemble = Vxyz(np.concatenate(v_ensemble_corns_ensemble, axis=1))

    # Concatenate all facet corners
    v_ensemble_facet_corns_all = Vxyz(np.concatenate([V.data for V in v_ensemble_facet_corns], axis=1))

    # Calculate raw mask
    data_image_processing_general.mask_raw = mask_raw

    # Plot mask
    if debug.debug_active:
        fig = plt.figure()
        debug.figures.append(fig)
        plt.imshow(mask_raw, cmap="gray")
        plt.title("Raw Mask")

    # Find edges of mask
    v_edges_image = ip.edges_from_mask(mask_raw)
    data_image_processing_general.v_edges_image = v_edges_image

    # Calculate centroid of mask
    v_mask_centroid_image = ip.centroid_mask(mask_raw)
    data_image_processing_general.v_mask_centroid_image = v_mask_centroid_image

    # Calculate expected position of ensemble centroid
    v_cam_ensemble_cent_cam_exp = sp.t_from_distance(
        v_mask_centroid_image, dist_optic_screen, camera, orientation.v_cam_screen_cam
    )
    data_geometry_general.v_cam_optic_centroid_cam_exp = v_cam_ensemble_cent_cam_exp

    # Calculate expected orientation of facet ensemble
    r_cam_ensemble_exp = sp.r_from_position(v_cam_ensemble_cent_cam_exp, orientation.v_cam_screen_cam)
    data_geometry_general.r_optic_cam_exp = r_cam_ensemble_exp.inv()

    # Calculate expected position of ensemble origin
    v_cam_ensemble_cam_exp = v_cam_ensemble_cent_cam_exp - v_centroid_ensemble.rotate(r_cam_ensemble_exp.inv())
    data_geometry_general.v_cam_optic_cam_exp = v_cam_ensemble_cam_exp

    # Project perimeter points
    v_ensemble_corners_exp_image = camera.project(
        v_ensemble_corns_ensemble, r_cam_ensemble_exp.inv(), v_cam_ensemble_cam_exp
    )
    loop_ensemble_exp = LoopXY.from_vertices(v_ensemble_corners_exp_image)
    data_image_processing_general.loop_optic_image_exp = loop_ensemble_exp

    # Plot expected perimeter points
    if debug.debug_active:
        fig = plt.figure()
        debug.figures.append(fig)
        plt.imshow(mask_raw)
        sdfs.plot_labeled_points(v_ensemble_corners_exp_image)
        plt.title("Expected Perimeter Points")

    # Refine perimeter points
    try:
        args = [
            params_geometry.perimeter_refine_axial_search_dist,
            params_geometry.perimeter_refine_perpendicular_search_dist,
        ]
        loop_ensemble_image_refine = ip.refine_mask_perimeter(debug, mask_raw, loop_ensemble_exp, v_edges_image, *args)
        data_image_processing_general.loop_optic_image_refine = loop_ensemble_image_refine
    except ValueError as er:
        lt.critical(repr(er))
        lt.error_and_raise(ValueError, "SOFAST failed to find the corners of the optic.")

    # Plot refined perimeter points
    if debug.debug_active:
        fig = plt.figure()
        debug.figures.append(fig)
        plt.imshow(mask_raw)
        sdfs.plot_labeled_points(loop_ensemble_image_refine.vertices)
        plt.title("Refined Perimeter Points")

    # Refine ensemble position/orientation with perimeter points
    r_ensemble_cam_refine_1, v_cam_ensemble_cam_refine_1 = sp.calc_rt_from_img_pts(
        loop_ensemble_image_refine.vertices, v_ensemble_corns_ensemble, camera
    )
    data_geometry_general.r_optic_cam_refine_1 = r_ensemble_cam_refine_1
    data_geometry_general.v_cam_optic_cam_refine_1 = v_cam_ensemble_cam_refine_1

    # Calculate refined measure point vector in optic coordinates
    v_meas_pt_ensemble_cam_refine_1 = v_meas_pt_ensemble.rotate(r_ensemble_cam_refine_1)

    # Calculate expected location of all facet corners and centroids
    v_facet_corners_image_exp = [
        camera.project(P, r_ensemble_cam_refine_1, v_cam_ensemble_cam_refine_1) for P in v_ensemble_facet_corns
    ]
    v_uv_facet_cent_exp = camera.project(v_facet_locs_ensemble, r_ensemble_cam_refine_1, v_cam_ensemble_cam_refine_1)
    for idx in range(num_facets):
        data_image_processing_facet[idx].v_facet_corners_image_exp = v_facet_corners_image_exp[idx]
        data_image_processing_facet[idx].v_facet_centroid_image_exp = v_uv_facet_cent_exp[idx]

    # Refine facet corners
    args = [
        params_geometry.facet_corns_refine_step_length,
        params_geometry.facet_corns_refine_perpendicular_search_dist,
        params_geometry.facet_corns_refine_frac_keep,
    ]
    loops_facets_refined: list[LoopXY] = []
    for idx in range(num_facets):
        try:
            loop = ip.refine_facet_corners(
                v_facet_corners_image_exp[idx], v_uv_facet_cent_exp[idx], v_edges_image, *args
            )
            loops_facets_refined.append(loop)
            data_image_processing_facet[idx].loop_facet_image_refine = loop
        except ValueError as er:
            lt.critical(repr(er))
            lt.error_and_raise(ValueError, "SOFAST failed to find the corners of the optic.")

        # Plot refined perimeter points
        if debug.debug_active:
            if idx == 0:
                fig = plt.figure()
                debug.figures.append(fig)
                plt.imshow(mask_raw)
                plt.title("Refined Facet Corners")
            loop.draw()

    # Concatenate all refined facet corners
    v_facet_corners_all_image_refine = []
    for loop in loops_facets_refined:
        v_facet_corners_all_image_refine.append(loop.vertices.data)
    v_facet_corners_all_image_refine = Vxy(np.concatenate(v_facet_corners_all_image_refine, axis=1))

    # Calculate fitted masks
    mask_fitted = np.zeros(mask_raw.shape + (num_facets,), dtype=bool)
    vx = np.arange(mask_raw.shape[1])
    vy = np.arange(mask_raw.shape[0])
    for idx in range(num_facets):
        mask_fitted[..., idx] = loops_facets_refined[idx].as_mask(vx, vy)
        data_image_processing_facet[idx].mask_fitted = mask_fitted[..., idx]

    # Calculate processed masks
    mask_processed = np.ones(mask_raw.shape + (num_facets,), dtype=bool)
    mask_processed *= mask_raw[..., np.newaxis]
    mask_processed = np.logical_and(mask_processed, mask_fitted)
    for idx in range(num_facets):
        mask = mask_processed[..., idx]
        # If enabled, keep largest mask area (fill holes) for each individual facet
        if params_mask.keep_largest_area:
            mask = ip.keep_largest_mask_area(mask)
        data_image_processing_facet[idx].mask_processed = mask

    # Refine R/T with all refined facet corners
    r_ensemble_cam_refine_2, v_cam_ensemble_cam_refine_2 = sp.calc_rt_from_img_pts(
        v_facet_corners_all_image_refine, v_ensemble_facet_corns_all, camera
    )
    r_cam_ensemble_refine_2 = r_ensemble_cam_refine_2.inv()
    data_geometry_general.r_optic_cam_refine_2 = r_ensemble_cam_refine_2
    data_geometry_general.v_cam_optic_cam_refine_2 = v_cam_ensemble_cam_refine_2

    # Calculate refined measure point location in optic coordinates vector
    v_meas_pt_ensemble_cam_refine_2 = v_meas_pt_ensemble.rotate(r_ensemble_cam_refine_2)

    # Refine T with measured distance
    v_cam_ensemble_cam_refine_3 = sp.refine_v_distance(
        v_cam_ensemble_cam_refine_2, dist_optic_screen, orientation.v_cam_screen_cam, v_meas_pt_ensemble_cam_refine_2
    )
    data_geometry_general.v_cam_optic_cam_refine_3 = v_cam_ensemble_cam_refine_3

    # Calculate error 1 (R/T calculated using only ensemble perimeter points)
    error_dist_optic_screen_1 = sp.distance_error(
        orientation.v_cam_screen_cam, v_cam_ensemble_cam_refine_1 + v_meas_pt_ensemble_cam_refine_1, dist_optic_screen
    )
    data_error.error_dist_optic_screen_1 = error_dist_optic_screen_1
    error_reprojection_1 = sp.reprojection_error(
        camera,
        v_ensemble_corns_ensemble,
        loop_ensemble_image_refine.vertices,
        r_ensemble_cam_refine_1,
        v_cam_ensemble_cam_refine_1,
    )
    data_error.error_reprojection_1 = error_reprojection_1

    # Calculate error 2 (R/T calculated using all facet corners)
    error_dist_optic_screen_2 = sp.distance_error(
        orientation.v_cam_screen_cam, v_cam_ensemble_cam_refine_2 + v_meas_pt_ensemble_cam_refine_2, dist_optic_screen
    )
    data_error.error_dist_optic_screen_2 = error_dist_optic_screen_2
    error_reprojection_2 = sp.reprojection_error(
        camera,
        v_ensemble_facet_corns_all,
        v_facet_corners_all_image_refine,
        r_ensemble_cam_refine_2,
        v_cam_ensemble_cam_refine_2,
    )
    data_error.error_reprojection_2 = error_reprojection_2

    # Calculate error 3 (T refined using measured distance)
    error_dist_optic_screen_3 = sp.distance_error(
        orientation.v_cam_screen_cam, v_cam_ensemble_cam_refine_3 + v_meas_pt_ensemble_cam_refine_2, dist_optic_screen
    )
    data_error.error_dist_optic_screen_3 = error_dist_optic_screen_3
    error_reprojection_3 = sp.reprojection_error(
        camera,
        v_ensemble_facet_corns_all,
        v_facet_corners_all_image_refine,
        r_cam_ensemble_refine_2,
        v_cam_ensemble_cam_refine_3,
    )
    data_error.error_reprojection_3 = error_reprojection_3

    # Spatially orient facets and the setup
    for idx in range(num_facets):
        # Calculate ensemble to facet vector in camera coordinates
        v_ensemble_facet_cam = v_facet_locs_ensemble[idx].rotate(r_ensemble_cam_refine_2)

        # Instantiate spatial orientation object
        facet_ori = SpatialOrientation(orientation.r_cam_screen, orientation.v_cam_screen_cam)

        # Orient facet
        r_cam_facet = r_facet_ensemble[idx].inv() * r_cam_ensemble_refine_2
        v_cam_facet_cam = v_cam_ensemble_cam_refine_3 + v_ensemble_facet_cam
        facet_ori.orient_optic_cam(r_cam_facet, v_cam_facet_cam)

        # Calculate facet measure point pointing direction (measure point defined as facet centroid)
        v_cam_meas_pt_facet = facet_ori.v_cam_optic_optic + v_facet_centroid_facet[idx]

        # Calculate facet measure point to screen distance
        v_cam_screen_optic = facet_ori.v_cam_screen_cam.rotate(facet_ori.r_cam_optic)
        dist = (v_cam_meas_pt_facet - v_cam_screen_optic).magnitude()[0]

        data_geometry_facet[idx].u_cam_measure_point_facet = Uxyz(v_cam_meas_pt_facet.data)
        data_geometry_facet[idx].measure_point_screen_distance = dist
        data_geometry_facet[idx].spatial_orientation = facet_ori
        data_geometry_facet[idx].v_align_point_facet = v_facet_centroid_facet[idx]

    return (
        data_geometry_general,
        data_image_processing_general,
        data_geometry_facet,
        data_image_processing_facet,
        data_error,
    )
