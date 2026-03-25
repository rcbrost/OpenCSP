"""Class that analyzes the configuration of a Sofast run. This includes:
- The physical system layout
- Statistics from a Sofast measurement
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from opencsp.app.sofast.lib.DefinitionFacet import DefinitionFacet
from opencsp.app.sofast.lib.DisplayShape import DisplayShape as Display
from opencsp.app.sofast.lib.DotLocationsFixedPattern import DotLocationsFixedPattern
from opencsp.app.sofast.lib.ProcessSofastFixed import ProcessSofastFixed
from opencsp.app.sofast.lib.ProcessSofastFringe import ProcessSofastFringe
from opencsp.app.sofast.lib.SpatialOrientation import SpatialOrientation
from opencsp.common.lib.deflectometry.Surface2DParabolic import Surface2DParabolic
from opencsp.common.lib.camera.Camera import Camera
import opencsp.common.lib.csp.embedding_mirror_surface as ems
from opencsp.common.lib.csp.MirrorParametric import MirrorParametric
import opencsp.common.lib.geometry.TransformXYZ as txyz
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.render.View3d import View3d
import opencsp.common.lib.render_control.RenderControlMirror as rcm
import opencsp.common.lib.render_control.RenderControlMirrorEmbedded as rcme
import opencsp.common.lib.render_control.RenderControlMirrorProjected as rcmp
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.render_control.RenderControlSofastWorld as rcsw
import opencsp.common.lib.render_control.RenderControlSofastScreen as rcss
import opencsp.common.lib.render_control.RenderControlSofastSetup as rcssp
import opencsp.common.lib.render_control.RenderControlSofastCamera as rcsc
import opencsp.common.lib.render_control.RenderControlSofastMirror as rcsm
import opencsp.common.lib.render_control.RenderControlText as rctxt
import opencsp.common.lib.tool.log_tools as lt


class SofastConfiguration:
    """Class for analyzing the configuration of a Sofast setup/measurement."""

    def __init__(self) -> "SofastConfiguration":
        self.data_sofast_object: ProcessSofastFringe | ProcessSofastFixed = None
        self._is_fringe = None
        self._is_fixed = None

    def load_sofast_object(self, process_sofast: ProcessSofastFringe | ProcessSofastFixed):
        """Loads ProcessSofast object (Fixed or Fringe) for further analysis

        Parameters
        ----------
        process_sofast: ProcessSofastFringe | ProcessSofastFixed
            ProcessSofast object
        """
        self.data_sofast_object = process_sofast
        if isinstance(self.data_sofast_object, ProcessSofastFringe):
            self._is_fringe = True
            self._is_fixed = False
        elif isinstance(self.data_sofast_object, ProcessSofastFixed):
            self._is_fringe = False
            self._is_fixed = True

    def get_measurement_stats(self) -> list[dict]:
        """Returns measurement statistics dictionary for each facet in Sofast calculation

        Returns
        -------
        list of dictionaries for each facet in the ProcessSofast object with following fields:
        - delta_x_sample_points_average
        - delta_y_sample_average
        - number_samples
        - focal_lengths_parabolic_xy
        """
        self._check_sofast_object_loaded()
        num_facets = self.data_sofast_object.num_facets

        stats = []
        for idx_facet in range(num_facets):
            if self._is_fringe:
                # Get surface data
                data_calc = self.data_sofast_object.data_calculation_facet[idx_facet]
                data_im_proc = self.data_sofast_object.data_image_processing_facet[idx_facet]
                data_surf = self.data_sofast_object.data_surfaces[idx_facet]

                # Assemble surface points in 2d arrays
                mask = data_im_proc.mask_processed
                im_x = np.zeros(mask.shape) * np.nan
                im_y = np.zeros(mask.shape) * np.nan
                im_x[mask] = data_calc.v_surf_points_facet.x
                im_y[mask] = data_calc.v_surf_points_facet.y

                # Number of points
                num_samps = len(data_calc.v_surf_points_facet)
            else:
                # Get surface data
                data_surf = self.data_sofast_object.slope_solvers[idx_facet].surface
                data_calc = self.data_sofast_object.data_calculation_facet[idx_facet]

                # Assemble surface points in 2d arrays
                surf_points = self.data_sofast_object.data_calculation_facet[idx_facet].v_surf_points_facet
                mask = self.data_sofast_object.data_calculation_blob_assignment[idx_facet].active_point_mask
                im_x = mask.astype(float) * np.nan
                im_y = mask.astype(float) * np.nan
                im_x[mask] = surf_points.x
                im_y[mask] = surf_points.y

                # Number of points
                num_samps = len(surf_points)

            # Calculate average sample resolution
            dx = np.diff(im_x, axis=1)  # meters
            dy = np.diff(im_y, axis=0)  # meters
            dx_avg = abs(np.nanmean(dx))  # meters
            dy_avg = abs(np.nanmean(dy))  # meters

            # Parabolic focal length in x and y
            if isinstance(data_surf, Surface2DParabolic):
                surf_coefs = data_calc.surf_coefs_facet
                focal_lengths_xy = [1 / 4 / surf_coefs[2], 1 / 4 / surf_coefs[5]]
            else:
                focal_lengths_xy = [np.nan, np.nan]

            stats.append(
                {
                    "delta_x_sample_points_average": dx_avg,
                    "delta_y_sample_points_average": dy_avg,
                    "number_samples": num_samps,
                    "focal_lengths_parabolic_xy": focal_lengths_xy,
                }
            )

        return stats

    def visualize_setup(
        self,
        view: View3d,
        title: str | None = "SOFAST Physical Setup\n(Screen Coordinates)",
        length_z_axis_cam: float = 8,
        axis_length: float = 2,
        min_axis_length_screen: float = 2,
        v_screen_object_screen: Vxyz = None,
        r_object_screen: Rotation = None,
    ) -> None:
        """Draws the given SOFAST setup components on a 3d axis.

        Parameters
        ----------
        ax : plt.Axes | None, optional
            3d matplotlib axes, if None, creates new axes, by default None
        length_z_axis_cam : float, optional
            Length of camera z axis to draw (m), by default 8
        axis_length : float, optional
            Length to draw individual coordinate system x, y, and z axes.
            This should be chosen to ensure legibility given the size of
            the SOFAST layout, which can vary widely.
            Default 0.1.
        min_axis_length_screen : float, optional
            Minimum length of axes to draw (m), by default 2
        v_screen_object_screen : Vxyz, optional
            Vector (m), screen to object in screen reference frame, by default None.
            If None, the object reference frame is not plotted.
        r_object_screen : Rotation, optional
            Rotation, object to screen reference frames, by default None.
            Only used if v_screen_object_screen is not None
        """
        # Call the stand-alone function.
        # Provide different display data depending on fringe or fixed.
        if self._is_fringe:
            display_local = self.data_sofast_object.display
            dot_locations_local = None
        elif self._is_fixed:
            display_local = None
            dot_locations_local = self.data_sofast_object.fixed_pattern_dot_dot_locs
        # Call
        draw_sofast_setup(
            view=view,
            sofast_is_fringe=self._is_fringe,
            sofast_is_fixed=self._is_fixed,
            camera=self.data_sofast_object.camera,
            display=display_local,
            dot_locations=dot_locations_local,
            mirror=mirror,
            facet_data=facet_data,
            orientation=self.data_sofast_object.orientation,
            sofast_setup_style=rcss.RenderControlSofastSetup(),
            axis_length=axis_length,
            v_screen_object_screen=v_screen_object_screen,
            r_object_screen=r_object_screen,
        )

    def _check_sofast_object_loaded(self) -> bool:
        if self.data_sofast_object is None:
            lt.error_and_raise(ValueError, "ProcessSofast object not loaded. Use self.load_sofast_object() first.")


# HELPER FUNCTIONS


def draw_sofast_setup(
    # Where to draw
    view: View3d,
    # Objects
    sofast_is_fringe: bool,
    sofast_is_fixed: bool,
    camera: Camera,
    display: Display,
    dot_locations: DotLocationsFixedPattern | None,
    mirror: MirrorParametric,
    facet_data: DefinitionFacet,
    # Extent
    world_box: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = ((-1, 1), (-1, 1), (0, 0.125)),
    # Locations
    world_transform: txyz.TransformXYZ | None = None,
    screen_transform: txyz.TransformXYZ | None = None,
    camera_transform: txyz.TransformXYZ | None = None,
    mirror_transform: txyz.TransformXYZ | None = None,
    # Render control
    sofast_setup_style: rcssp.RenderControlSofastSetup = rcssp.RenderControlSofastSetup(),
    z_axis_fov_distance: float = 0.6,  # m
    mirror_needle_length=0.1,  # m
    axis_length: float = 0.1,  # m
) -> None:
    """
    Draws the given SOFAST setup components on a 3d axis.

    This version is not a member of the SofastConfiguration class, which means that it
    can be called at any time, including from within a SOFAST member function in the
    midst of primary SOFAST setup analysis.

    (The SofastConfiguration class contains a SOFAST object as a data member, and is
    designed to mostly operate after the SOFAST setup is complete.)

    Parameters
    ----------
    view: View3d,
        View to to draw on.
    sofast_is_fringe: bool
        True if this is a SOFAST fringe setup.  Displays a screen rectangle.
    sofast_is_fixed: bool
        True if this is a SOFAST fixed setup.  Displays a screen dot pattern.
    camera: Camera
        SOFAST camera model.
    display: Display
        SOFAST display model, modeling relationship between screen
        coordinates and 3-d coordinates.
    dot_locations: DotLocationsFixedPattern | None
        For SOFAST Fixed setups, this is the pattern of dots on the screen.
    mirror: MirrorParametric
        Model of nominal mirror to be measured in the setup.
    facet_data: DefinitionFacet
        Additional mirror information, including centroid, normal at centroid, and vertex list.
    world_box : tuple[tuple[float, float], tuple[float, float], tuple[float, float]], optional
        Box boundary to draw.
        Form:    ((x_min, x_max), (y_min, y_max), (z_min, z_max))
        Default: ((-1,1), (-1,1), (0,2))
    world_transform: txyz.TransformXYZ | None
    screen_transform: txyz.TransformXYZ | None
    camera_transform: txyz.TransformXYZ | None
    mirror_transform: txyz.TransformXYZ | None
        Transforms describing the position of the world, screen, camera, and
        mirror relative to the base coordinate system.
    sofast_setup_style: RenderControlSofastSetup, optional
        Rendering control parameters for the SOFAST setup and its components.
    z_axis_fov_distance : float
        Distance from camera front nodal point (origin) to draw the
        field of view (FOV) boundary.  A good choice is to set this
        to the distance from the camera to the observed mirror.
    mirror_needle_length, float
        Length to draw surface normal needles at the mirror centroid and vertices.
    axis_length : float, optional
        Length to draw individual coordinate system x, y, and z axes.
        This should be chosen to ensure legibility given the size of
        the SOFAST layout, which can vary widely.
        Default 0.1.
    """

    # Draw world.
    if sofast_setup_style.draw_world:
        transformed_world_origin = draw_world(
            view,
            world_box=world_box,
            transform=world_transform,
            sofast_world_style=sofast_setup_style.sofast_world_style,
            axis_length=axis_length,
        )
    else:
        transformed_world_origin = None

    # Draw screen.
    if sofast_setup_style.draw_screen:
        transformed_screen_origin = draw_screen(
            view,
            sofast_is_fringe,
            sofast_is_fixed,
            display,
            dot_locations,
            transform=screen_transform,
            sofast_screen_style=sofast_setup_style.sofast_screen_style,
            axis_length=axis_length,
        )
    else:
        transformed_screen_origin = None

    # Draw camera.
    if sofast_setup_style.draw_camera:
        transformed_camera_origin = draw_camera(
            view,
            camera,
            transform=camera_transform,
            sofast_camera_style=sofast_setup_style.sofast_camera_style,
            z_axis_fov_distance=z_axis_fov_distance,
            axis_length=axis_length,
        )
    else:
        transformed_camera_origin = None

    # Draw mirror.
    if sofast_setup_style.draw_mirror:
        transformed_mirror_origin = draw_mirror(
            view,
            mirror,
            facet_data,
            transform=mirror_transform,
            sofast_mirror_style=sofast_setup_style.sofast_mirror_style,
            needle_length=mirror_needle_length,
            axis_length=axis_length,
        )
    else:
        transformed_mirror_origin = None

    # Provide handle to a breakpoint to search the stack for this routine in the debugger.
    return transformed_world_origin, transformed_screen_origin, transformed_camera_origin, transformed_mirror_origin


def draw_csys(
    view: View3d,
    origin: Vxyz,
    transform: txyz.TransformXYZ = None,
    short_name: str = None,
    long_name: str = None,
    origin_color: str = 'red',
    draw_origin_label=True,
    draw_axis_labels=False,
    label_color: str = 'darkred',
    axis_length: float = 0.1,  # m
) -> Vxyz:
    """Draws a coordinate system axis at the specified origin."""
    # Prepare geometry.
    # Origin
    x = origin.x[0]  # m
    y = origin.y[0]  # m
    z = origin.z[0]  # m
    # Axis tip points.
    x_tip = Vxyz([x + axis_length, y, z])
    y_tip = Vxyz([x, y + axis_length, z])
    z_tip = Vxyz([x, y, z + axis_length])
    # Apply transform.
    if transform is None:
        transformed_origin = origin
        transformed_x_tip = x_tip
        transformed_y_tip = y_tip
        transformed_z_tip = z_tip
    else:
        transformed_origin = transform.apply(origin)
        transformed_x_tip = transform.apply(x_tip)
        transformed_y_tip = transform.apply(y_tip)
        transformed_z_tip = transform.apply(z_tip)

    # SolidWorks axis colors are red(x), green(y), and blue(z)
    color_x = 'red'
    color_y = 'green'
    color_z = 'blue'

    # Style for drawing the origin and axis labels.
    label_style = rctxt.RenderControlText(
        fontsize='small', color=label_color, horizontalalignment='right', verticalalignment='top'
    )

    # Draw
    # X
    x_axis = transformed_origin.concatenate(transformed_x_tip)
    x_axis.draw_line(view, style=rcps.outline(color=color_x))
    if draw_axis_labels:
        view.draw_xyz_text(x_tip.as_xyz_list()[0], (short_name + 'x'), style=label_style)
    # Y
    y_axis = transformed_origin.concatenate(transformed_y_tip)
    y_axis.draw_line(view, style=rcps.outline(color=color_y))
    if draw_axis_labels:
        view.draw_xyz_text(y_tip.as_xyz_list()[0], (short_name + 'y'), style=label_style)
    # Z
    z_axis = transformed_origin.concatenate(transformed_z_tip)
    z_axis.draw_line(view, style=rcps.outline(color=color_z))
    if draw_axis_labels:
        view.draw_xyz_text(z_tip.as_xyz_list()[0], (short_name + 'z'), style=label_style)
    # Origin
    # Draw origin last, so it is over the axis lines.
    if (short_name is not None) and (long_name is not None):
        origin_legend = short_name + ' = ' + long_name
        origin_labels = [origin_legend]
    elif short_name is not None:
        origin_legend = short_name
        origin_labels = [origin_legend]
    elif long_name is not None:
        origin_legend = long_name
        origin_labels = [origin_legend]
    else:
        origin_labels = None
    transformed_origin.draw_points(view, style=rcps.marker(color=origin_color, markersize=4), labels=origin_labels)
    if draw_origin_label and (short_name is not None):
        view.draw_xyz_text(transformed_origin.as_xyz_list()[0], short_name, style=label_style)

    # Return.
    return transformed_origin


def draw_annotated_vector_from_origin(
    view: View3d,
    vector: Vxyz,
    transform: txyz.TransformXYZ = None,
    short_name: str = None,
    long_name: str = None,
    line_style=rcps.outline(),
    draw_base=True,
    base_style=rcps.marker(),
    draw_label=True,
    label_frac=0.3,
    label_style=rctxt.RenderControlText(
        fontsize='small', color='k', horizontalalignment='center', verticalalignment='center'
    ),
) -> Vxyz:
    """
    Draws a vector (origin --> head) in 3-d space, with annotations.

    We include this special version without a tail, to clarify code
    rendering the tvec vectors thata re the translation components
    of a transform relating two coordinate systems.  Those vectors
    are specified simply as vectors within the source coortdinate
    system, and are generally understood to be from the origin of
    the source coordinate system.

    Parameters
    ----------
    view : View3d
        View to draw in.
    tail : Vxyz
        Coordinates of vector tail.
    vector : Vxyz
        Coordinates of vector head.  It is a representation of a direction,
        or a translation between coordinate systems, in which case it is
        assumed to emanate from the origin.
    transform : txyz.TransformXYZ, optional
        Transform to apply before drawing.  Default None.
    short_name : str, optional
        Very short string to use as vector label.  For example, if the vector
        points from the camera coordinate origin to the mirror origin, a good
        choice might be 'C->M'.  Default None.
    long_name : str, optional
        Longer string to include in the plot legend, for example 'Camera to Mirror'.
        If both a short name and long_name are provided, then the legend will
        contain the translation:  'C->M: Camera to Mirror'.
        Default None.
    line_style : RenderControlPointSeq, optional
        Style to draw vector line.  Default rcps.outline().
    draw_base : bool, optional
        Whether to draw a symbol at the vector base.  Default True.
    base_style : RenderControlPointSeq, optional
        Style to draw vector base.  Default rcps.marker().
    draw_label : bool, optional
        Whether to draw string with the vector short name on the vector line.  Default True.
    label_frac : float, optional
        Fractional distance from the tail to head to draw the vector label.  Default 0.3.
    label_style : RenderControlText, optional
        Style to draw vector label.  Default is black.

    Returns
    -------
    Vxyz
        The (origin --> head) vector, after applying the input transform
        to both origin and head.
    """
    return draw_annotated_vector(
        view=view,
        tail=Vxyz([0, 0, 0]),
        head=vector,
        transform=transform,
        short_name=short_name,
        long_name=long_name,
        line_style=line_style,
        draw_base=draw_base,
        base_style=base_style,
        draw_label=draw_label,
        label_frac=label_frac,
        label_style=label_style,
    )


def draw_annotated_vector(
    view: View3d,
    tail: Vxyz,
    head: Vxyz,
    transform: txyz.TransformXYZ = None,
    short_name: str = None,
    long_name: str = None,
    line_style=rcps.outline(),
    draw_base=True,
    base_style=rcps.marker(),
    draw_label=True,
    label_frac=0.3,
    label_style=rctxt.RenderControlText(
        fontsize='small', color='k', horizontalalignment='center', verticalalignment='center'
    ),
) -> Vxyz:
    """
    Draws a vector (tail --> head) in 3-d space, with annotations.

    Parameters
    ----------
    view : View3d
        View to draw in.
    tail : Vxyz
        Coordinates of vector tail.
    head : Vxyz
        Coordinates of vector head.
    transform : txyz.TransformXYZ, optional
        Transform to apply before drawing.  Default None.
    short_name : str, optional
        Very short string to use as vector label.  For example, if the vector
        points from the camera coordinate origin to the mirror origin, a good
        choice might be 'C->M'.  Default None.
    long_name : str, optional
        Longer string to include in the plot legend, for example 'Camera to Mirror'.
        If both a short name and long_name are provided, then the legend will
        contain the translation:  'C->M: Camera to Mirror'.
        Default None.
    line_style : RenderControlPointSeq, optional
        Style to draw vector line.  Default rcps.outline().
    draw_base : bool, optional
        Whether to draw a symbol at the vector base.  Default True.
    base_style : RenderControlPointSeq, optional
        Style to draw vector base.  Default rcps.marker().
    draw_label : bool, optional
        Whether to draw string with the vector short name on the vector line.  Default True.
    label_frac : float, optional
        Fractional distance from the tail to head to draw the vector label.  Default 0.3.
    label_style : RenderControlText, optional
        Style to draw vector label.  Default is black.

    Returns
    -------
    Vxyz
        The (tail --> head) vector, after applying the input transform.
    """
    # Apply transform.
    if transform is None:
        transformed_tail = tail
        transformed_head = head
    else:
        transformed_tail = transform.apply(tail)
        transformed_head = transform.apply(head)

    # Construct vector.
    transformed_vec = transformed_tail.concatenate(transformed_head)

    # Draw.
    # Line
    if (short_name is not None) and (long_name is not None):
        legend_str = short_name + ': ' + long_name
    elif short_name is not None:
        legend_str = short_name
    elif long_name is not None:
        legend_str = long_name
    else:
        legend_str = None
    transformed_vec.draw_line(view, close=False, style=line_style, label=legend_str)
    # Base
    if draw_base:
        transformed_tail.draw_points(view, style=base_style)
    # Label
    if draw_label and (short_name is not None):
        transformed_vec_xyz_list = transformed_vec.as_xyz_list()
        transformed_tail_tuple = transformed_vec_xyz_list[0]
        tail_x = transformed_tail_tuple[0]
        tail_y = transformed_tail_tuple[1]
        tail_z = transformed_tail_tuple[2]
        transformed_head_tuple = transformed_vec_xyz_list[1]
        head_x = transformed_head_tuple[0]
        head_y = transformed_head_tuple[1]
        head_z = transformed_head_tuple[2]
        label_x = tail_x + (label_frac * (head_x - tail_x))
        label_y = tail_y + (label_frac * (head_y - tail_y))
        label_z = tail_z + (label_frac * (head_z - tail_z))
        label_tuple = (label_x, label_y, label_z)
        view.draw_xyz_text(label_tuple, short_name, style=label_style)

    # Return.
    return transformed_vec


def annotated_vector_line_style(color: str = 'blue', linewidth=1) -> rcps.RenderControlPointSeq:
    """Line style for annotated vectors."""
    return rcps.outline(color=color, linewidth=linewidth)


def annotated_vector_base_style(marker: str = 'x', color: str = 'blue') -> rcps.RenderControlPointSeq:
    """Annotated vector style for marker at tail."""
    return rcps.marker(marker=marker, color=color, markersize=5)


def annotated_vector_label_style(color: str = 'blue') -> rctxt.RenderControlText:
    """Label style for annotated vectors."""
    return rctxt.RenderControlText(
        fontsize='small', color=color, horizontalalignment='center', verticalalignment='center'
    )


def draw_world(
    view: View3d,
    world_box: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = ((-1, 1), (-1, 1), (0, 2.0)),
    transform: txyz.TransformXYZ = None,
    short_name: str = 'W',
    long_name: str = 'World',
    sofast_world_style: rcsw.RenderControlSofastWorld = rcsw.RenderControlSofastWorld(),
    axis_length: float = 0.1,  # m
) -> Vxyz:
    """
    Draws a box defining the conceptual world boundaries, and also the
    world coordinate system origin.

    The boundaries are not enforced by graphics clipping or other checks;
    they are merely for visual reference and orientation.

    Parameters
    ----------
    view : View3d
        View to draw in.
    world_box : tuple[tuple[float, float], tuple[float, float], tuple[float, float]], optional
        Box boundary to draw.
        Form:    ((x_min, x_max), (y_min, y_max), (z_min, z_max))
        Default: ((-1,1), (-1,1), (0,2))
    transform : txyz.TransformXYZ, optional
        Transform to apply before drawing.  Default None.
    short_name : str, optional
        Very short string to use as an origin label.  Default 'W'.
    long_name : str, optional
        Longer string to include in the plot legend.  Default 'World'.
        If both short_name and long_name are provided, then the legend
        will contain the translation:  'W = World'.
        Default None.
    sofast_world_style: RenderControlSofastWorld, optional
        Rendering control parameters.
    axis_length : float, optional
        Length to draw individual coordinate system x, y, and z axes.
        This should be chosen to ensure legibility given the size of
        the SOFAST layout, which can vary widely.
        Default 0.1.

    Returns
    -------
    Vxyz
        The world origin, after applying the input transform.
    """
    # Draw world corners, to stabilize xy, xz, and yz plot limits. While maintining axis equal scale.
    # Range limits for each axis.
    world_x_limits = world_box[0]
    world_y_limits = world_box[1]
    world_z_limits = world_box[2]
    world_x_min = world_x_limits[0]
    world_x_max = world_x_limits[1]
    world_y_min = world_y_limits[0]
    world_y_max = world_y_limits[1]
    world_z_min = world_z_limits[0]
    world_z_max = world_z_limits[1]
    # Corner points.
    w_nnn = Vxyz([world_x_min, world_y_min, world_z_min])
    w_xnn = Vxyz([world_x_max, world_y_min, world_z_min])
    w_xxn = Vxyz([world_x_max, world_y_max, world_z_min])
    w_nxn = Vxyz([world_x_min, world_y_max, world_z_min])
    w_nnx = Vxyz([world_x_min, world_y_min, world_z_max])
    w_xnx = Vxyz([world_x_max, world_y_min, world_z_max])
    w_xxx = Vxyz([world_x_max, world_y_max, world_z_max])
    w_nxx = Vxyz([world_x_min, world_y_max, world_z_max])

    # Ensure that a transform is available.
    if transform is None:
        transform = txyz.identity_transform()

    # Corner points.
    w_nnn = Vxyz([world_x_min, world_y_min, world_z_min])
    w_xnn = Vxyz([world_x_max, world_y_min, world_z_min])
    w_xxn = Vxyz([world_x_max, world_y_max, world_z_min])
    w_nxn = Vxyz([world_x_min, world_y_max, world_z_min])
    w_nnx = Vxyz([world_x_min, world_y_min, world_z_max])
    w_xnx = Vxyz([world_x_max, world_y_min, world_z_max])
    w_xxx = Vxyz([world_x_max, world_y_max, world_z_max])
    w_nxx = Vxyz([world_x_min, world_y_max, world_z_max])

    # Transformed corner points.
    transformed_w_nnn = transform.apply(w_nnn)
    transformed_w_xnn = transform.apply(w_xnn)
    transformed_w_xxn = transform.apply(w_xxn)
    transformed_w_nxn = transform.apply(w_nxn)
    transformed_w_nnx = transform.apply(w_nnx)
    transformed_w_xnx = transform.apply(w_xnx)
    transformed_w_xxx = transform.apply(w_xxx)
    transformed_w_nxx = transform.apply(w_nxx)

    # Assemble list of corners for plotting.
    transformed_corner_list = [
        transformed_w_nnn,
        transformed_w_xnn,
        transformed_w_xxn,
        transformed_w_nxn,
        transformed_w_nnx,
        transformed_w_xnx,
        transformed_w_xxx,
        transformed_w_nxx,
    ]

    # Draw corners.
    if sofast_world_style.draw_corners:
        for transformed_corner in transformed_corner_list:
            transformed_corner.draw_points(view, style=rcps.marker(color=sofast_world_style.color))

    # Draw edges.
    if sofast_world_style.draw_edges:
        # Bottom
        transformed_w_nnn.concatenate(transformed_w_xnn).draw_line(
            view, style=rcps.outline(color=sofast_world_style.color)
        )
        transformed_w_nxn.concatenate(transformed_w_xxn).draw_line(
            view, style=rcps.outline(color=sofast_world_style.color)
        )
        transformed_w_nnn.concatenate(transformed_w_nxn).draw_line(
            view, style=rcps.outline(color=sofast_world_style.color)
        )
        transformed_w_xnn.concatenate(transformed_w_xxn).draw_line(
            view, style=rcps.outline(color=sofast_world_style.color)
        )
        # Top
        transformed_w_nnx.concatenate(transformed_w_xnx).draw_line(
            view, style=rcps.outline(color=sofast_world_style.color)
        )
        transformed_w_nxx.concatenate(transformed_w_xxx).draw_line(
            view, style=rcps.outline(color=sofast_world_style.color)
        )
        transformed_w_nnx.concatenate(transformed_w_nxx).draw_line(
            view, style=rcps.outline(color=sofast_world_style.color)
        )
        transformed_w_xnx.concatenate(transformed_w_xxx).draw_line(
            view, style=rcps.outline(color=sofast_world_style.color)
        )
        # Vertical
        transformed_w_nnn.concatenate(transformed_w_nnx).draw_line(
            view, style=rcps.outline(color=sofast_world_style.color)
        )
        transformed_w_nxn.concatenate(transformed_w_nxx).draw_line(
            view, style=rcps.outline(color=sofast_world_style.color)
        )
        transformed_w_xnn.concatenate(transformed_w_xnx).draw_line(
            view, style=rcps.outline(color=sofast_world_style.color)
        )
        transformed_w_xxn.concatenate(transformed_w_xxx).draw_line(
            view, style=rcps.outline(color=sofast_world_style.color)
        )

    # Draw coordinate system.
    transformed_origin = draw_csys(
        view,
        origin=Vxyz([0, 0, 0]),
        transform=transform,
        short_name=short_name,
        long_name=long_name,
        origin_color=sofast_world_style.color,
        draw_origin_label=True,
        draw_axis_labels=False,
        label_color=sofast_world_style.color,
        axis_length=axis_length,
    )

    # Return.
    return transformed_origin


def draw_screen(
    view: View3d,
    sofast_is_fringe: bool,
    sofast_is_fixed: bool,
    display: Display,
    dot_locations: DotLocationsFixedPattern | None,
    transform: txyz.TransformXYZ = None,
    short_name: str = 'S',
    long_name: str = 'Screen',
    sofast_screen_style: rcss.RenderControlSofastScreen = rcss.RenderControlSofastScreen(),
    axis_length: float = 0.1,  # m
) -> Vxyz:
    """
    Draws the screen coordinate system and border in the specified pose.

    Parameters
    ----------
    view : View3d
        View to draw in.
    sofast_is_fringe : bool
        Whether this SOFAST configuration supports fringe measurement.
    sofast_is_fixed : bool
        Whether this SOFAST configuration supports fixed measurement.
    display : Display
        Data structure representing the extent of the SOFAST optical
        target, which we refer to as a screen.
    dot_locations : DotLocationsFixedPattern | None
        Data structure representing the pattern of dots on the screen
        for SOFAST Fixed measurement.
    transform : txyz.TransformXYZ, optional
        Transform to apply before drawing.  Default None.
    short_name : str, optional
        Very short string to use as an origin label.  Default 'S'.
    long_name : str, optional
        Longer string to include in the plot legend.  Default 'Screen'.
        If both short_name and long_name are provided, then the legend
        will contain the translation:  'S = Screen'.
        Default None.
    sofast_screen_style: RenderControlSofastScreen
        Rendering control parameters.
    axis_length : float, optional
        Length to draw individual coordinate system x, y, and z axes.
        This should be chosen to ensure legibility given the size of
        the SOFAST layout, which can vary widely.
        Default 0.1.

    Returns
    -------
    Vxyz
        The screen origin, after applying the input transform.
    """
    # Calculate screen features and center
    if sofast_is_fringe:
        frac = 0.95  # &&&& DELETE-SCAFFOLDING -- WHY THIS VALUE?  MAYBE 1.0 INSTEAD?
        screen_outline = display.interp_func(Vxy(([0, frac, frac, 0, 0], [0, 0, frac, frac, 0])))
        screen_dots = None
        screen_center = display.interp_func(Vxy((0.5, 0.5)))
    elif sofast_is_fixed:
        lt.error_and_raise(
            ValueError,
            'In draw_screen(), rendering dots and dot bounding box not developed yet, due to lack of test data.',
        )
        dot_loc_array = dot_locations.xyz_dot_loc
        screen_outline = Vxyz((dot_loc_array[..., 0], dot_loc_array[..., 1], dot_loc_array[..., 2]))
        screen_dots = Vxyz((dot_loc_array[..., 0], dot_loc_array[..., 1], dot_loc_array[..., 2]))
        screen_center = dot_locations.xy_indices_to_screen_coordinates(Vxy([0, 0], dtype=int))

    # Ensure that a transform is available.
    if transform is None:
        transform = txyz.identity_transform()

    # Example colors to use if a contrasting label is desired.
    #     color = (0, 1.0, 0),  # Green
    #     label_color = (0, 0.8, 0),  # Dark green, for contrast with origin dot
    label_color = sofast_screen_style.color

    # Transform the screen dots, outline, and center point.
    transformed_screen_outline = transform.apply(screen_outline)
    transformed_screen_center = transform.apply(screen_center)

    # Draw screen dots.
    if sofast_is_fixed and (screen_dots is not None):
        transformed_screen_dots = transform.apply(screen_dots)
        transformed_screen_dots.draw_points(
            view, style=rcps.marker(color=sofast_screen_style.color), label="Screen Points"
        )

    # Draw screen outline.
    # ax.plot(*screen_outline.data, color="red", label="Screen Outline")
    transformed_screen_outline.draw_line(
        view, style=rcps.outline(color=sofast_screen_style.color), label="Screen Outline"
    )

    # Draw center point.
    center_style = rcps.marker(marker="+", color=sofast_screen_style.color, markersize=6)
    transformed_screen_center.draw_points(view, style=center_style, labels=['Screen Center'])

    # Draw coordinate system.
    transformed_origin = draw_csys(
        view,
        origin=Vxyz([0, 0, 0]),
        transform=transform,
        short_name=short_name,
        long_name=long_name,
        origin_color=sofast_screen_style.color,
        draw_origin_label=True,
        draw_axis_labels=False,
        label_color=label_color,
        axis_length=axis_length,
    )

    # Return.
    return transformed_origin


def draw_camera(
    view: View3d,
    camera: Camera,
    transform: txyz.TransformXYZ = None,
    short_name: str = 'C',
    long_name: str = 'Camera',
    sofast_camera_style: rcsc.RenderControlSofastCamera = rcsc.RenderControlSofastCamera(),
    z_axis_fov_distance: float = 0.6,  # m
    axis_length: float = 0.1,  # m
) -> Vxyz:
    """
    Draws the camera coordinate system and field of view in the specified pose.

    Parameters
    ----------
    view : View3d
        View to draw in.
    camera : Camera
        Camera model, including image size, focal length, distortion, etc.
    transform : txyz.TransformXYZ, optional
        Transform to apply before drawing.  Default None.
    short_name : str, optional
        Very short string to use as an origin label.  Default 'C'.
    long_name : str, optional
        Longer string to include in the plot legend.  Default 'Camera'.
        If both short_name and long_name are provided, then the legend
        will contain the translation:  'C = Camera'.
        Default None.
    sofast_camera_style: RenderControlSofastCamera
        Rendering control parameters.
    z_axis_fov_distance : float
        Distance from camera front nodal point (origin) to draw the
        field of view (FOV) boundary.  A good choice is to set this
        to the distance from the camera to the observed mirror.
    axis_length : float, optional
        Length to draw individual coordinate system x, y, and z axes.
        This should be chosen to ensure legibility given the size of
        the SOFAST layout, which can vary widely.
        Default 0.1.

    Returns
    -------
    Vxyz
        The camera origin, after applying the input transform.
    """
    # Calculate camera field of view (FOV), in camera coordinates.
    x = camera.image_shape_xy[0]
    y = camera.image_shape_xy[1]
    if sofast_camera_style.show_fov_lens_distortion:
        # Include points along the straight image edges, so the effect of
        # lens distortion will be evident in the field of view.
        # Note that distortion effects are inverted.  For example, barrel
        # distortion causes objects in the image cornes to appear nearer
        # the center than actual, leading to a square appearing as a barrel.
        # But this means that the field of view at the corners is extended
        # further than linear.  Thus for a lens with barrel distortion,
        # the FOV looks like a pincushion.  And vice versa for pincushion
        # distortion.
        fov_directions_uxyz = camera.vector_from_pixel(
            Vxy(
                (
                    [0, 0, 0, 0, 0, 0, x / 4, x / 2, (3 * x) / 4, x, x, x, x, x, (3 * x / 4), x / 2, x / 4],
                    [0, y / 4, y / 2, y, (3 * y / 4), y, y, y, y, y, (3 * y / 4), y / 2, y / 4, 0, 0, 0, 0],
                )
            )
        )
    else:
        # Just draw the four corners.
        fov_directions_uxyz = camera.vector_from_pixel(Vxy(([0, 0, x, x], [0, y, y, 0])))
    # We began with a regular rectangle in pixel space, but
    # the resulting 3-d FOV corner direction vectors are not
    # necessarily symmetrical, because of the presence of lens
    # distortion in the camera model.  Therefore we individually
    # project them onto the constant-z FOV plane.
    n_fov_corners = fov_directions_uxyz.len()
    corner_xyz_list = []
    for idx in range(n_fov_corners):
        corner_direction_xyz_array = fov_directions_uxyz.data[:, idx]
        corner_direction_x = corner_direction_xyz_array[0]
        corner_direction_y = corner_direction_xyz_array[1]
        corner_direction_z = corner_direction_xyz_array[2]
        scale = z_axis_fov_distance / corner_direction_z
        corner_xyz = ((corner_direction_x * scale), (corner_direction_y * scale), (corner_direction_z * scale))
        corner_xyz_list.append(corner_xyz)
    fov = Vxyz.from_list(corner_xyz_list)

    # Ensure that a transform is available.
    if transform is None:
        transform = txyz.identity_transform()

    # Example colors to use if a contrasting label is desired.
    #     color = (0, 1.0, 1.0),  # Cyan
    #     label_color = (0, 0.8, 0.8),  # Dark cyan, for contrast with origin dot
    label_color = sofast_camera_style.color

    # Draw coordinate system.
    transformed_origin = draw_csys(
        view,
        origin=Vxyz([0, 0, 0]),
        transform=transform,
        short_name=short_name,
        long_name=long_name,
        origin_color=sofast_camera_style.color,
        draw_origin_label=True,
        draw_axis_labels=False,
        label_color=label_color,
        axis_length=axis_length,
    )

    # Transform the FOV polygon.
    transformed_fov = transform.apply(fov)

    # Draw camera FOV polygon.
    transformed_fov.draw_line(view, close=True, style=rcps.data_curve(color=sofast_camera_style.color))

    # Add lines connecting camera origin and camera FOV corners.
    for idx in range(n_fov_corners):
        transformed_v_cam_to_fov = transformed_origin.concatenate(Vxyz(transformed_fov.data[:, idx]))
        transformed_v_cam_to_fov.draw_line(view, close=False, style=rcps.outline(color=sofast_camera_style.color))

    # Return.
    return transformed_origin


def draw_mirror(
    view: View3d,
    mirror: MirrorParametric,
    facet_data: DefinitionFacet,
    transform: txyz.TransformXYZ = None,
    short_name: str = 'M',
    long_name: str = 'Mirror',
    sofast_mirror_style: rcsm.RenderControlSofastMirror = rcsm.RenderControlSofastMirror(),
    needle_length: float = 0.1,
    axis_length: float = 0.1,  # m
) -> Vxyz:
    """
    Draws the mirror to be measured, with annotations.

    Parameters
    ----------
    view : View3d
        View to draw in.
    mirror : MirrorParametric
        The mirror to draw.  A parametric mirror enables drawing the
        mirror embedded surface, lifted edges, etc.
    facet_data : DefinitionFacet
        Definition of the facet boundary and surface normals.
    transform : txyz.TransformXYZ, optional
        Transform to apply before drawing.  Default None.
    short_name : str, optional
        Very short string to use as an origin label.  Default 'M'.
    long_name : str, optional
        Longer string to include in the plot legend.  Default 'Mirror'.
        If both short_name and long_name are provided, then the legend
        will contain the translation:  'M = Mirror'.
        Default None.
    sofast_mirror_style: RenderControlSofastMirror
        Rendering control parameters.
    needle_length : float, optional
        Length to draw surface normals at mirror centroid and vertices.
        Default 0.1.
    axis_length : float, optional
        Length to draw individual coordinate system x, y, and z axes.
        This should be chosen to ensure legibility given the size of
        the SOFAST layout, which can vary widely.
        Default 0.1.

    Returns
    -------
    Vxyz
        The mirror origin, after applying the input transform.
    """
    # Ensure that a transform is available.
    if transform is None:
        transform = txyz.identity_transform()

    # Example colors to use if a contrasting label is desired.
    #     color = (1, 0, 1)  # Magenta
    #     label_color = (0.8, 0, 0.8)  # Dark magenta, for contrast with origin dot
    label_color = sofast_mirror_style.color

    # Draw the mirror and its embedding surface.
    mirror_style = rcm.RenderControlMirror()
    ems.draw_mirror_and_embedding_mirror(
        mirror,
        view=view,
        mirror_style=mirror_style,
        draw_embedding=sofast_mirror_style.draw_embedding_mirror,
        draw_projection=True,  # See above.
        projected_style=sofast_mirror_style.mirror_projected_style,
        embedding_style=sofast_mirror_style.embedding_mirror_style,
        transform=transform,
    )

    # Draw the mirror centroid defined by the facet definition file.
    # Apply transform
    transformed_v_facet_centroid = transform.apply(facet_data.v_facet_centroid)
    # Draw centroid
    transformed_v_facet_centroid.draw_points(view, style=rcps.marker(color=sofast_mirror_style.color))

    # Draw the surface normal at the mirror centroid, also defined in the facet definition file.
    # Rotate the surface normal vector, without translation.
    rotated_u_facet_centroid_normal = facet_data.u_facet_centroid_normal.rotate(transform.R)
    # Draw surface normal at facet centroid
    needle_base = transformed_v_facet_centroid
    needle_tip = needle_base + (rotated_u_facet_centroid_normal.as_Vxyz() * needle_length)
    needle = Vxyz.from_list((needle_base, needle_tip))
    needle.draw_line(view, style=rcps.outline(color=sofast_mirror_style.color))

    # Draw coordinate system.
    transformed_origin = draw_csys(
        view,
        origin=Vxyz([0, 0, 0]),
        transform=transform,
        short_name=short_name,
        long_name=long_name,
        origin_color=sofast_mirror_style.color,
        draw_origin_label=True,
        draw_axis_labels=False,
        label_color=label_color,
        axis_length=axis_length,
    )

    # Return.
    return transformed_origin


# def draw_sofast_setup():
