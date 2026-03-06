"""Library of functions used to process the geometry of a deflectometry setup."""

from scipy.spatial.transform import Rotation

from opencsp.common.lib.camera.Camera import Camera
from opencsp.app.sofast.lib.DebugOpticsGeometry import DebugOpticsGeometry
from opencsp.app.sofast.lib.DefinitionFacet import DefinitionFacet
import opencsp.app.sofast.lib.sofast_debug_figure_support as sdfs
from opencsp.app.sofast.lib.SpatialOrientation import SpatialOrientation
import opencsp.common.lib.geometry.TransformXYZ as txyz
from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.geometry.Vxyz import Vxyz
import opencsp.common.lib.render.view_spec as vs


# SOFAST SETUP, WITHOUT MIRROR


def figure_sofast_setup_without_mirror(
    figure_title: str,
    camera: Camera,
    facet_data: DefinitionFacet,
    orientation: SpatialOrientation,
    debug: DebugOpticsGeometry,
):
    """Sets up and draws a figure showing SOFAST layout, including only screen and camera."""
    view_spec_az_el_roll_list = [
        (vs.view_spec_3d(), None),
        (vs.view_spec_xy(), None),
        (vs.view_spec_xz(), None),
        (vs.view_spec_yz(), None),
    ]
    for view_spec_az_el_roll in view_spec_az_el_roll_list:
        view_spec = view_spec_az_el_roll[0]
        az_el_roll_deg = view_spec_az_el_roll[1]
        if az_el_roll_deg is None:
            this_title = figure_title
        else:
            this_title = figure_title + ' (Az,El,Roll)=' + str(az_el_roll_deg)
        figure_sofast_setup_without_mirror_aux(
            this_title, view_spec, camera, facet_data, orientation, debug, view_az_el_roll_deg=az_el_roll_deg
        )


def figure_sofast_setup_without_mirror_aux(
    figure_title: str,
    view_spec: dict,
    camera: Camera,
    facet_data: DefinitionFacet,
    orientation: SpatialOrientation,
    debug: DebugOpticsGeometry,
    view_az_el_roll_deg: float = None,
) -> None:
    """Supports routine without aux extension."""

    # Camera-to-screen transform.
    trans_cam_screen = orientation.trans_screen_cam.inv()

    # Draw.
    fig_rec, _, _, _ = sdfs.start_and_draw_sofast_setup_figure(
        figure_title=figure_title,
        view_spec=view_spec,
        camera=camera,
        facet_data=facet_data,
        trans_cam_screen=trans_cam_screen,
        trans_mirror_screen=None,  # Signal don't draw mirror.
        debug=debug,
        view_az_el_roll_deg=view_az_el_roll_deg,
        grid=debug.draw_sofast_setup_axis_grid,
        axis_prefix="World ",
    )

    # Finish and save the figure.
    sdfs.finish_debug_3d_figure(figure_title, 'geometry', fig_rec, debug)


# SOFAST SETUP, BEFORE MIRROR TRANSLATION OR ROTATION


def figure_setup_before_mirror_translation_fit(
    figure_title: str,
    camera: Camera,
    facet_data: DefinitionFacet,
    orientation: SpatialOrientation,
    debug: DebugOpticsGeometry,
):
    """
    Sets up and draws a figure showing SOFAST layout, including screen, camera,
    and mirror, before all facet position estimation operations.
    """
    view_spec_az_el_roll_list = [
        (vs.view_spec_3d(), None),
        (vs.view_spec_xy(), None),
        (vs.view_spec_xz(), None),
        (vs.view_spec_yz(), None),
    ]
    for view_spec_az_el_roll in view_spec_az_el_roll_list:
        view_spec = view_spec_az_el_roll[0]
        az_el_roll_deg = view_spec_az_el_roll[1]
        if az_el_roll_deg is None:
            this_title = figure_title
        else:
            this_title = figure_title + ' (Az,El,Roll)=' + str(az_el_roll_deg)
        figure_setup_before_mirror_translation_fit_aux(
            this_title, view_spec, camera, facet_data, orientation, debug, view_az_el_roll_deg=az_el_roll_deg
        )


def figure_setup_before_mirror_translation_fit_aux(
    figure_title: str,
    view_spec: dict,
    camera: Camera,
    facet_data: DefinitionFacet,
    orientation: SpatialOrientation,
    debug: DebugOpticsGeometry,
    view_az_el_roll_deg: float = None,
) -> None:
    """Supports routine without aux extension."""

    # Camera-to-screen transform.
    trans_cam_screen = orientation.trans_screen_cam.inv()
    # Mirror-to-screen transform.
    trans_mirror_screen = txyz.identity_transform()

    # Draw.
    fig_rec, trans_screen_world, trans_camera_world, trans_mirror_world = sdfs.start_and_draw_sofast_setup_figure(
        figure_title=figure_title,
        view_spec=view_spec,
        camera=camera,
        facet_data=facet_data,
        trans_cam_screen=trans_cam_screen,
        trans_mirror_screen=trans_mirror_screen,
        debug=debug,
        view_az_el_roll_deg=view_az_el_roll_deg,
        grid=debug.draw_sofast_setup_axis_grid,
        axis_prefix="World ",
    )

    # Finish and save the figure.
    sdfs.finish_debug_3d_figure(figure_title, 'geometry', fig_rec, debug)


# SOFAST SETUP, WITH MIRROR POSITION ESTIMATE CONSIDERING ONLY TRANSLATION


def figure_setup_mirror_translation_only_fit(
    figure_title: str,
    camera: Camera,
    facet_data: DefinitionFacet,
    orientation: SpatialOrientation,
    debug: DebugOpticsGeometry,
    v_facet_centroid: Vxyz,
    v_cam_optic_centroid_cam_exp: Vxyz,
):
    """
    Sets up and draws a figure showing SOFAST layout, including screen, camera,
    and mirror, with facet position estimated using only translation.
    """
    view_spec_az_el_roll_list = [
        (vs.view_spec_3d(), None),
        (vs.view_spec_xy(), None),
        (vs.view_spec_xz(), None),
        (vs.view_spec_yz(), None),
        (vs.view_spec_3d(), (20, 20, 0)),
    ]
    for view_spec_az_el_roll in view_spec_az_el_roll_list:
        view_spec = view_spec_az_el_roll[0]
        az_el_roll_deg = view_spec_az_el_roll[1]
        if az_el_roll_deg is None:
            this_title = figure_title
        else:
            this_title = figure_title + ' (Az,El,Roll)=' + str(az_el_roll_deg)
        figure_setup_mirror_translation_only_fit_aux(
            this_title,
            view_spec,
            camera,
            facet_data,
            orientation,
            debug,
            v_facet_centroid,
            v_cam_optic_centroid_cam_exp,
            view_az_el_roll_deg=az_el_roll_deg,
        )


def figure_setup_mirror_translation_only_fit_aux(
    figure_title: str,
    view_spec: dict,
    camera: Camera,
    facet_data: DefinitionFacet,
    orientation: SpatialOrientation,
    debug: DebugOpticsGeometry,
    v_facet_centroid: Vxyz,
    v_cam_optic_centroid_cam_exp: Vxyz,
    view_az_el_roll_deg: float = None,
) -> None:
    """Supports routine without aux extension."""

    # Camera-to-screen transform.
    trans_cam_screen = orientation.trans_screen_cam.inv()

    # Mirror pose.
    v_cam_optic_origin_cam_exp = v_cam_optic_centroid_cam_exp - v_facet_centroid
    v_cam_optic_origin_screen_exp = trans_cam_screen.apply(v_cam_optic_origin_cam_exp)
    trans_mirror_screen = txyz.TransformXYZ.from_R_V(R=Rotation.identity(), V=v_cam_optic_origin_screen_exp)

    # Draw.
    fig_rec, trans_screen_world, trans_camera_world, trans_mirror_world = sdfs.start_and_draw_sofast_setup_figure(
        figure_title=figure_title,
        view_spec=view_spec,
        camera=camera,
        facet_data=facet_data,
        trans_cam_screen=trans_cam_screen,
        trans_mirror_screen=trans_mirror_screen,
        debug=debug,
        view_az_el_roll_deg=view_az_el_roll_deg,
        grid=debug.draw_sofast_setup_axis_grid,
        axis_prefix="World ",
    )

    # Import here, to avoid circular import.
    import opencsp.app.sofast.lib.SofastConfiguration as sfcfg

    # Draw the vector from camera to mirror centroid.
    v_camera_to_centroid_color = 'red'
    sfcfg.draw_annotated_vector_from_origin(
        view=fig_rec.view,
        vector=v_cam_optic_centroid_cam_exp,
        transform=trans_camera_world,
        short_name='C->Mc',
        long_name='Camera to Mirror Centroid',
        line_style=sfcfg.annotated_vector_line_style(color=v_camera_to_centroid_color),
        draw_base=True,
        base_style=sfcfg.annotated_vector_base_style(color=v_camera_to_centroid_color),
        draw_label=True,
        label_style=sfcfg.annotated_vector_label_style(color=v_camera_to_centroid_color),
    )

    # Draw the vector from camera to mirror origin.
    v_camera_to_origin_color = 'blue'
    sfcfg.draw_annotated_vector_from_origin(
        view=fig_rec.view,
        vector=v_cam_optic_origin_cam_exp,
        transform=trans_camera_world,
        short_name='C->Mo',
        long_name='Camera to Mirror Origin',
        line_style=sfcfg.annotated_vector_line_style(color=v_camera_to_origin_color),
        draw_base=True,
        base_style=sfcfg.annotated_vector_base_style(color=v_camera_to_origin_color),
        draw_label=True,
        label_style=sfcfg.annotated_vector_label_style(color=v_camera_to_origin_color),
    )

    # Finish and save the figure.
    sdfs.finish_debug_3d_figure(figure_title, 'geometry', fig_rec, debug)


# SOFAST SETUP, WITH MIRROR POSITION ESTIMATE CONSIDERING TRANSLATION AND ROTATION,
# BUT NOT CENTROID SURFACE NORMAL


def figure_setup_mirror_fit_translation_rotation_but_not_normal(
    figure_title: str,
    camera: Camera,
    facet_data: DefinitionFacet,
    orientation: SpatialOrientation,
    debug: DebugOpticsGeometry,
    v_facet_centroid: Vxyz,
    v_cam_optic_centroid_cam_exp: Vxyz,
    r_cam_optic_exp_A: Rotation,
    u_reflection_norm_cam: Uxyz,
    draw_reflection: bool,
):
    """
    Sets up and draws a figure showing SOFAST layout, including screen, camera,
    and mirror, with facet position estimated using translation and rotation,
    but not the facet centroid surface normal.
    """
    view_spec_az_el_roll_list = [
        (vs.view_spec_3d(), None),
        (vs.view_spec_xy(), None),
        (vs.view_spec_xz(), None),
        (vs.view_spec_yz(), None),
        (vs.view_spec_3d(), (20, 20, 0)),
        (vs.view_spec_3d(), (-15, 135, 180)),
    ]
    for view_spec_az_el_roll in view_spec_az_el_roll_list:
        view_spec = view_spec_az_el_roll[0]
        az_el_roll_deg = view_spec_az_el_roll[1]
        if az_el_roll_deg is None:
            this_title = figure_title
        else:
            this_title = figure_title + ' (Az,El,Roll)=' + str(az_el_roll_deg)
        figure_setup_mirror_fit_translation_rotation_but_not_normal_aux(
            this_title,
            view_spec,
            camera,
            facet_data,
            orientation,
            debug,
            v_facet_centroid,
            v_cam_optic_centroid_cam_exp,
            r_cam_optic_exp_A,
            u_reflection_norm_cam,
            draw_reflection,
            view_az_el_roll_deg=az_el_roll_deg,
        )


def figure_setup_mirror_fit_translation_rotation_but_not_normal_aux(
    figure_title: str,
    view_spec: dict,
    camera: Camera,
    facet_data: DefinitionFacet,
    orientation: SpatialOrientation,
    debug: DebugOpticsGeometry,
    v_facet_centroid: Vxyz,
    v_cam_optic_centroid_cam_exp: Vxyz,
    r_cam_optic_exp_A: Rotation,
    u_reflection_norm_cam: Uxyz,
    draw_reflection: bool,
    view_az_el_roll_deg: float = None,
) -> None:
    """Supports routine without aux extension."""

    # Camera-to-screen transform.
    trans_cam_screen = orientation.trans_screen_cam.inv()
    # # Mirror pose.
    # # mirror_rotation = Rotation.from_euler('zx', [-30.0, 45.0], degrees=True)
    # # mirror_translation = Vxyz([0.25, -0.325, 0.1])
    # mirror_rotation = Rotation.from_euler('zxz', [180.0, -75.0, -20.0], degrees=True)
    # mirror_translation = Vxyz([-0.45, 0.0, 1.0])
    # trans_mirror_world = txyz.TransformXYZ.from_R_V(R=mirror_rotation, V=mirror_translation)
    # # Mirror-to-screen transform.
    # trans_mirror_screen = debug.trans_screen_world.inv() * trans_mirror_world

    # Mirror pose.
    # v_cam_optic_origin_cam_exp = v_cam_optic_centroid_cam_exp - v_facet_centroid
    # &&&& DELETE-SCAFFOLDING -- USED TO BE NOT .INV()
    v_cam_optic_origin_cam_exp = v_cam_optic_centroid_cam_exp - v_facet_centroid.rotate(r_cam_optic_exp_A)
    # v_cam_optic_origin_cam_exp = v_cam_optic_centroid_cam_exp  # &&&& DELETE-SCAFFOLDING -- USED TO INCLUDE OFFSET

    # v_cam_optic_origin_cam_exp = v_cam_optic_centroid_cam_exp - v_facet_centroid.rotate(r_cam_optic_exp_A.inv())
    v_cam_optic_origin_screen_exp = trans_cam_screen.apply(v_cam_optic_origin_cam_exp)
    trans_mirror_screen = txyz.TransformXYZ.from_R_V(R=r_cam_optic_exp_A, V=v_cam_optic_origin_screen_exp)
    # trans_mirror_screen = txyz.TransformXYZ.from_R_V(R=r_cam_optic_exp_A.inv(), V=v_cam_optic_origin_screen_exp)

    # # In other words, position of optic origin in 3-d space, in camera coordinates.
    # translation_2 = v_cam_optic_centroid_cam_exp - v_facet_centroid.rotate(r_cam_optic_exp_A)
    # # Computed 3-d position of facet corners in camera coordinates.
    # v_facet_corners_cam_2 = v_facet_corners.rotate(r_cam_optic_exp_A) + translation_2

    # Draw.
    fig_rec, trans_screen_world, trans_camera_world, trans_mirror_world = sdfs.start_and_draw_sofast_setup_figure(
        figure_title=figure_title,
        view_spec=view_spec,
        camera=camera,
        facet_data=facet_data,
        trans_cam_screen=trans_cam_screen,
        trans_mirror_screen=trans_mirror_screen,
        debug=debug,
        view_az_el_roll_deg=view_az_el_roll_deg,
        grid=debug.draw_sofast_setup_axis_grid,
        axis_prefix="World ",
    )

    # Import here, to avoid circular import.
    import opencsp.app.sofast.lib.SofastConfiguration as sfcfg

    # Draw the vector from camera to mirror centroid.
    v_camera_to_centroid_color = 'red'
    sfcfg.draw_annotated_vector_from_origin(
        view=fig_rec.view,
        vector=v_cam_optic_centroid_cam_exp,
        transform=trans_camera_world,
        short_name='C->Mc',
        long_name='Camera to Mirror Centroid',
        line_style=sfcfg.annotated_vector_line_style(color=v_camera_to_centroid_color),
        draw_base=True,
        base_style=sfcfg.annotated_vector_base_style(color=v_camera_to_centroid_color),
        draw_label=True,
        label_style=sfcfg.annotated_vector_label_style(color=v_camera_to_centroid_color),
    )

    if draw_reflection:
        # Construct the vector from mirror centroid to the screen origin.
        v_centroid_to_screen_color = 'purple'
        v_centroid_world = trans_camera_world.apply(v_cam_optic_centroid_cam_exp)
        # Mirror to screen
        sfcfg.draw_annotated_vector(
            view=fig_rec.view,
            tail=v_centroid_world,
            head=trans_screen_world.V,
            transform=txyz.identity_transform(),
            short_name='Mc->So',
            long_name='Mirror Centroid to Screen Origin',
            line_style=sfcfg.annotated_vector_line_style(color=v_centroid_to_screen_color),
            draw_base=True,
            base_style=sfcfg.annotated_vector_base_style(color=v_centroid_to_screen_color),
            draw_label=True,
            label_style=sfcfg.annotated_vector_label_style(color=v_centroid_to_screen_color),
        )
        # Reflection normal
        reflection_normal_color = 'brown'
        sfcfg.draw_annotated_vector(
            view=fig_rec.view,
            tail=v_centroid_world,
            head=v_centroid_world + u_reflection_norm_cam.rotate(trans_camera_world.R),
            transform=txyz.identity_transform(),
            short_name='Rnorm',
            long_name='Reflection Surface Normal',
            line_style=sfcfg.annotated_vector_line_style(color=reflection_normal_color, linewidth=0.5),
            draw_base=True,
            base_style=sfcfg.annotated_vector_base_style(color=reflection_normal_color),
            draw_label=True,
            label_style=sfcfg.annotated_vector_label_style(color=reflection_normal_color),
        )
    else:
        # Draw the vector from camera to mirror origin.
        v_camera_to_origin_color = 'blue'
        sfcfg.draw_annotated_vector_from_origin(
            view=fig_rec.view,
            vector=v_cam_optic_origin_cam_exp,
            transform=trans_camera_world,
            short_name='C->Mo',
            long_name='Camera to Mirror Origin',
            line_style=sfcfg.annotated_vector_line_style(color=v_camera_to_origin_color),
            draw_base=True,
            base_style=sfcfg.annotated_vector_base_style(color=v_camera_to_origin_color),
            draw_label=True,
            label_style=sfcfg.annotated_vector_label_style(color=v_camera_to_origin_color),
        )

    # Finish and save the figure.
    sdfs.finish_debug_3d_figure(figure_title, 'geometry', fig_rec, debug)


# SOFAST SETUP, WITH MIRROR POSITION ESTIMATE CONSIDERING TRANSLATION AND ROTATION,
# INCLUDING CENTROID SURFACE NORMAL


def figure_setup_mirror_fit_translation_rotation(
    figure_title: str,
    camera: Camera,
    facet_data: DefinitionFacet,
    orientation: SpatialOrientation,
    debug: DebugOpticsGeometry,
    v_facet_centroid: Vxyz,
    u_facet_centroid_normal: Uxyz,
    v_cam_optic_centroid_cam_exp: Vxyz,
    r_cam_optic_exp_B: Rotation,
    u_reflection_norm_cam: Uxyz,
    draw_reflection: bool,
):
    """
    Sets up and draws a figure showing SOFAST layout, including screen, camera,
    and mirror, with facet position estimated using translation and rotation.
    The facet centroid surface normal is included.
    """
    view_spec_az_el_roll_list = [
        (vs.view_spec_3d(), None),
        (vs.view_spec_xy(), None),
        (vs.view_spec_xz(), None),
        (vs.view_spec_yz(), None),
        (vs.view_spec_3d(), (20, 20, 0)),
        (vs.view_spec_3d(), (-15, 135, 180)),
    ]
    for view_spec_az_el_roll in view_spec_az_el_roll_list:
        view_spec = view_spec_az_el_roll[0]
        az_el_roll_deg = view_spec_az_el_roll[1]
        if az_el_roll_deg is None:
            this_title = figure_title
        else:
            this_title = figure_title + ' (Az,El,Roll)=' + str(az_el_roll_deg)
        figure_setup_mirror_fit_translation_rotation_aux(
            this_title,
            view_spec,
            camera,
            facet_data,
            orientation,
            debug,
            v_facet_centroid,
            u_facet_centroid_normal,
            v_cam_optic_centroid_cam_exp,
            r_cam_optic_exp_B,
            u_reflection_norm_cam,
            draw_reflection,
            view_az_el_roll_deg=az_el_roll_deg,
        )


def figure_setup_mirror_fit_translation_rotation_aux(
    figure_title: str,
    view_spec: dict,
    camera: Camera,
    facet_data: DefinitionFacet,
    orientation: SpatialOrientation,
    debug: DebugOpticsGeometry,
    v_facet_centroid: Vxyz,
    u_facet_centroid_normal: Uxyz,
    v_cam_optic_centroid_cam_exp: Vxyz,
    r_cam_optic_exp_B: Rotation,
    u_reflection_norm_cam: Uxyz,
    draw_reflection: bool,
    view_az_el_roll_deg: float = None,
) -> None:
    """Supports routine without aux extension."""

    # Camera-to-screen transform.
    trans_cam_screen = orientation.trans_screen_cam.inv()
    # # Mirror pose.
    # # mirror_rotation = Rotation.from_euler('zx', [-30.0, 45.0], degrees=True)
    # # mirror_translation = Vxyz([0.25, -0.325, 0.1])
    # mirror_rotation = Rotation.from_euler('zxz', [180.0, -75.0, -20.0], degrees=True)
    # mirror_translation = Vxyz([-0.45, 0.0, 1.0])
    # trans_mirror_world = txyz.TransformXYZ.from_R_V(R=mirror_rotation, V=mirror_translation)
    # # Mirror-to-screen transform.
    # trans_mirror_screen = debug.trans_screen_world.inv() * trans_mirror_world

    # # Find expected position of optic origin
    # v_cam_optic_cam_exp = v_cam_optic_centroid_cam_exp - v_facet_centroid.rotate(r_cam_optic_exp_B2.inv())

    # Mirror pose.
    # v_cam_optic_origin_cam_exp = v_cam_optic_centroid_cam_exp - v_facet_centroid
    # v_cam_optic_origin_cam_exp = v_cam_optic_centroid_cam_exp - v_facet_centroid.rotate(r_cam_optic_exp_B.inv())
    v_cam_optic_origin_cam_exp = v_cam_optic_centroid_cam_exp - v_facet_centroid.rotate(r_cam_optic_exp_B)
    v_cam_optic_origin_screen_exp = trans_cam_screen.apply(v_cam_optic_origin_cam_exp)
    trans_mirror_screen = txyz.TransformXYZ.from_R_V(R=r_cam_optic_exp_B, V=v_cam_optic_origin_screen_exp)

    # # In other words, position of optic origin in 3-d space, in camera coordinates.
    # translation_2 = v_cam_optic_centroid_cam_exp - v_facet_centroid.rotate(r_cam_optic_exp_B)
    # # Computed 3-d position of facet corners in camera coordinates.
    # v_facet_corners_cam_2 = v_facet_corners.rotate(r_cam_optic_exp_B) + translation_2

    # Draw.
    fig_rec, trans_screen_world, trans_camera_world, trans_mirror_world = sdfs.start_and_draw_sofast_setup_figure(
        figure_title=figure_title,
        view_spec=view_spec,
        camera=camera,
        facet_data=facet_data,
        trans_cam_screen=trans_cam_screen,
        trans_mirror_screen=trans_mirror_screen,
        debug=debug,
        view_az_el_roll_deg=view_az_el_roll_deg,
        grid=debug.draw_sofast_setup_axis_grid,
        axis_prefix="World ",
    )

    # Import here, to avoid circular import.
    import opencsp.app.sofast.lib.SofastConfiguration as sfcfg

    # Draw the vector from camera to mirror centroid.
    v_camera_to_centroid_color = 'red'
    sfcfg.draw_annotated_vector_from_origin(
        view=fig_rec.view,
        vector=v_cam_optic_centroid_cam_exp,
        transform=trans_camera_world,
        short_name='C->Mc',
        long_name='Camera to Mirror Centroid',
        line_style=sfcfg.annotated_vector_line_style(color=v_camera_to_centroid_color),
        draw_base=True,
        base_style=sfcfg.annotated_vector_base_style(color=v_camera_to_centroid_color),
        draw_label=True,
        label_style=sfcfg.annotated_vector_label_style(color=v_camera_to_centroid_color),
    )

    # Construct the vector from mirror centroid to the screen origin.
    v_centroid_to_screen_color = 'purple'
    v_centroid_world = trans_camera_world.apply(v_cam_optic_centroid_cam_exp)
    # Mirror to screen
    sfcfg.draw_annotated_vector(
        view=fig_rec.view,
        tail=v_centroid_world,
        head=trans_screen_world.V,
        transform=txyz.identity_transform(),
        short_name='Mc->So',
        long_name='Mirror Centroid to Screen Origin',
        line_style=sfcfg.annotated_vector_line_style(color=v_centroid_to_screen_color),
        draw_base=True,
        base_style=sfcfg.annotated_vector_base_style(color=v_centroid_to_screen_color),
        draw_label=True,
        label_style=sfcfg.annotated_vector_label_style(color=v_centroid_to_screen_color),
    )
    # Reflection normal
    reflection_normal_color = 'brown'
    sfcfg.draw_annotated_vector(
        view=fig_rec.view,
        tail=v_centroid_world,
        head=v_centroid_world + u_reflection_norm_cam.rotate(trans_camera_world.R),
        transform=txyz.identity_transform(),
        short_name='Rnorm',
        long_name='Reflection Surface Normal',
        line_style=sfcfg.annotated_vector_line_style(color=reflection_normal_color, linewidth=0.5),
        draw_base=True,
        base_style=sfcfg.annotated_vector_base_style(color=reflection_normal_color),
        draw_label=True,
        label_style=sfcfg.annotated_vector_label_style(color=reflection_normal_color),
    )
    # Current mirror surface normal
    centroid_normal_color = 'pink'
    u_centroid_norm_cam = u_facet_centroid_normal.rotate(r_cam_optic_exp_B)
    v_centroid_norm_cam = u_centroid_norm_cam.as_Vxyz() * 0.5  # Make shorter
    sfcfg.draw_annotated_vector(
        view=fig_rec.view,
        tail=v_centroid_world,
        head=v_centroid_world + v_centroid_norm_cam.rotate(trans_camera_world.R),
        transform=txyz.identity_transform(),
        short_name='Cnorm',
        long_name='Centroid Surface Normal',
        line_style=sfcfg.annotated_vector_line_style(color=centroid_normal_color, linewidth=0.5),
        draw_base=True,
        base_style=sfcfg.annotated_vector_base_style(color=centroid_normal_color),
        draw_label=True,
        label_style=sfcfg.annotated_vector_label_style(color=centroid_normal_color),
    )

    # Draw the vector from camera to mirror origin.
    v_camera_to_origin_color = 'blue'
    sfcfg.draw_annotated_vector_from_origin(
        view=fig_rec.view,
        vector=v_cam_optic_origin_cam_exp,
        transform=trans_camera_world,
        short_name='C->Mo',
        long_name='Camera to Mirror Origin',
        line_style=sfcfg.annotated_vector_line_style(color=v_camera_to_origin_color),
        draw_base=True,
        base_style=sfcfg.annotated_vector_base_style(color=v_camera_to_origin_color),
        draw_label=True,
        label_style=sfcfg.annotated_vector_label_style(color=v_camera_to_origin_color),
    )

    # Finish and save the figure.
    sdfs.finish_debug_3d_figure(figure_title, 'geometry', fig_rec, debug)


# SOFAST SETUP, MIRROR POSE REFINED BY solvePNP()


def figure_setup_mirror_refined_by_solvePnP(
    figure_title: str,
    camera: Camera,
    facet_data: DefinitionFacet,
    orientation: SpatialOrientation,
    debug: DebugOpticsGeometry,
    v_facet_centroid: Vxyz,
    v_cam_optic_origin_cam_refine_1: Vxyz,
    r_optic_cam_refine_1: Rotation,
    u_reflection_norm_cam: Uxyz,
    draw_reflection: bool,
):
    """
    Sets up and draws a figure showing SOFAST layout, including screen, camera,
    and mirror, with facet position estimated using translation and rotation,
    but not the facet centroid surface normal.
    """
    view_spec_az_el_roll_list = [
        (vs.view_spec_3d(), None),
        (vs.view_spec_xy(), None),
        (vs.view_spec_xz(), None),
        (vs.view_spec_yz(), None),
        (vs.view_spec_3d(), (20, 20, 0)),
        (vs.view_spec_3d(), (-15, 135, 180)),
    ]
    for view_spec_az_el_roll in view_spec_az_el_roll_list:
        view_spec = view_spec_az_el_roll[0]
        az_el_roll_deg = view_spec_az_el_roll[1]
        if az_el_roll_deg is None:
            this_title = figure_title
        else:
            this_title = figure_title + ' (Az,El,Roll)=' + str(az_el_roll_deg)
        figure_setup_mirror_refined_by_solvePnP_aux(
            this_title,
            view_spec,
            camera,
            facet_data,
            orientation,
            debug,
            v_facet_centroid,
            v_cam_optic_origin_cam_refine_1,
            r_optic_cam_refine_1,
            u_reflection_norm_cam,
            draw_reflection,
            view_az_el_roll_deg=az_el_roll_deg,
        )


def figure_setup_mirror_refined_by_solvePnP_aux(
    figure_title: str,
    view_spec: dict,
    camera: Camera,
    facet_data: DefinitionFacet,
    orientation: SpatialOrientation,
    debug: DebugOpticsGeometry,
    v_facet_centroid: Vxyz,
    v_cam_optic_origin_cam_refine_1: Vxyz,
    r_optic_cam_refine_1: Rotation,
    u_reflection_norm_cam: Uxyz,
    draw_reflection: bool,
    view_az_el_roll_deg: float = None,
) -> None:
    """Supports routine without aux extension."""

    # Camera-to-screen transform.
    trans_cam_screen = orientation.trans_screen_cam.inv()
    # # Mirror pose.
    # # mirror_rotation = Rotation.from_euler('zx', [-30.0, 45.0], degrees=True)
    # # mirror_translation = Vxyz([0.25, -0.325, 0.1])
    # mirror_rotation = Rotation.from_euler('zxz', [180.0, -75.0, -20.0], degrees=True)
    # mirror_translation = Vxyz([-0.45, 0.0, 1.0])
    # trans_mirror_world = txyz.TransformXYZ.from_R_V(R=mirror_rotation, V=mirror_translation)
    # # Mirror-to-screen transform.
    # trans_mirror_screen = debug.trans_screen_world.inv() * trans_mirror_world

    # Mirror pose.
    # v_cam_optic_origin_cam_exp = v_cam_optic_centroid_cam_exp - v_facet_centroid
    # &&&& DELETE-SCAFFOLDING -- USED TO BE NOT .INV()
    # &&&& DELETE-SCAFFOLDING -- NOT NEEDED
    # # v_cam_optic_origin_cam_exp = v_cam_optic_centroid_cam_exp - v_facet_centroid.rotate(r_cam_optic_exp_A)
    # v_cam_optic_origin_cam_exp = v_cam_optic_centroid_cam_exp  # &&&& DELETE-SCAFFOLDING -- USED TO INCLUDE OFFSET
    # &&&& DELETE-SCAFFOLDING -- DOCUMENT THIS
    v_cam_optic_centroid_cam_exp = v_cam_optic_origin_cam_refine_1 + v_facet_centroid.rotate(r_optic_cam_refine_1)

    # v_cam_optic_origin_cam_exp = v_cam_optic_centroid_cam_exp - v_facet_centroid.rotate(r_cam_optic_exp_A.inv())
    v_cam_optic_origin_screen_exp = trans_cam_screen.apply(v_cam_optic_origin_cam_refine_1)
    trans_mirror_screen = txyz.TransformXYZ.from_R_V(R=r_optic_cam_refine_1, V=v_cam_optic_origin_screen_exp)
    # trans_mirror_screen = txyz.TransformXYZ.from_R_V(R=r_cam_optic_exp_A.inv(), V=v_cam_optic_origin_screen_exp)

    # # In other words, position of optic origin in 3-d space, in camera coordinates.
    # translation_2 = v_cam_optic_centroid_cam_exp - v_facet_centroid.rotate(r_cam_optic_exp_A)
    # # Computed 3-d position of facet corners in camera coordinates.
    # v_facet_corners_cam_2 = v_facet_corners.rotate(r_cam_optic_exp_A) + translation_2

    # Draw.
    fig_rec, trans_screen_world, trans_camera_world, trans_mirror_world = sdfs.start_and_draw_sofast_setup_figure(
        figure_title=figure_title,
        view_spec=view_spec,
        camera=camera,
        facet_data=facet_data,
        trans_cam_screen=trans_cam_screen,
        trans_mirror_screen=trans_mirror_screen,
        debug=debug,
        view_az_el_roll_deg=view_az_el_roll_deg,
        grid=debug.draw_sofast_setup_axis_grid,
        axis_prefix="World ",
    )

    # Import here, to avoid circular import.
    import opencsp.app.sofast.lib.SofastConfiguration as sfcfg

    # Draw the vector from camera to mirror centroid.
    v_camera_to_centroid_color = 'red'
    sfcfg.draw_annotated_vector_from_origin(
        view=fig_rec.view,
        vector=v_cam_optic_centroid_cam_exp,
        transform=trans_camera_world,
        short_name='C->Mc',
        long_name='Camera to Mirror Centroid',
        line_style=sfcfg.annotated_vector_line_style(color=v_camera_to_centroid_color),
        draw_base=True,
        base_style=sfcfg.annotated_vector_base_style(color=v_camera_to_centroid_color),
        draw_label=True,
        label_style=sfcfg.annotated_vector_label_style(color=v_camera_to_centroid_color),
    )

    if draw_reflection:
        # Construct the vector from mirror centroid to the screen origin.
        v_centroid_to_screen_color = 'purple'
        v_centroid_world = trans_camera_world.apply(v_cam_optic_centroid_cam_exp)
        # Mirror to screen
        sfcfg.draw_annotated_vector(
            view=fig_rec.view,
            tail=v_centroid_world,
            head=trans_screen_world.V,
            transform=txyz.identity_transform(),
            short_name='Mc->So',
            long_name='Mirror Centroid to Screen Origin',
            line_style=sfcfg.annotated_vector_line_style(color=v_centroid_to_screen_color),
            draw_base=True,
            base_style=sfcfg.annotated_vector_base_style(color=v_centroid_to_screen_color),
            draw_label=True,
            label_style=sfcfg.annotated_vector_label_style(color=v_centroid_to_screen_color),
        )
        # Reflection normal
        reflection_normal_color = 'brown'
        sfcfg.draw_annotated_vector(
            view=fig_rec.view,
            tail=v_centroid_world,
            head=v_centroid_world + u_reflection_norm_cam.rotate(trans_camera_world.R),
            transform=txyz.identity_transform(),
            short_name='Rnorm',
            long_name='Reflection Surface Normal',
            line_style=sfcfg.annotated_vector_line_style(color=reflection_normal_color, linewidth=0.5),
            draw_base=True,
            base_style=sfcfg.annotated_vector_base_style(color=reflection_normal_color),
            draw_label=True,
            label_style=sfcfg.annotated_vector_label_style(color=reflection_normal_color),
        )
    else:
        # Draw the vector from camera to mirror origin.
        v_camera_to_origin_color = 'blue'
        sfcfg.draw_annotated_vector_from_origin(
            view=fig_rec.view,
            vector=v_cam_optic_origin_cam_refine_1,
            transform=trans_camera_world,
            short_name='C->Mo',
            long_name='Camera to Mirror Origin',
            line_style=sfcfg.annotated_vector_line_style(color=v_camera_to_origin_color),
            draw_base=True,
            base_style=sfcfg.annotated_vector_base_style(color=v_camera_to_origin_color),
            draw_label=True,
            label_style=sfcfg.annotated_vector_label_style(color=v_camera_to_origin_color),
        )

    # Finish and save the figure.
    sdfs.finish_debug_3d_figure(figure_title, 'geometry', fig_rec, debug)
