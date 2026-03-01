import opencsp.common.lib.render_control.RenderControlSofastWorld as rcsw
import opencsp.common.lib.render_control.RenderControlSofastScreen as rcss
import opencsp.common.lib.render_control.RenderControlSofastCamera as rcsc
import opencsp.common.lib.render_control.RenderControlSofastMirror as rcsm


class RenderControlSofastSetup:
    """
    A class for controlling the rendering of a SOFAST setup.

    This class controls the rendering settings for the world, screen, camera, and mirror.
    """

    def __init__(
        self,
        draw_world: bool = True,
        sofast_world_style: rcsw.RenderControlSofastWorld = rcsw.RenderControlSofastWorld(),
        draw_screen: bool = True,
        sofast_screen_style: rcss.RenderControlSofastScreen = rcss.RenderControlSofastScreen(),
        draw_camera: bool = True,
        sofast_camera_style: rcsc.RenderControlSofastCamera = rcsc.RenderControlSofastCamera(),
        draw_mirror: bool = True,
        sofast_mirror_style: rcsm.RenderControlSofastMirror = rcsm.RenderControlSofastMirror(),
    ) -> None:
        # Initializes a RenderControlSofastSetup object with the specified parameters.
        self.draw_world = draw_world
        self.sofast_world_style = sofast_world_style
        self.draw_screen = draw_screen
        self.sofast_screen_style = sofast_screen_style
        self.draw_camera = draw_camera
        self.sofast_camera_style = sofast_camera_style
        self.draw_mirror = draw_mirror
        self.sofast_mirror_style = sofast_mirror_style


# Common Configurations
