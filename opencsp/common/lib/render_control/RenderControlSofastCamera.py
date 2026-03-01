"""
A class for controlling the rendering of a SOFAST camera.
"""


class RenderControlSofastCamera:
    """
    A class for controlling the rendering of the camera in a SOFAST setup.
    """

    def __init__(self, color: str | tuple[float, float, float] = 'cyan', show_fov_lens_distortion: bool = True) -> None:
        """
        Initializes a RenderControlSofastCamera object with the specified parameters.

        Parameters
        ----------
        color : str | tuple[float, float, float], optional
            The color to draw all camera elements.  Default 'cyan'.
        show_fov_lens_distortion : bool, optional
            Whether to draw adition field of view vertices so that the
            field of view shown the effect of lens distortion.  This is
            informative, but at the penalty of additional lines cluttering
            the view.  Default True.
        """
        self.color = color
        self.show_fov_lens_distortion = show_fov_lens_distortion


# Common Configurations
