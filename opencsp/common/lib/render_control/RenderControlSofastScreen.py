"""
A class for controlling the rendering of a SOFAST screen.
"""


class RenderControlSofastScreen:
    """
    A class for controlling the rendering of the screen in a SOFAST setup.
    """

    def __init__(self, color: str | tuple[float, float, float] = 'green') -> None:
        """
        Initializes a RenderControlSofastScreen object with the specified parameters.

        Parameters
        ----------
        color : str | tuple[float, float, float], optional
            The color to draw all screen elements.  Default 'green'.
        """
        self.color = color


# Common Configurations
