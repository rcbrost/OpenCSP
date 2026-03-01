"""
A class for controlling the rendering of a SOFAST world.
"""


class RenderControlSofastWorld:
    """
    A class for controlling the rendering of the world in a SOFAST setup.
    """

    def __init__(
        self, color: str | tuple[float, float, float] = 'lightgrey', draw_corners: bool = True, draw_edges: bool = True
    ) -> None:
        """
        Initializes a RenderControlSofastWorld object with the specified parameters.

        Parameters
        ----------
        color : str | tuple[float, float, float], optional
            The color to draw all screen elements.  Default 'green'.
        draw_corners: bool, optional
            Whether to draw dots at the corners of the world box.  Default False.
        draw_edges: bool, optional
            Whether to draw edges delineating the world box.  Default True.
        """
        self.color = color
        self.draw_corners = draw_corners
        self.draw_edges = draw_edges


# Common Configurations
