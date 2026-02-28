"""Class holding debug information for proces_optics_geometry calculations"""


class DebugOpticsGeometry:
    """Class for holding debug data for used in 'process_optics_geometry"""

    def __init__(self):
        self.debug_active: bool = False
        """To activate geometry debugging. Default False"""
        self.figures: list = []
        """List to hold figure objects once created."""
        self.save_dir = None
        """Where to save figure objects."""
        self.figure_idx = 0
        """Incrementing figure index, so figures appear in order created."""
        self.display = None
        """Used for diagnostic output showing SOFAST setup."""
        self.mirror = None
        """Used for diagnostic output showing SOFAST setup."""
        self.draw_sofast_setup_axis_grid = True
        """Used for diagnostic output showing SOFAST setup."""
        self.draw_sofast_setup_embedding_mirror = True
        """Used for diagnostic output showing SOFAST setup."""
        self.draw_sofast_setup_mirror_projection = True
        """Used for diagnostic output showing SOFAST setup."""
