"""Class holding debug information for proces_optics_geometry calculations"""

from opencsp.app.sofast.lib.DisplayShape import DisplayShape as Display
from opencsp.common.lib.csp.MirrorParametric import MirrorParametric
import opencsp.common.lib.geometry.TransformXYZ as txyz


class DebugOpticsGeometry:
    """Class for holding debug data for used in 'process_optics_geometry"""

    def __init__(self):
        self.debug_active: bool = False
        """To activate geometry debugging. Default False"""
        self.figures: list = []
        """List to hold figure objects once created."""
        self.save_dir: str = None
        """Where to save figure objects."""
        self.figure_idx: int = 0
        """Incrementing figure index, so figures appear in order created."""
        self.world_box: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
            (-1, 1),
            (-1, 1),
            (0, 2),
        )
        """Used for diagnostic output showing SOFAST setup."""
        self.trans_screen_world: txyz.TransformXYZ | None = None
        """Used for diagnostic output showing SOFAST setup."""
        z_axis_fov_distance: float = 1.0  # m
        """Used for diagnostic output showing SOFAST setup."""
        mirror_needle_length: float = 0.1  # m
        """Used for diagnostic output showing SOFAST setup."""
        axis_length: float = 0.1  # m
        """Used for diagnostic output showing SOFAST setup."""
        self.display: Display | None = None
        """Used for diagnostic output showing SOFAST setup."""
        self.mirror: MirrorParametric | None = None
        """Used for diagnostic output showing SOFAST setup."""
        self.draw_sofast_setup_axis_grid: bool = True
        """Used for diagnostic output showing SOFAST setup."""
        self.draw_sofast_setup_embedding_mirror: bool = True
        """Used for diagnostic output showing SOFAST setup."""
        self.draw_sofast_setup_mirror_projection: bool = True
        """Used for diagnostic output showing SOFAST setup."""
