import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
from opencsp.common.lib.render_control.RenderControlPointSeq import RenderControlPointSeq
from opencsp.common.lib.render_control.RenderControlSurface import RenderControlSurface

# from opencsp.common.lib.geometry.Resolution import Resolution


class RenderControlMirrorProjected:
    """
    A class for controlling the rendering of a mirror projection onto the (x,y) plane
    in a graphical environment.

    This class allows for the configuration of various visual aspects of a projection,
    including its lifted, projected, and connection line features.
    """

    def __init__(
        self,
        draw_lifted: bool = True,
        lifted_line_style: RenderControlPointSeq = None,
        lifted_vertex_style: RenderControlPointSeq = None,
        draw_projected: bool = True,
        projected_style: RenderControlPointSeq = None,
        draw_connection: bool = True,
        connection_style: RenderControlPointSeq = None,
        slice_n_vertices: int = 9,
        slice_fine_n_vertices: int = 41,
    ) -> None:
        """
        Initializes a RenderControlMirrorProjected object with the specified parameters.

        Parameters
        ----------
        draw_lifted : bool, optional
            Whether to draw the lifted boundary.
            Default True.
        lifted_line_style : RenderControlPointSeq | None, optional
            Style to draw the contour of the lifted slice.
            Default None.
        lifted_vertex_style : RenderControlPointSeq | None, optional
            Style to draw the coarse vertices of the lifted slice.
            Default None.
        draw_projected : bool, optional
            Whether to draw the projected boundary.
            Default True.
        projected_style: RenderControlPointSeq | None, optional
            Style to draw the projected slice.
            Default None.
        draw_connection : bool, optional
            Whether to draw the connection_lines.
            Default True.
        connection_style: RenderControlPointSeq | None, optional
            Style to draw the connection line between the projected and lifted slices.
            Default None.
        slice_n_vertices : int, optional
            Default 9.
        slice_fine_n_vertices:int, optional
            Default 41.
        """
        self.draw_lifted = draw_lifted
        self.lifted_line_style = lifted_line_style
        self.lifted_vertex_style = lifted_vertex_style
        self.draw_projected = draw_projected
        self.projected_style = projected_style
        self.draw_connection = draw_connection
        self.connection_style = connection_style
        self.slice_n_vertices = slice_n_vertices
        self.slice_fine_n_vertices = slice_fine_n_vertices


# Common Configurations


def mirror_contour(slice_n_vertices: int = 5) -> RenderControlMirrorProjected:
    """Style for drawing a contour illustrating a mirror surface"""
    return RenderControlMirrorProjected(
        lifted_line_style=rcps.outline(color='grey', linewidth=0.5), slice_n_vertices=slice_n_vertices
    )


def mirror_origin_contour(slice_n_vertices: int = 5) -> RenderControlMirrorProjected:
    """Style for drawing a contour illustrating a mirror surface slice passing through the (0,0) origin."""
    return RenderControlMirrorProjected(
        lifted_line_style=rcps.outline(color='grey', linewidth=1.0),
        lifted_vertex_style=rcps.marker(color='grey', markersize=1.3),
        projected_style=rcps.outline(color='grey', linewidth=0.75),
        connection_style=rcps.outline(color='grey', linewidth=0.75),
        slice_n_vertices=slice_n_vertices,
    )


def mirror_boundary(boundary_color: str = 'red', projected_color: str = 'blue') -> RenderControlMirrorProjected:
    """
    Style for drawing both the RegionXY curve on the (x,y) plane and the lifted boundary
    on the embedding surface, as well as the connection lines between them.
    """
    return RenderControlMirrorProjected(
        lifted_line_style=rcps.outline(color=boundary_color),
        lifted_vertex_style=rcps.marker(color=boundary_color, markersize=2),
        projected_style=rcps.outline(color=projected_color),
        connection_style=rcps.outline(color=projected_color, linewidth=0.6),
    )


def mirror_lifted(boundary_color: str = 'red') -> RenderControlMirrorProjected:
    """Style for drawing the mirror boundary resulting from lifting the RegionXY up to the embedding surface."""
    return RenderControlMirrorProjected(
        draw_lifted=True,
        lifted_line_style=rcps.outline(color=boundary_color),
        lifted_vertex_style=rcps.marker(color=boundary_color, markersize=2),
        draw_projected=False,
        draw_connection=False,
    )


def mirror_projected(projected_color: str = 'red') -> RenderControlMirrorProjected:
    """Style for drawing the RegionXY curve on the (x,y) plane that will be lifted to the embedding surface."""
    return RenderControlMirrorProjected(
        draw_lifted=False,
        draw_projected=True,
        projected_style=rcps.outline(color=projected_color),
        draw_connection=False,
    )
