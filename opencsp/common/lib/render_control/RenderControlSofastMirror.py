"""
A class for controlling the rendering of a SOFAST mirror.
"""

import opencsp.common.lib.render_control.RenderControlMirror as rcm
import opencsp.common.lib.render_control.RenderControlMirrorEmbedded as rcme
import opencsp.common.lib.render_control.RenderControlMirrorProjected as rcmp
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps


class RenderControlSofastMirror:
    """
    A class for controlling the rendering of the mirror in a SOFAST setup.
    """

    def __init__(
        self,
        color: str | tuple[float, float, float] = 'magenta',
        # Doesn't use color, so okay to set default here.
        mirror_style: rcm.RenderControlMirror = rcm.RenderControlMirror(),
        draw_embedding_mirror: bool = False,
        # Doesn't use color, so okay to set default here.
        embedding_mirror_style: rcme.RenderControlMirrorEmbedded = rcme.standard_embedding_mirror(),
        # For the following styles, don't put default styles here,
        # because default values can't respond to color.
        draw_lifted_boundary: bool = True,
        draw_lifted_boundary_lines: bool = True,
        draw_lifted_boundary_vertices: bool = True,
        # Split the lifted line and vertex styles, because drawing generates many small line segments
        # to achieve curved edges between vertices.
        lifted_line_style: rcps.RenderControlPointSeq = None,
        lifted_vertex_style: rcps.RenderControlPointSeq = None,
        draw_projected_boundary: bool = False,
        projected_boundary_style: rcps.RenderControlPointSeq = None,
        draw_connection_lines: bool = False,
        connection_style: rcps.RenderControlPointSeq = None,
    ) -> None:
        """
        Initializes a RenderControlSofastMirror object with the specified parameters.

        Parameters
        ----------
        color : str | tuple[float, float, float], optional
            The color to draw all Mirror elements.  Default 'magenta'.
        mirror_style : rcm.RenderControlMirror, optional
            Render control for the mirror.
        draw_embedding_mirror : bool, optional
            Whether to draw the mirror embedding surface.
            Default False.
        embedding_mirror_style : rcme.RenderControlMirrorEmbedded, optional
            Style to draw the mirror embedding surface.
            Default rcme.standard_embedding_mirror()
        draw_lifted_boundary : bool, optional
            Whether to draw the lifted boundary.
            Default True.
        lifted_line_style : rcps.RenderControlPointSeq | None, optional
            Style to draw the contour of the lifted slice.
            Default None.
        lifted_vertex_style : rcps.RenderControlPointSeq | None, optional
            Style to draw the coarse vertices of the lifted slice.
            Default None.
        draw_projected_boundary : bool, optional
            Whether to draw the projected boundary.
            Default True.
        projected_boundary_style: rcps.RenderControlPointSeq | None, optional
            Style to draw the projected boundary.
            Default None.
        draw_connection_lines : bool, optional
            Whether to draw the connection_lines.
            Default True.
        connection_style: rcps.RenderControlPointSeq | None, optional
            Style to draw the connection line between the projected and lifted slices.
            Default None.
        """
        self.color = color
        # Doesn't use color, so simply pass through.
        self.mirror_style = mirror_style
        self.draw_embedding_mirror = draw_embedding_mirror
        # Doesn't use color, so simply pass through.
        self.embedding_mirror_style = embedding_mirror_style
        # We need to set color for all of the following styles.
        # Lifted boundary.
        if draw_lifted_boundary:
            if draw_lifted_boundary_lines:
                if lifted_line_style is None:
                    lifted_line_style = rcps.outline(color=color)
            if draw_lifted_boundary_vertices:
                if lifted_vertex_style is None:
                    lifted_vertex_style = rcps.marker(color=color, markersize=2)
        # Projected RegionXy.
        if draw_projected_boundary:
            if projected_boundary_style is None:
                projected_boundary_style = rcps.outline(color=color)
        # Connection lines.
        if draw_connection_lines:
            if connection_style is None:
                connection_style = rcps.outline(color=color, linewidth=0.6)

        # Set the needed RenderControlMirrorProjected object.
        self.mirror_projected_style = rcmp.RenderControlMirrorProjected(
            draw_lifted=draw_lifted_boundary,
            lifted_line_style=lifted_line_style,
            lifted_vertex_style=lifted_vertex_style,
            draw_projected=draw_projected_boundary,
            projected_style=projected_boundary_style,
            draw_connection=draw_connection_lines,
            connection_style=connection_style,
        )


# Common Configurations


def tight_embedding_mirror(margin=0.01, round_to=0.25) -> RenderControlSofastMirror:
    sofast_mirror_style = RenderControlSofastMirror()
    sofast_mirror_style.embedding_mirror_style = rcme.RenderControlMirrorEmbedded(margin=margin, round_to=round_to)
    return sofast_mirror_style
