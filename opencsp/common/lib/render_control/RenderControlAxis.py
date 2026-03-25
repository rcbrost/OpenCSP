""" """


class RenderControlAxis:
    """
    Render control for plot axes.
    """

    def __init__(
        self,
        x_label: str = 'x',
        y_label: str = 'y',
        z_label: str = 'z',
        p_label: str = 'p',
        q_label: str = 'q',
        w_label: str = 'w',
        draw_axes: bool = True,
        grid: bool = True,
    ) -> None:
        super(RenderControlAxis, self).__init__()

        # Axis control.
        self.x_label = x_label
        self.y_label = y_label
        self.z_label = z_label
        self.p_label = p_label
        self.q_label = q_label
        self.w_label = w_label
        self.draw_axes = draw_axes
        self.grid = grid


def meters(draw_axes: bool = True, grid: bool = True, axis_prefix: str = None) -> RenderControlAxis:
    """
    Labels indicating units of meters.

    axis_prefix should include a separator character.
    For example, if the goal is to have an axis label of "World x (m)",
    then axis_prefix should be "World ".
    """
    if axis_prefix == None:
        axis_prefix_str = ''
    else:
        axis_prefix_str = axis_prefix
    return RenderControlAxis(
        x_label=(axis_prefix_str + 'x (m)'),
        y_label=(axis_prefix_str + 'y (m)'),
        z_label=(axis_prefix_str + 'z (m)'),
        p_label=(axis_prefix_str + 'p (m)'),
        q_label=(axis_prefix_str + 'q (m)'),
        w_label=(axis_prefix_str + 'w (m)'),
        draw_axes=draw_axes,
        grid=grid,
    )


def latlon(
    decimal_t_degminsecs_f: bool = True, draw_axes: bool = True, grid: bool = True, axis_prefix: str = None
) -> RenderControlAxis:
    """
    Labels indicating units of latitude and longitude.

    axis_prefix should include a separator character.
    For example, if the goal is to have an axis label of "World x (deg,min,sec)",
    then axis_prefix should be "World ".
    """
    if axis_prefix == None:
        axis_prefix_str = ''
    else:
        axis_prefix_str = axis_prefix
    unit = "deg" if decimal_t_degminsecs_f else "deg,min,sec"
    return RenderControlAxis(
        x_label=axis_prefix_str + f"longitude ({unit})",
        y_label=axis_prefix_str + f"latitude ({unit})",
        z_label=axis_prefix_str + f"z ({unit})",
        p_label=axis_prefix_str + f"p ({unit})",
        q_label=axis_prefix_str + f"q ({unit})",
        w_label=axis_prefix_str + f"w ({unit})",
        draw_axes=draw_axes,
        grid=grid,
    )


def image(draw_axes: bool = True, grid: bool = True, axis_prefix: str = None) -> RenderControlAxis:
    """
    Labels indicating image.

    axis_prefix should include a separator character.
    For example, if the goal is to have an axis label of "World x (pix)",
    then axis_prefix should be "World ".
    """
    if axis_prefix == None:
        axis_prefix_str = ''
    else:
        axis_prefix_str = axis_prefix
    return RenderControlAxis(
        x_label=axis_prefix_str + 'x N/A',
        y_label=axis_prefix_str + 'y N/A',
        z_label=axis_prefix_str + 'z N/A',
        p_label=axis_prefix_str + 'x (pix)',
        q_label=axis_prefix_str + 'y (pix)',
        w_label=axis_prefix_str + 'w N/A',
        draw_axes=draw_axes,
        grid=grid,
    )
