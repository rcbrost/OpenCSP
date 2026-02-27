"""Functions to simplify code that outputs SOFAST debugging figures."""

import matplotlib.pyplot as plt

from opencsp.app.sofast.lib.DebugOpticsGeometry import DebugOpticsGeometry
from opencsp.common.lib.geometry.Vxy import Vxy
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render.view_spec as vs
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlFigure as rcfg


def plot_labeled_points(
    pts: Vxy, marker_size: int = 30, point_color: str = 'r', label_color: str = 'g', legend_label: str = ''
) -> None:
    """Plots labeled points on axis for debugging"""
    plt.scatter(*pts.data, s=marker_size, c=point_color, label=legend_label)
    for idx, pt in enumerate(pts):
        plt.text(*pt.data, idx, color=label_color)


def start_debug_image_figure(figure_title: str) -> rcfg.RenderControlFigure:
    """Begins a debug figure setup to show an image,
    possibly with other annotations."""
    fig_rec = fm.setup_figure(
        figure_control=rcfg.RenderControlFigure(tile=False),
        axis_control=rca.image(grid=False),
        view_spec=vs.view_spec_im(),
        title=figure_title,
    )
    return fig_rec


def finish_debug_image_figure(
    figure_title: str, phase: str, fig_rec: rcfg.RenderControlFigure, debug: DebugOpticsGeometry
) -> None:
    """Closes and saves debug image figure."""
    figure_title_clean = figure_title.replace(' ', '_').replace(',', '')
    figure_file_body = f"{debug.figure_idx:02d}_{phase}_{figure_title_clean}"
    debug.figure_idx += 1
    fig_rec.save(
        output_dir=debug.save_dir,
        output_file_body=figure_file_body,
        dpi=200,
        format='png',
        close_after_save=True,
        include_view_suffix=False,
        include_limit_suffix=False,
    )


def start_debug_3d_figure(
    figure_title: str, view_spec: dict = vs.view_spec_3d(), equal: bool = True, grid=True, figsize=(6.4, 4.8)  # Inch
) -> rcfg.RenderControlFigure:
    """Begins a debug figure setup to show data in a 3-d space."""
    fig_rec = fm.setup_figure_for_3d_data(
        figure_control=rcfg.RenderControlFigure(tile=False, figsize=figsize),
        axis_control=rca.meters(grid=grid),
        view_spec=view_spec,
        equal=equal,
        title=figure_title,
    )
    return fig_rec


def finish_debug_3d_figure(
    figure_title: str, phase: str, fig_rec: rcfg.RenderControlFigure, debug: DebugOpticsGeometry
) -> None:
    """Closes and saves debug 3-d figure."""
    figure_title_clean = figure_title.replace(' ', '_').replace(',', '')
    figure_file_body = f"{debug.figure_idx:02d}_{phase}_{figure_title_clean}"
    debug.figure_idx += 1
    fig_rec.save(
        output_dir=debug.save_dir,
        output_file_body=figure_file_body,
        dpi=200,
        format='png',
        close_after_save=True,
        include_view_suffix=True,
        include_limit_suffix=False,
    )
