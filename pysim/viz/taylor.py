from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.projections import PolarAxes
from mpl_toolkits.axisartist import floating_axes, grid_finder

# TODO - move init with new correlation labels


class TaylorDiagram:
    """Creates a Taylor diagram
    
    Parameters
    ----------
    ref_point: float
        The reference point for the 

    fig :
    
    """

    def __init__(
        self,
        ref_point: float,
        fig: Optional[plt.figure] = None,
        subplot: Optional[int] = 111,
        extend_angle: bool = False,
        corr_range: Tuple[float, float] = (0, 10),
        ref_label: str = "Reference Point",
        angle_label: str = "Correlation",
        var_label: str = "Standard Deviation",
    ) -> None:

        self.angle_label = angle_label
        self.ref_label = ref_label
        self.var_label = var_label
        self.extend_angle = extend_angle
        # corr_locations
        corr_labels = np.array([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1])

        # extend
        if extend_angle:
            # Extend to negative correlations
            self.tmax = np.pi
            corr_labels = np.concatenate((-corr_labels[:0:-1], corr_labels))
        else:
            # limit to only positive correlations
            self.tmax = np.pi / 2.0

        # init figure
        if fig is None:
            fig = plt.figure(figsize=(8, 8))

        # extend or not
        self.smin = corr_range[0] * ref_point
        self.smax = (corr_range[1] / 100) * ref_point + ref_point

        corr_ticks = np.arccos(corr_labels)
        gl1 = grid_finder.FixedLocator(corr_ticks)
        tf1 = grid_finder.DictFormatter(dict(zip(corr_ticks, map(str, corr_labels))))

        # Grid Helper
        ghelper = floating_axes.GridHelperCurveLinear(
            aux_trans=PolarAxes.PolarTransform(),
            extremes=(0, self.tmax, self.smin, self.smax),
            grid_locator1=gl1,
            tick_formatter1=tf1,
        )

        # create graphical axies

        ax = floating_axes.FloatingSubplot(fig, subplot, grid_helper=ghelper)
        fig.add_subplot(ax)
        self.graph_axes = ax
        self.polar_axes = ax.get_aux_axes(PolarAxes.PolarTransform())

        self.sample_points = []
        # Setup Axes
        self.reset_axes()

    def add_reference_point(self, ref_point: float, *args, **kwargs) -> None:
        l, = self.polar_axes.plot([0], ref_point, *args, **kwargs)

        self.sample_points.append(l)

        return None

    def add_reference_line(self, ref_point: float, *args, **kwargs) -> None:

        t = np.linspace(0, self.tmax)
        r = np.zeros_like(t) + ref_point
        self.polar_axes.plot(t, r, *args, **kwargs)

        return None

    def add_grid(self, *args, **kwargs):

        self.graph_axes.grid(*args, **kwargs)

    def add_contours(self, ref_point: float, levels: int = 4, **kwargs) -> None:

        # create meshgrid of values
        rs, ts = np.meshgrid(
            np.linspace(self.smin, self.smax), np.linspace(0, self.tmax)
        )

        # calculate the distance
        dist = np.sqrt(ref_point ** 2 + rs ** 2 - 2 * ref_point * rs * np.cos(ts))

        self.contours = self.polar_axes.contour(ts, rs, dist, levels, **kwargs)
        return None

    def reset_axes(self):
        self._setup_angle_axes()
        self._setup_xaxis()
        self._setup_yaxis()

    def reset_axes_labels(
        self, angle_label: str = "Correlation", var_label: str = "Variance"
    ):
        # Adjust axes
        self.graph_axes.axis["left"].label.set_text(var_label)
        self.graph_axes.axis["top"].label.set_text(angle_label)

    def _setup_angle_axes(self):
        self.graph_axes.axis["top"].set_axis_direction("bottom")
        self.graph_axes.axis["top"].toggle(ticklabels=True, label=True)
        self.graph_axes.axis["top"].major_ticklabels.set_axis_direction("top")
        self.graph_axes.axis["top"].label.set_axis_direction("top")
        self.graph_axes.axis["top"].label.set_text(self.angle_label)

    def _setup_xaxis(self):
        self.graph_axes.axis["left"].set_axis_direction("bottom")
        self.graph_axes.axis["left"].label.set_text(self.var_label)

    def _setup_yaxis(self):
        self.graph_axes.axis["right"].set_axis_direction("top")
        self.graph_axes.axis["right"].toggle(ticklabels=True)
        self.graph_axes.axis["right"].major_ticklabels.set_axis_direction(
            "bottom" if self.extend_angle else "left"
        )
        self.graph_axes.axis["bottom"].toggle(ticklabels=False, label=False)


# TODO - add kwargs for reference point
# TODO - add kwargs for reference line


def demo():

    # reference point
    ref_point = 10.0

    # correlation range
    corr_range = (0, 50)

    # model comparison points
    model_stats = [
        (0.9, 3),  # high correlation, low variance
        (0.1, 10),  # low correlation, similar variance
        (0.6, 11),  # ok correlation, similar variance
    ]

    # =======================
    # Init Figure
    # =======================
    fig = plt.figure(figsize=(8, 8))

    # add taylor diagram to plot
    taylor_fig = TaylorDiagram(
        ref_point=ref_point,
        fig=fig,
        subplot=111,
        extend_angle=False,
        corr_range=corr_range,
    )

    # plot reference point
    taylor_fig.add_reference_point(
        ref_point, color="black", marker=".", markersize=20, label="Model I"
    )

    # plot reference line
    taylor_fig.add_reference_line(ref_point, color="black", linestyle="--", label="_")

    # add grid
    taylor_fig.add_grid()

    # add contours
    taylor_fig.add_contours(ref_point, levels=3, colors="gray")
    taylor_fig.polar_axes.clabel(taylor_fig.contours, inline=1, fontsize=20, fmt="%.1f")
    plt.show()

    pass


if __name__ == "__main__":
    demo()
