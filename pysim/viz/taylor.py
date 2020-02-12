from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
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
    
    Information
    -----------
    Author: J. Emmanuel Johnson
    Date: 10-02-2020

    References
    ----------
    Original Implementation:
        - Yannick Copin
        - https://gist.github.com/ycopin/3342888

    Modified Implementation:
        - StackOverFlow Question
        - https://codereview.stackexchange.com/questions/82919/modified-taylor-diagrams
    """

    def __init__(
        self,
        ref_point: float,
        fig: Optional[plt.figure] = None,
        subplot: Optional[int] = 111,
        extend_angle: bool = False,
        corr_labels: Optional[np.ndarray] = None,
        ref_range: Tuple[float, float] = (0, 10),
        ref_label: str = "Reference Point",
        angle_label: str = "Correlation",
        var_label: str = "Standard Deviation",
    ) -> None:

        self.angle_label = angle_label
        self.ref_label = ref_label
        self.var_label = var_label
        self.extend_angle = extend_angle

        # correlation labels
        if corr_labels is None:
            corr_labels = np.array([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0])

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
        self.smin = ref_range[0] * ref_point
        self.smax = (ref_range[1] / 100) * ref_point + ref_point

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

        # create graphical axes

        ax = floating_axes.FloatingSubplot(fig, subplot, grid_helper=ghelper)
        fig.add_subplot(ax)
        self.graph_axes = ax
        self.polar_axes = ax.get_aux_axes(PolarAxes.PolarTransform())

        self.sample_points = []
        # Setup Axes
        self.reset_axes()

    def add_reference_point(self, ref_point: float, *args, **kwargs) -> None:
        line = self.polar_axes.plot([0], ref_point, *args, **kwargs)

        self.sample_points.append(line[0])

        return None

    def add_reference_line(self, ref_point: float, *args, **kwargs) -> None:

        t = np.linspace(0, self.tmax)
        r = np.zeros_like(t) + ref_point
        self.polar_axes.plot(t, r, *args, **kwargs)

        return None

    def add_point(self, var_point: float, corr_point: float, *args, **kwargs) -> None:

        # add sample to plot
        line = self.polar_axes.plot(np.arccos(corr_point), var_point, *args, **kwargs)

        # add line to sample points
        self.sample_points.append(line[0])

        return None

    def add_scatter(
        self, var_points: np.ndarray, corr_points: np.ndarray, *args, **kwargs
    ) -> None:

        pts = self.polar_axes.scatter(
            np.arccos(corr_points), var_points, *args, **kwargs
        )

        self.sample_points.append(pts)
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

        self.contours = self.polar_axes.contour(ts, rs, dist, levels=levels, **kwargs)
        return None

    def add_legend(self, fig, *args, **kwargs):

        fig.legend(
            self.sample_points,
            [p.get_label() for p in self.sample_points],
            *args,
            **kwargs
        )
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


def demo() -> None:

    # reference point
    ref_point = 9.0

    # correlation range
    corr_range = (0, 50)

    # model comparison points
    model_stats = [
        (0.9, 3),  # high correlation, low variance
        (0.1, 9),  # low correlation, similar variance
        (0.6, 11),  # ok correlation, similar variance
    ]

    # =======================
    # Init Figure
    # =======================
    fig = plt.figure(figsize=(10, 10))

    # add taylor diagram to plot
    taylor_fig = TaylorDiagram(
        ref_point=ref_point,
        fig=fig,
        subplot=111,
        extend_angle=False,
        corr_range=corr_range,
    )
    # ========================
    # plot reference point
    # ========================
    taylor_fig.add_reference_point(
        ref_point,
        color="black",
        marker=".",
        markersize=20,
        linestyle="",
        label="Reference Data",
    )

    # ========================
    # plot reference line
    # ========================
    taylor_fig.add_reference_line(ref_point, color="black", linestyle="--", label="_")

    # ========================
    # add grid
    # ========================
    taylor_fig.add_grid()

    # ========================
    # add contours
    # ========================
    taylor_fig.add_contours(ref_point, levels=3, colors="gray")

    # modify contour labels
    taylor_fig.polar_axes.clabel(taylor_fig.contours, inline=1, fontsize=20, fmt="%.1f")

    # ========================
    # add samples
    # ========================
    colors = ["red", "green", "blue"]
    names = ["Model I", "Model II", "Model III"]

    for i, isample in enumerate(model_stats):
        taylor_fig.add_point(
            isample[1],
            isample[0],
            color=colors[i],
            markersize=20,
            linestyle="",
            marker=".",
            label=names[i],
            zorder=3,
        )

    # ========================
    # add samples
    # ========================
    taylor_fig.add_legend(fig, numpoints=1, prop=dict(size="large"), loc="upper right")
    # show plot
    plt.show()

    return None


def demo_scatter() -> None:

    # ====================
    # Reference Data
    # ====================
    ref_point = 9.0

    # correlation range
    corr_range = (0, 50)

    # model comparison points

    # =========================
    # Model Comparison Points
    # =========================

    # SAMPLE I
    np.random.seed(123)
    n_samples = 10
    corr_points = np.random.rand(n_samples)
    var_points = 2 * np.random.random(n_samples) + 5
    param_points = np.random.rand(n_samples)

    # SAMPLE II
    np.random.seed(111)
    n_samples = 10
    corr_points2 = np.random.rand(n_samples)
    var_points2 = 2 * np.random.random(n_samples) + 1
    param_points2 = np.random.rand(n_samples)

    # =======================
    # Init Figure
    # =======================
    fig = plt.figure(figsize=(10, 10))

    # add taylor diagram to plot
    taylor_fig = TaylorDiagram(
        ref_point=ref_point,
        fig=fig,
        subplot=111,
        extend_angle=False,
        corr_range=corr_range,
    )
    # ========================
    # plot reference point
    # ========================
    taylor_fig.add_reference_point(
        ref_point,
        color="black",
        marker=".",
        markersize=20,
        linestyle="",
        label="Reference Data",
    )

    # ========================
    # plot reference line
    # ========================
    taylor_fig.add_reference_line(ref_point, color="black", linestyle="--", label="_")

    # ========================
    # add grid
    # ========================
    taylor_fig.add_grid()

    # ========================
    # add contours
    # ========================
    taylor_fig.add_contours(ref_point, levels=3, colors="gray")

    # modify contour labels
    taylor_fig.polar_axes.clabel(taylor_fig.contours, inline=1, fontsize=20, fmt="%.1f")

    # ========================
    # add samples
    # ========================
    colors = ["red", "green", "blue"]
    names = ["Model I", "Model II", "Model III"]
    import matplotlib

    boundaries = (0.0, 1.0)
    norm = matplotlib.colors.Normalize(vmin=boundaries[0], vmax=boundaries[1])

    cm = plt.cm.get_cmap("RdYlBu")

    # Samples I
    taylor_fig.add_scatter(
        var_points,
        corr_points,
        c=param_points,
        cmap=cm,
        norm=norm,
        s=20,
        marker="*",
        label="Model I",
        zorder=3,
    )

    taylor_fig.add_scatter(
        var_points2,
        corr_points2,
        c=param_points2,
        cmap=cm,
        norm=norm,
        s=20,
        marker="x",
        label="Model II",
        zorder=3,
    )

    # ========================
    # add colorbar
    # ========================
    # Normalize

    cbar = plt.colorbar(
        taylor_fig.sample_points[1], fraction=0.046, extend="both", norm=norm
    )
    cbar.set_label("Parameter", rotation=270, fontsize=20, labelpad=20)

    # ========================
    # add legend
    # ========================
    taylor_fig.add_legend(fig, numpoints=1, prop=dict(size="large"), loc="upper right")
    # show plot
    plt.show()

    return None


if __name__ == "__main__":
    demo_scatter()
