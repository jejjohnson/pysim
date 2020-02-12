import argparse

import matplotlib.pyplot as plt
import numpy as np

from taylor import TaylorDiagram

plt.style.use(["seaborn-talk"])



def demo_simple() -> None:

    # reference point
    ref_point = 9.0

    # correlation range
    ref_range = (0, 50)

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
        ref_range=ref_range,
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

    # reference point range
    ref_range = (0, 50)  # 0% less as min, 50% more as max

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
    parser = argparse.ArgumentParser(description="Taylor Diagram demo")
    parser.add_argument(
        "--type", type=int, default=1, metavar="T", help="Which demo to run."
    )

    args = parser.parse_args()

    if args.type == 1:
        demo_simple()
    elif args.type == 2:
        demo_scatter()
    else:
        raise ValueError(f"Unrecognized demo: {args.type}")
