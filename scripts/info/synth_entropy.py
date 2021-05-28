import sys, os
from pyprojroot import here

# spyder up to find the root
root = here(project_files=[".here"])

import numpy as np

# MATPLOTLIB Settings
import matplotlib as mpl
import matplotlib.pyplot as plt


# SEABORN SETTINGS
import seaborn as sns

sns.set_context(context="poster", font_scale=0.7)

# Logging
import wandb
from wandb.sdk import wandb_config
from argparse import ArgumentParser

# =========================================
# ARGUMENTS
# =========================================
parser = ArgumentParser(description="Entropy Experiments with Synthetic Data")
parser.add_argument("--wandb-entity", type=str, default="ipl_uv")
parser.add_argument("--wandb-project", type=str, default="rbig4it_ent")
parser.add_argument(
    "-sm",
    "--smoke-test",
    action="store_true",
    help="to do a smoke test without logging",
)
parser.add_argument(
    "--demo", action="store_true", help="to do a simple demo only",
)
# Dataset
parser.add_argument("--dataset", type=str, default="gaussian")
parser.add_argument("--degree_freedom", type=int, default=5)
parser.add_argument("--n_base_samples", type=int, default=500_000)
parser.add_argument("--bins", type=str, default="auto")

# Entropy Estimators
parser.add_argument("--n_neighbors_nbs", type=int, default=10)
parser.add_argument("--n_neighbors_eps", type=int, default=3)

# Experiment
parser.add_argument("--n_trials", type=int, default=20)


# ========================================
# LOGGING
# ========================================
args = parser.parse_args()
# change this so we don't bug wandb with our BS
if args.smoke_test:
    os.environ["WANDB_MODE"] = "dryrun"
    args.n_trials = 1
    args.n_samples = 1_000


wandb_logger = wandb.init(project=args.wandb_project, entity=args.wandb_entity)
wandb_logger.config.update(args)
config = wandb_logger.config

# =========================================
# DATASET
# =========================================
from functools import partial


def generate_dataset(dataset: str, n_samples: int, n_features: int, seed: int) -> tuple:

    if dataset == "gaussian":

        from pysim.data.information.gaussian import generate_gaussian_data

        res_tuple = generate_gaussian_data(
            n_samples=n_samples,
            n_features=n_features,
            seed=seed,
            n_base_samples=config.n_base_samples,
        )

    elif dataset == "gaussian_rotate":
        from pysim.data.information.gaussian import generate_gaussian_rotation_data

        # create seed (trial number)
        res_tuple = generate_gaussian_rotation_data(
            n_samples=n_samples,
            n_features=n_features,
            seed=seed,
            n_base_samples=config.n_base_samples,
        )
    elif dataset == "uniform_rotate":
        from pysim.data.information.linear import generate_linear_entropy_data
        from pysim.information.entropy import marginal_entropy
        from pysim.information.histogram import hist_entropy

        # create marginal entropy estimator
        f = partial(marginal_entropy, estimator=hist_entropy, bins=config.bins)

        res_tuple = generate_linear_entropy_data(
            n_samples=n_samples,
            n_features=n_features,
            seed=seed,
            marg_h_estimator=f,
            estimator_name="histogram",
            n_base_samples=config.n_base_samples,
        )
    elif dataset == "student_t":
        from pysim.data.information.studentt import generate_studentt_data

        # create seed (trial number)
        res_tuple = generate_studentt_data(
            n_samples=n_samples,
            n_features=n_features,
            df=config.degree_freedom,
            seed=seed,
            n_base_samples=config.n_base_samples,
        )

    elif dataset == "cauchy":
        from pysim.data.information.studentt import generate_studentt_data

        # create seed (trial number)
        res_tuple = generate_studentt_data(
            n_samples=n_samples,
            n_features=n_features,
            df=1,
            seed=seed,
            n_base_samples=config.n_base_samples,
        )
    else:
        raise ValueError(f"Unrecognized dataset: {dataset}")

    return res_tuple


# DEMO PLot
demo_tuple = generate_dataset(
    dataset=config.dataset, n_samples=50_000, n_features=10, seed=123
)
import corner

fig = corner.corner(demo_tuple.X[:, :10])
wandb.log({"original_data": wandb.Image(plt)})

# =========================================
# ALGORITHMS
# =========================================
from pysim.information.gaussian import gauss_entropy_multi
from pysim.information.knn import knn_entropy, knn_entropy_npeet
import time


# def get_entropy_estimators(data):

#     t0 = time.time()
#     H_g = gauss_entropy_multi(data)
#     print(f"Gaussian: {H_g:.4f} | Time: {time.time()-t0:.4f} secs")

#     t0 = time.time()
#     H_knn_nbs = knn_entropy(X=data, n_neighbors=10, base=2)
#     print(f"KNN (Neighbours): {H_knn_nbs:.4f} | Time: {time.time()-t0:.4f} secs")

#     t0 = time.time()
#     H_knn_eps = knn_entropy_npeet(X=data, n_neighbors=10, base=2)
#     print(f"KNN (epsilon): {H_knn_eps:.4f} | Time: {time.time()-t0:.4f} secs")


# =========================================
# PARAMS
# =========================================
from pysim.utils import dict_product


if args.smoke_test:
    n_features = [
        2,
        10,
    ]
    n_samples = [500, 750, 1_000]
    params_dict = {
        "n_samples": n_samples,
        "n_features": n_features,
    }
elif args.demo:
    n_features = [
        10,
    ]
    n_samples = [500, 1_000, 5_000, 10_000, 50_000, 100_000]
    params_dict = {
        "n_samples": n_samples,
        "n_features": n_features,
    }
else:
    n_samples = [500, 1_000, 5_000, 10_000, 50_000]
    n_features = [2, 3, 5, 10, 50, 100]
    params_dict = {
        "n_samples": n_samples,
        "n_features": n_features,
    }
params_dict = dict_product(params_dict)


# =========================================
# TRAINING LOOP
# =========================================
from tqdm import tqdm, trange
import pandas as pd
import itertools
import numpy as np


iteration = itertools.count()


all_stats = pd.DataFrame()


with tqdm(params_dict, leave=True) as pbar_params:
    for i_params in pbar_params:

        i_features = i_params["n_features"]
        i_samples = i_params["n_samples"]

        pbar_params.set_description(f"Samples: {i_samples} | Features: {i_features} - ")

        results = {}

        with trange(config.n_trials, leave=False) as pbar_trials:

            for i_trial in pbar_trials:

                pbar_trials.set_description(f"Trials: {i_trial} | Method: GT")

                # generate data
                t0 = time.time()
                truth_data = generate_dataset(
                    dataset=config.dataset,
                    n_samples=i_samples,
                    n_features=i_features,
                    seed=i_trial,
                )
                t1 = time.time() - t0

                results = {
                    "algorithm": "truth",
                    "n_samples": i_samples,
                    "n_features": i_features,
                    "trial": i_trial,
                    "H": truth_data.H,
                    "time": t1,
                }
                wandb.log(results)
                all_stats = pd.concat(
                    [all_stats, pd.DataFrame(results, index=[next(iteration)])], axis=0
                )

                # ======================
                # Gaussian Estimation
                # ======================
                pbar_trials.set_description(f"Trials: {i_trial} | Method: Gaussian")

                t0 = time.time()
                output = gauss_entropy_multi(truth_data.X.copy())
                t1 = time.time() - t0
                results = {
                    "algorithm": "gaussian",
                    "n_samples": i_samples,
                    "n_features": i_features,
                    "trial": i_trial,
                    "H": output,
                    "time": t1,
                }
                wandb.log(results)
                all_stats = pd.concat(
                    [all_stats, pd.DataFrame(results, index=[next(iteration)])], axis=0
                )

                # ==========================
                # KNN estimated (Neighbors)
                # ==========================

                pbar_trials.set_description(
                    f"Trials: {i_trial} | Method: KNN (Neighbors)"
                )

                t0 = time.time()
                output = knn_entropy(
                    X=truth_data.X.copy(),
                    n_neighbors=config.n_neighbors_nbs,
                    base=2,
                    n_jobs=-1,
                )
                t1 = time.time()

                results = {
                    "algorithm": "knn_nbs",
                    "n_samples": i_samples,
                    "n_features": i_features,
                    "trial": i_trial,
                    "H": output,
                    "time": t1,
                }
                wandb.log(results)
                all_stats = pd.concat(
                    [all_stats, pd.DataFrame(results, index=[next(iteration)])], axis=0
                )

                # ==========================
                # KNN estimated (Epsilon)
                # ==========================
                pbar_trials.set_description(
                    f"Trials: {i_trial} | Method: KNN (Epsilon-Ball)"
                )

                t0 = time.time()
                output = knn_entropy_npeet(
                    X=truth_data.X.copy(),
                    n_neighbors=config.n_neighbors_eps,
                    base=2,
                    n_jobs=-1,
                )
                t1 = time.time() - t0

                results = {
                    "algorithm": "knn_eps",
                    "n_samples": i_samples,
                    "n_features": i_features,
                    "trial": i_trial,
                    "H": output,
                    "time": t1,
                }
                wandb.log(results)
                all_stats = pd.concat(
                    [all_stats, pd.DataFrame(results, index=[next(iteration)])], axis=0
                )


# =============================
# RESULTS
# =============================

all_stats_ = all_stats.copy()

all_stats_ = all_stats_.set_index(["n_samples", "n_features", "trial", "algorithm"])

wandb.log({"results": wandb.Table(dataframe=all_stats_)})

# =============================
# Figures
# =============================
stats_ds = all_stats_.to_xarray()


def percent_error(real, pred):
    pred = np.abs(np.abs(pred - real) / real)
    return pred


def get_mean_std(ds):
    mean = ds.mean(["trial"])
    std = 1.96 * ds.std(["trial"])
    return mean, std


for i_feature in tqdm(n_features):

    fig, ax = plt.subplots()

    # ==========================
    # TRUTH
    # ==========================
    x_data = stats_ds.coords["n_samples"].values
    y_mu, y_std = get_mean_std(stats_ds.sel(n_features=i_feature, algorithm="truth").H)

    ub = y_mu + y_std.values.ravel()
    lb = y_mu - y_std.values.ravel()

    ax.plot(x_data, y_mu, linewidth=5, linestyle="-", color="black", label="Truth")
    ax.fill_between(x_data, lb, ub, alpha=0.2, color="black")

    # ==========================
    # GAUSSIAN ESTIMATE
    # ==========================
    y_mu, y_std = get_mean_std(
        stats_ds.sel(n_features=i_feature, algorithm="gaussian").H
    )

    ub = y_mu + y_std.values.ravel()
    lb = y_mu - y_std.values.ravel()

    ax.plot(
        x_data, y_mu, linewidth=5, linestyle="-", color="tab:green", label="Gaussian"
    )
    ax.fill_between(x_data, lb, ub, alpha=0.2, color="tab:green")

    # ==========================
    # KNN (NEIGHBOURS) ESTIMATE
    # ==========================
    y_mu, y_std = get_mean_std(
        stats_ds.sel(n_features=i_feature, algorithm="knn_nbs").H
    )

    ub = y_mu + y_std.values.ravel()
    lb = y_mu - y_std.values.ravel()

    ax.plot(
        x_data, y_mu, linewidth=5, linestyle="-", color="gold", label="kNN (Neighbors)",
    )
    ax.fill_between(x_data, lb, ub, alpha=0.2, color="gold")

    # ==========================
    # KNN (EPSILON) ESTIMATE
    # ==========================
    y_mu, y_std = get_mean_std(
        stats_ds.sel(n_features=i_feature, algorithm="knn_eps").H
    )

    ub = y_mu + y_std.values.ravel()
    lb = y_mu - y_std.values.ravel()

    ax.plot(
        x_data,
        y_mu,
        linewidth=5,
        linestyle="-",
        color="tab:blue",
        label="kNN (Epsilon)",
    )
    ax.fill_between(x_data, lb, ub, alpha=0.2, color="tab:blue")

    # ==========================
    # Extras
    # ==========================
    yval_min = stats_ds.sel(n_features=i_feature).H.min().values
    yval_max = stats_ds.sel(n_features=i_feature).H.max().values
    ax.set(
        xlabel="Number of Samples",
        ylabel="Entropy (nats)",
        title="",
        xscale="log",
        ylim=[yval_min - 0.1 * yval_min, yval_max + 0.2 * yval_max],
    )
    ax.grid(
        True, which="both",
    )
    ax.legend()
    plt.show()
    wandb.log({"entropy": wandb.Image(plt), "n_features": i_feature})
    plt.close(fig)


for i_feature in tqdm(n_features):
    fig, ax = plt.subplots()

    real = stats_ds.sel(n_features=i_feature, algorithm="truth").mean(["trial"]).H

    # stats_ds.sel(n_features=10, algorithm="truth_nats").mean(["trial"]).H.plot(
    #     ax=ax, label="Truth", linewidth=5, color="black"
    # )

    # ==========================
    # GAUSSIAN ESTIMATE
    # ==========================
    mu, std = get_mean_std(stats_ds.sel(n_features=i_feature, algorithm="gaussian").H)

    # calculate percent error
    err_mu = percent_error(real, mu)
    err_std = percent_error(real, std)

    x_data = err_mu.coords["n_samples"].values
    y_mu = err_mu.values
    ub = y_mu + err_std.values.ravel()
    lb = y_mu - err_std.values.ravel()

    ax.plot(
        x_data, y_mu, linewidth=5, linestyle="-", color="tab:green", label="Gaussian"
    )
    ax.fill_between(x_data, lb, ub, alpha=0.2, color="tab:green")

    # ==========================
    # KNN (NEIGHBOURS) ESTIMATE
    # ==========================
    mu, std = get_mean_std(stats_ds.sel(n_features=i_feature, algorithm="knn_nbs").H)

    # calculate percent error
    err_mu = percent_error(real, mu)
    err_std = percent_error(real, std)

    x_data = err_mu.coords["n_samples"].values
    y_mu = err_mu.values
    ub = y_mu + err_std.values.ravel()
    lb = y_mu - err_std.values.ravel()

    ax.plot(
        x_data, y_mu, linewidth=5, linestyle="-", color="gold", label="kNN (Neighbors)",
    )
    ax.fill_between(x_data, lb, ub, alpha=0.2, color="gold")

    # ==========================
    # KNN (EPSILON) ESTIMATE
    # ==========================
    mu, std = get_mean_std(stats_ds.sel(n_features=i_feature, algorithm="knn_eps").H)

    # calculate percent error
    err_mu = percent_error(real, mu)
    err_std = percent_error(real, std)

    x_data = err_mu.coords["n_samples"].values
    y_mu = err_mu.values
    ub = y_mu + err_std.values.ravel()
    lb = y_mu - err_std.values.ravel()

    ax.plot(
        x_data,
        y_mu,
        linewidth=5,
        linestyle="-",
        color="tab:blue",
        label="kNN (Epsilon)",
    )
    ax.fill_between(x_data, lb, ub, alpha=0.2, color="tab:blue")

    ax.set(
        xlabel="Number of Samples",
        ylabel="Percent Error",
        title="",
        xscale="log",
        ylim=[0, 50],
    )
    import matplotlib.ticker as mtick

    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.grid(True, which="both")
    ax.legend()
    plt.show()
    wandb.log({"entropy_mape": wandb.Image(plt), "n_features": i_feature})
    plt.close(fig)