import sys

sys.path.insert(0, "/Users/eman/Documents/code_projects/pysim")

import numpy as np
from scipy import stats
from pysim.information.entropy import multivariate_entropy
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use(["seaborn-paper"])

seed = 123
n_samples = 1_000
np.random.seed(seed)

n_features = 5
mean = np.random.rand(n_features)
cov = np.random.rand(n_features)

# initialize data distribution
data_dist = stats.multivariate_normal(mean=mean, cov=cov, seed=123)

# get some samples
X_samples = data_dist.rvs(size=n_samples)

# ===========================
# True Entropy
# ===========================
H_true = data_dist.entropy()

print(f"Entropy (True): {H_true:.4f}")

# ===========================
# KNN Entropy
# ===========================
method = "knn"
n_neighbors = 5
n_jobs = 1
algorithm = "brute"

X_knn_entropy = multivariate_entropy(
    X_samples,
    method=method,
    n_neighbors=n_neighbors,
    n_jobs=n_jobs,
    algorithm=algorithm,
)

print(f"Entropy (KNN): {X_knn_entropy:.4f}")

# ===========================
# Gaussian Assumption
# ===========================
method = "gaussian"

X_gaus_entropy = multivariate_entropy(X_samples, method=method)

print(f"Entropy (Gauss): {X_gaus_entropy:.4f}")
