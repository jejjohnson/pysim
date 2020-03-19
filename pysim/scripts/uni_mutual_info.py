import sys

sys.path.insert(0, "/Users/eman/Documents/code_projects/pysim")

import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt


plt.style.use(["seaborn-paper"])

from pysim.information.mutual import univariate_mutual_info


seed = 123
np.random.seed(seed)

n_samples = 2_000
mu1, mu2 = 0.5, 0.1
scale1, scale2 = 0.9, 0.1


# initialize data distribution
data_dist1 = stats.norm(loc=mu1, scale=scale1)
data_dist2 = stats.norm(loc=mu2, scale=scale2)

# get some samples
X1_samples = data_dist1.rvs(size=n_samples)[:, None]
X2_samples = data_dist2.rvs(size=n_samples)[:, None]
X_samples = np.hstack([X1_samples, X2_samples])

assert X_samples.shape[1] == 2

sns.jointplot(X_samples[:, 0], X_samples[:, 1])
plt.show()


# ===========================
# Mutual Info - True
# ===========================
method = "knn"

H_true = 0.5 * np.log(1 + scale1 ** 2 / scale2 ** 2)

print(f"Entropy (True): {H_true:.4f}")

# ===========================
# Mutual Info - KNN
# ===========================
method = "knn"

H_knn = univariate_mutual_info(X1_samples, X2_samples, method=method)

print(f"Entropy (kNN): {H_knn:.4f}")

# ===========================
# Mutual Info - Gauss
# ===========================
method = "gauss"

H_gauss = univariate_mutual_info(X1_samples, X2_samples, method=method)

print(f"Entropy (Gauss): {H_gauss:.4f}")
