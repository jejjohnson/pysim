from scipy import stats
from pysim.information import histogram_entropy


def test_histogram_entropy_correction():

    X = stats.gamma(a=10).rvs(1_000, random_state=123)

    x_entropy = histogram_entropy(X)

    assert x_entropy == 2.52771628


def test_histogram_entropy():

    X = stats.gamma(a=10).rvs(1_000, random_state=123)

    x_entropy = histogram_entropy(X, correction=False)

    assert x_entropy == 2.5311

