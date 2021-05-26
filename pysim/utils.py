import itertools
from typing import Callable, Dict, Iterable, List, Union


def dict_product(dicts: Dict) -> List[Dict]:
    """Returns the product of a dictionary with lists
    Parameters
    ----------
    dicts : Dict,
        a dictionary where each key has a list of inputs
    Returns
    -------
    prod : List[Dict]
        the list of dictionary products
    Example
    -------
    >>> parameters = {
        "samples": [100, 1_000, 10_000],
        "dimensions": [2, 3, 10, 100, 1_000]
        }
    >>> parameters = list(dict_product(parameters))
    >>> parameters
    [{'samples': 100, 'dimensions': 2},
    {'samples': 100, 'dimensions': 3},
    {'samples': 1000, 'dimensions': 2},
    {'samples': 1000, 'dimensions': 3},
    {'samples': 10000, 'dimensions': 2},
    {'samples': 10000, 'dimensions': 3}]
    """
    return list(dict(zip(dicts.keys(), x)) for x in itertools.product(*dicts.values()))
