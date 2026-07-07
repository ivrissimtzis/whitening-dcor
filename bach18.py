
import numpy as np


def sample(name, n, rng=None):
    """
    Sample from one of the Bach-Jordan source distributions.

    Parameters
    ----------
    name : str
        Distribution label ('a' ... 'r')
    n : int
        Number of samples
    rng : np.random.Generator, optional

    Returns
    -------
    ndarray of shape (n,)
    """
    if rng is None:
        rng = np.random.default_rng()

    name = name.lower()

    # ---------- elementary distributions ----------

    if name == "a":
        return rng.standard_t(df=3, size=n)

    elif name == "b":
        return rng.laplace(
            loc=0.0,
            scale=1 / np.sqrt(2),
            size=n,
        )

    elif name == "c":
        return rng.uniform(
            low=-np.sqrt(3),
            high=np.sqrt(3),
            size=n,
        )

    elif name == "d":
        return rng.standard_t(df=5, size=n)

    elif name == "e":
        return rng.exponential(size=n) - 1.0

    # ---------- mixture of Laplace ----------

    elif name == "f":

        weights = np.array([0.5, 0.5])
        means = np.array([-1.0, 1.0])
        covs = np.array([0.5, 0.5])

        idx = rng.choice(2, size=n, p=weights)

        return (
            rng.laplace(
                loc=0.0,
                scale=covs[idx] / np.sqrt(2),
                size=n
            )
            + means[idx]
        )

    # ---------- Gaussian mixtures ----------

    params = {

        "g": {
            "weights": [0.5, 0.5],
            "means": [-0.5, 0.5],
            "sigmas": [0.15, 0.15],
        },

        "h": {
            "weights": [0.5, 0.5],
            "means": [-0.5, 0.5],
            "sigmas": [0.4, 0.4],
        },

        "i": {
            "weights": [0.5, 0.5],
            "means": [-0.5, 0.5],
            "sigmas": [0.5, 0.5],
        },

        "j": {
            "weights": [1/4, 3/4],
            "means": [-0.5, 0.5],
            "sigmas": [0.15, 0.15],
        },

        "k": {
            "weights": [1/3, 2/3],
            "means": [-0.7, 0.5],
            "sigmas": [0.4, 0.4],
        },

        "l": {
            "weights": [1/3, 2/3],
            "means": [-0.7, 0.5],
            "sigmas": [0.5, 0.5],
        },

        "m": {
            "weights": [1/6, 2/6, 2/6, 1/6],
            "means": [-1, -0.33, 0.33, 1],
            "sigmas": [0.16, 0.16, 0.16, 0.16],
        },

        "n": {
            "weights": [1/6, 2/6, 2/6, 1/6],
            "means": [-1, -0.2, 0.2, 1],
            "sigmas": [0.2, 0.3, 0.3, 0.2],
        },

        "o": {
            "weights": [1/6, 2/6, 2/6, 1/6],
            "means": [-0.7, -0.2, 0.2, 0.7],
            "sigmas": [0.2, 0.3, 0.3, 0.2],
        },

        "p": {
            "weights": [1/5, 1/5, 2/5, 1/5],
            "means": [-1, 0.3, -0.3, 1.1],
            "sigmas": [0.2, 0.2, 0.2, 0.2],
        },

        "q": {
            "weights": np.array([1, 3, 2, 0.5]) / 6.5,
            "means": [-1, -0.2, 0.3, 1],
            "sigmas": [0.2, 0.3, 0.2, 0.2],
        },

        "r": {
            "weights": [1/6, 2/6, 2/6, 1/6],
            "means": [-0.8, -0.2, 0.2, 0.5],
            "sigmas": [0.22, 0.3, 0.3, 0.2],
        },
    }

    if name not in params:
        raise ValueError(f"Unknown distribution '{name}'")

    p = params[name]

    idx = rng.choice(
        len(p["weights"]),
        size=n,
        p=p["weights"]
    )

    means = np.asarray(p["means"])
    sigmas = np.asarray(p["sigmas"])

    return rng.normal(
        loc=means[idx],
        scale=sigmas[idx],
        size=n
    )



def sample_sources(
        names,
        n,
        rng=None,
        standardize=False):

    if rng is None:
        rng = np.random.default_rng()

    S = np.vstack([
        sample(name, n, rng)
        for name in names
    ])

    if standardize:
        S -= S.mean(axis=1, keepdims=True)
        S /= S.std(axis=1, keepdims=True)

    return S
