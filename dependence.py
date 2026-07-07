import numpy as np
from scipy.spatial.distance import pdist, squareform


def distance_correlation(x, y):
    """
    Distance correlation between two 1D samples.
    """

    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y).reshape(-1, 1)

    A = squareform(pdist(x))
    B = squareform(pdist(y))

    A -= A.mean(axis=0, keepdims=True)
    A -= A.mean(axis=1, keepdims=True)
    A += A.mean()

    B -= B.mean(axis=0, keepdims=True)
    B -= B.mean(axis=1, keepdims=True)
    B += B.mean()

    dcov2 = np.mean(A * B)
    dvarx2 = np.mean(A * A)
    dvary2 = np.mean(B * B)

    if dvarx2 <= 0 or dvary2 <= 0:
        return 0.0

    return np.sqrt(
        dcov2 /
        np.sqrt(dvarx2 * dvary2)
    )
