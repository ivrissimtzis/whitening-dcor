import numpy as np


def sim_ortho(d, non_neg_diag=False, rng=None):
    """
    Generate a random orthogonal matrix.

    Parameters
    ----------
    d : int
        Matrix dimension.

    non_neg_diag : bool
        If True, force positive diagonal in Q.

    rng : np.random.Generator

    Returns
    -------
    Q : ndarray (d,d)
        Random orthogonal matrix.
    """

    if rng is None:
        rng = np.random.default_rng()

    A = rng.normal(size=(d, d))

    Q, R = np.linalg.qr(A)

    if non_neg_diag:
        sgn = np.sign(np.diag(Q))
    else:
        sgn = np.sign(np.diag(R))

    sgn[sgn == 0] = 1.0

    Q = Q @ np.diag(sgn)

    return Q
