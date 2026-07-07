import numpy as np


def make_pos_diag(U):
    """
    Flip eigenvector signs so that diagonal
    entries are positive.
    """
    signs = np.sign(np.diag(U))
    signs[signs == 0] = 1.0

    return U * signs



def whitening_matrix(Sigma, method="ZCA"):
    """
    Compute whitening matrix from covariance matrix.

    Parameters
    ----------
    Sigma : ndarray (p,p)

    method : str
        One of:
            'ZCA'
            'PCA'
            'Cholesky'
            'ZCA-cor'
            'PCA-cor'

    Returns
    -------
    W : ndarray (p,p)
    """

    method = method.upper()

    v = np.diag(Sigma)

    # ------------------
    # ZCA and PCA
    # ------------------

    if method in ("ZCA", "PCA"):

        eigvals, U = np.linalg.eigh(Sigma)

    # ------------------
    # ZCA-cor and PCA-cor
    # ------------------

    if method in ("ZCA-COR", "PCA-COR"):

        Dinv = np.diag(1 / np.sqrt(v))

        R = Dinv @ Sigma @ Dinv

        theta, G = np.linalg.eigh(R)

    # ------------------
    # ZCA
    # ------------------

    if method == "ZCA":

        W = (
            U
            @ np.diag(1 / np.sqrt(eigvals))
            @ U.T
        )

    # ------------------
    # PCA
    # ------------------

    elif method == "PCA":

        U = make_pos_diag(U)

        W = (
            np.diag(1 / np.sqrt(eigvals))
            @ U.T
        )

    # ------------------
    # Cholesky
    # ------------------

    elif method == "CHOLESKY":

        L = np.linalg.cholesky(Sigma)

        W = np.linalg.inv(L)

    # ------------------
    # ZCA-cor
    # ------------------

    elif method == "ZCA-COR":

        W = (
            G
            @ np.diag(1 / np.sqrt(theta))
            @ G.T
            @ np.diag(1 / np.sqrt(v))
        )

    # ------------------
    # PCA-cor
    # ------------------

    elif method == "PCA-COR":

        G = make_pos_diag(G)

        W = (
            np.diag(1 / np.sqrt(theta))
            @ G.T
            @ np.diag(1 / np.sqrt(v))
        )

    else:

        raise ValueError(
            f"Unknown whitening method '{method}'"
        )

    return W



def whiten(X, method="ZCA", center=True):
    """
    Whiten data matrix.

    Parameters
    ----------
    X : ndarray (p,n)
        Variables in rows, observations in columns.

    method : str
        Whitening method.

    center : bool
        Remove row means before whitening.

    Returns
    -------
    Z : ndarray
        Whitened data.

    W : ndarray
        Whitening matrix.

    Sigma : ndarray
        Covariance matrix before whitening.
    """

    if center:
        X = X - X.mean(axis=1, keepdims=True)

    Sigma = np.cov(X)

    W = whitening_matrix(
        Sigma,
        method=method
    )

    Z = W @ X

    return Z, W, Sigma
