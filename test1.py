import numpy as np

from bach18 import sample_sources
from whitening import whiten
from mixing import sim_ortho
from dependence import distance_correlation


def distance_correlation_matrix(X):
    """
    Pairwise distance correlation matrix.

    X shape = (variables, samples)
    """

    p = X.shape[0]

    D = np.zeros((p, p))

    for i in range(p):
        for j in range(i, p):

            d = distance_correlation(
                X[i, :],
                X[j, :]
            )

            D[i, j] = d
            D[j, i] = d

    return D


def mean_offdiag(M):
    """
    Mean off-diagonal value.
    """

    p = M.shape[0]

    return (
        np.sum(M) - np.trace(M)
    ) / (p * (p - 1))


def max_offdiag(M):

    mask = ~np.eye(M.shape[0], dtype=bool)

    return np.max(M[mask])




# --------------------------------------------------
# Generate Bach sources
# --------------------------------------------------

SEED = 123

rng = np.random.default_rng(SEED)

S = sample_sources(
    list("abcdefghijklmnopqr"),
    1000,
    rng=rng
)

# Standardize sources

S -= S.mean(axis=1, keepdims=True)
S /= S.std(axis=1, keepdims=True)

# --------------------------------------------------
# Mix
# --------------------------------------------------

A = rng.normal(size=(18,18))

X = A @ S

# --------------------------------------------------
# Baseline dependence
# --------------------------------------------------

D_source = distance_correlation_matrix(S)
D_mixed = distance_correlation_matrix(X)

print()
print("Mean off-diagonal dCor")
print("----------------------")
print(f"Sources : {mean_offdiag(D_source):.6f}")
print(f"Mixed   : {mean_offdiag(D_mixed):.6f}")

print()
print("Mean off-diagonal dCor")
print("----------------------")
print(f"Sources : {max_offdiag(D_source):.6f}")
print(f"Mixed   : {max_offdiag(D_mixed):.6f}")

# --------------------------------------------------
# Whitening comparison
# --------------------------------------------------

methods = [
    "ZCA",
    "PCA",
    "Cholesky",
    "ZCA-cor",
    "PCA-cor",
]

resultsMean = []
resultsMax = []

for method in methods:

    print(f"Processing {method}")

    Z, W, Sigma = whiten(
        X,
        method=method
    )

    D = distance_correlation_matrix(Z)

    scoreMean = mean_offdiag(D)
    scoreMax  = max_offdiag(D)

    resultsMean.append(
        (method, scoreMean)
    )

    resultsMax.append(
        (method, scoreMax)
    )

resultsMean.sort(key=lambda x: x[1])
resultsMax.sort(key=lambda x: x[1])

print()
print("Whitening ranking Mean")
print("----------------------")

for method, score in resultsMean:
    print(f"{method:10s} {score:.6f}")

print()
print("Whitening ranking Max")
print("---------------------")

for method, score in resultsMax:
    print(f"{method:10s} {score:.6f}")
