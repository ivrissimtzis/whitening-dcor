import numpy as np
import pandas as pd

from bach18 import sample_sources
from whitening import whiten
from dependence import distance_correlation


def distance_correlation_matrix(X):

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

    p = M.shape[0]

    return (
        np.sum(M) - np.trace(M)
    ) / (p * (p - 1))


def max_offdiag(M):

    mask = ~np.eye(M.shape[0], dtype=bool)

    return np.max(M[mask])


# -----------------------------------------
# Experiment
# -----------------------------------------

methods = [
    "ZCA",
    "PCA",
    "Cholesky",
    "ZCA-cor",
    "PCA-cor",
]

all_results = []

for seed in range(20):

    print()
    print(f"Seed {seed}")

    rng = np.random.default_rng(seed)

    S = sample_sources(
        list("abcdefghijklmnopqr"),
        1000,
        rng=rng
    )

    # standardize

    S -= S.mean(axis=1, keepdims=True)
    S /= S.std(axis=1, keepdims=True)

    # Gaussian mixing

    while True:

        A = rng.normal(size=(18, 18))

        if np.linalg.cond(A) < 20:
            break

    X = A @ S

    for method in methods:

        print(f"  {method}")

        Z, W, Sigma = whiten(
            X,
            method=method
        )

        D = distance_correlation_matrix(Z)

        all_results.append({
            "seed": seed,
            "method": method,
            "mean_dcor": mean_offdiag(D),
            "max_dcor": max_offdiag(D),
        })

# -----------------------------------------
# Summary
# -----------------------------------------

df = pd.DataFrame(all_results)

print()
print("Mean over seeds")
print("----------------")

summary = df.groupby("method").agg(
    mean_mean_dcor=("mean_dcor", "mean"),
    sd_mean_dcor=("mean_dcor", "std"),
    mean_max_dcor=("max_dcor", "mean"),
    sd_max_dcor=("max_dcor", "std"),
)

print(summary)

print()
print("Ranking by mean dCor")
print("--------------------")

print(
    summary.sort_values(
        "mean_mean_dcor"
    )
)

print()
print("Ranking by max dCor")
print("-------------------")

print(
    summary.sort_values(
        "mean_max_dcor"
    )
)
