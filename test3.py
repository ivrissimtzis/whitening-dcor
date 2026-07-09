import numpy as np
import pandas as pd

from collections import Counter

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


def top_offdiag_pairs(D, labels, k=10):

    pairs = []

    p = D.shape[0]

    for i in range(p):
        for j in range(i + 1, p):

            pairs.append(
                (
                    D[i, j],
                    labels[i],
                    labels[j]
                )
            )

    pairs.sort(reverse=True)

    return pairs[:k]


# ---------------------------------------
# Experiment settings
# ---------------------------------------

labels = list("abcdefghijklmnopqr")

methods = [
    "ZCA",
    "PCA",
    "Cholesky",
    "ZCA-cor",
    "PCA-cor",
]

n_seeds = 20

all_results = []

pair_counter = {
    method: Counter()
    for method in methods
}

# ---------------------------------------
# Main loop
# ---------------------------------------

for seed in range(n_seeds):

    print()
    print(f"Seed {seed}")

    rng = np.random.default_rng(seed)

    S = sample_sources(
        labels,
        1000,
        rng=rng
    )

    # standardize

    S -= S.mean(
        axis=1,
        keepdims=True
    )

    S /= S.std(
        axis=1,
        keepdims=True
    )

    # random Gaussian mixing

    while True:

        A = rng.normal(
            size=(18, 18)
        )

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

        mean_score = mean_offdiag(D)
        max_score = max_offdiag(D)

        all_results.append({
            "seed": seed,
            "method": method,
            "mean_dcor": mean_score,
            "max_dcor": max_score,
        })

        # top-10 most dependent pairs

        top_pairs = top_offdiag_pairs(
            D,
            labels,
            k=10
        )

        for value, a, b in top_pairs:

            pair_counter[method][(a, b)] += 1


# ---------------------------------------
# Summary statistics
# ---------------------------------------

df = pd.DataFrame(all_results)

summary = df.groupby("method").agg(
    mean_mean_dcor=("mean_dcor", "mean"),
    sd_mean_dcor=("mean_dcor", "std"),
    mean_max_dcor=("max_dcor", "mean"),
    sd_max_dcor=("max_dcor", "std"),
)

print()
print("Mean over seeds")
print("----------------")
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

# ---------------------------------------
# Save results
# ---------------------------------------

df.to_csv(
    "results/whitening_dcor_results.csv",
    index=False
)

summary.to_csv(
    "results/whitening_dcor_summary.csv"
)

# ---------------------------------------
# Frequent pairs
# ---------------------------------------

for method in methods:

    print()
    print("=" * 40)
    print(method)
    print("=" * 40)

    for pair, count in (
        pair_counter[method]
        .most_common(20)
    ):

        print(
            f"{pair}: {count}"
        )
