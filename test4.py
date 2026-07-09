import numpy as np
import pandas as pd

from collections import Counter

from bach18 import sample_sources
from whitening import whiten
from dependence import distance_correlation


# ==================================================
# Distance-correlation utilities
# ==================================================

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


def top_offdiag_pairs(
        D,
        labels,
        k=10):

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


# ==================================================
# Experiment settings
# ==================================================

labels = list("ghi")

# Examples:
# labels = list("ghi")
# labels = list("jkl")
# labels = list("mno")
# labels = list("pqr")

p = len(labels)

n_samples = 1000

n_seeds = 20

experiment_name = "".join(labels)

methods = [
    "ZCA",
    "PCA",
    "Cholesky",
    "ZCA-cor",
    "PCA-cor",
]


# ==================================================
# Storage
# ==================================================

all_results = []

pair_counter = {
    method: Counter()
    for method in methods
}


# ==================================================
# Main loop
# ==================================================

for seed in range(n_seeds):

    print()
    print(f"Seed {seed}")

    rng = np.random.default_rng(seed)

    # --------------------------------------
    # Generate sources
    # --------------------------------------

    S = sample_sources(
        labels,
        n_samples,
        rng=rng
    )

    # standardise

    S -= S.mean(
        axis=1,
        keepdims=True
    )

    S /= S.std(
        axis=1,
        keepdims=True
    )

    # --------------------------------------
    # Random Gaussian mixing
    # --------------------------------------

    while True:

        A = rng.normal(
            size=(p, p)
        )

        if np.linalg.cond(A) < 20:
            break

    X = A @ S

    # --------------------------------------
    # Whitening methods
    # --------------------------------------

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

        top_pairs = top_offdiag_pairs(
            D,
            labels,
            k=10
        )

        for value, a, b in top_pairs:

            pair_counter[method][(a, b)] += 1


# ==================================================
# Summary statistics
# ==================================================

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


# ==================================================
# Save outputs
# ==================================================

df.to_csv(
    f"results/{experiment_name}_results.csv",
    index=False
)

summary.to_csv(
    f"results/{experiment_name}_summary.csv"
)


# ==================================================
# Save pair frequencies
# ==================================================

for method in methods:

    filename = (
        f"results/"
        f"{experiment_name}_{method}_pairs.txt"
    )

    with open(filename, "w") as f:

        f.write(f"{method}\n")
        f.write("=" * 40)
        f.write("\n\n")

        for pair, count in (
            pair_counter[method]
            .most_common(50)
        ):

            line = f"{pair}: {count}\n"

            f.write(line)

            print(line.strip())


print()
print("Results saved in results folder.")
