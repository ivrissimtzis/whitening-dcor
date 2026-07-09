import numpy as np
import pandas as pd

from bach18 import sample_sources
from whitening import whiten
from dependence import distance_correlation


# ==================================================
# Distance correlation matrix
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


# ==================================================
# Experiment settings
# ==================================================

labels = list("bdd")

# labels = list("jkl")
# labels = list("mno")
# labels = list("pqr")

p = len(labels)

if p != 3:
    raise ValueError(
        "This version is intended for exactly 3 distributions."
    )

n_samples = 1000

n_seeds = 100

experiment_name = "".join(labels)

methods = [
    "ZCA",
    "PCA",
    "Cholesky",
    "ZCA-cor",
    "PCA-cor"
]


# ==================================================
# Storage
# ==================================================

all_results = []


# ==================================================
# Main loop
# ==================================================

for seed in range(n_seeds):

    print(f"Seed {seed}")

    rng = np.random.default_rng(seed)

    # ------------------------------------------
    # Generate sources
    # ------------------------------------------

    S = sample_sources(
        labels,
        n_samples,
        rng=rng
    )

    S -= S.mean(
        axis=1,
        keepdims=True
    )

    S /= S.std(
        axis=1,
        keepdims=True
    )

    # ------------------------------------------
    # Gaussian mixing
    # ------------------------------------------

    while True:

        A = rng.normal(
            size=(p, p)
        )

        if np.linalg.cond(A) < 20:
            break

    X = A @ S

    # ------------------------------------------
    # Whitening methods
    # ------------------------------------------

    for method in methods:

        Z, W, Sigma = whiten(
            X,
            method=method
        )

        D = distance_correlation_matrix(Z)

        d01 = D[0, 1]
        d02 = D[0, 2]
        d12 = D[1, 2]

        all_results.append({

            "seed": seed,
            "method": method,

            "pair_01": d01,
            "pair_02": d02,
            "pair_12": d12,

            "mean_dcor":
                np.mean([d01, d02, d12])

        })


# ==================================================
# Long-format dataframe
# ==================================================

df = pd.DataFrame(all_results)

df.to_csv(
    f"results/{experiment_name}_results.csv",
    index=False
)


# ==================================================
# Wide-format pair files
# ==================================================

pair_names = [
    ("pair_01", f"{labels[0]}{labels[1]}"),
    ("pair_02", f"{labels[0]}{labels[2]}"),
    ("pair_12", f"{labels[1]}{labels[2]}")
]

for pair_col, pair_label in pair_names:

    wide = df.pivot(
        index="seed",
        columns="method",
        values=pair_col
    )

    wide.to_csv(
        f"results/{experiment_name}_{pair_label}.csv"
    )

    print()
    print(f"Pair {pair_label}")
    print("----------------")

    summary = wide.agg(
        ["mean", "std"]
    ).T

    print(
        summary.sort_values("mean")
    )

    summary.to_csv(
        f"results/{experiment_name}_{pair_label}_summary.csv"
    )


# ==================================================
# Overall mean dCor
# ==================================================

wide_mean = df.pivot(
    index="seed",
    columns="method",
    values="mean_dcor"
)

wide_mean.to_csv(
    f"results/{experiment_name}_mean.csv"
)

summary_mean = (
    wide_mean
    .agg(["mean", "std"])
    .T
)

print()
print("Overall mean dCor")
print("-----------------")

print(
    summary_mean.sort_values("mean")
)

summary_mean.to_csv(
    f"results/{experiment_name}_mean_summary.csv"
)

print()
print("Results saved in results/")
