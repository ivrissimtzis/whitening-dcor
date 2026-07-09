import numpy as np
import pandas as pd

from bach18 import sample_sources
from whitening import whiten
from dependence import distance_correlation


# ==================================================
# Pairwise dCor matrix
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
# Summary statistics from dCor matrix
# ==================================================

def dcor_summary(D):

    iu = np.triu_indices_from(D, k=1)

    vals = D[iu]

    return {
        "mean_dcor": np.mean(vals),
        "sd_dcor": np.std(vals, ddof=1),
        "p95_dcor": np.percentile(vals, 95),
        "max_dcor": np.max(vals),
    }


# ==================================================
# Experiment settings
# ==================================================

distributions = list("abcdefghijklmnopqr")

p = 10               # number of sources

n_samples = 1000

n_seeds = 100

methods = [
    "Mixed",
    "ZCA",
    "PCA",
    "Cholesky",
    "ZCA-cor",
    "PCA-cor"
]


# ==================================================
# Storage
# ==================================================

rows = []


# ==================================================
# Main experiment
# ==================================================

for label in distributions:

    print()
    print("=" * 60)
    print(f"Distribution {label}")
    print("=" * 60)

    for seed in range(n_seeds):

        print(f"  Seed {seed}")

        rng = np.random.default_rng(seed)

        # ------------------------------------------
        # Generate 10 independent copies
        # of the SAME distribution
        # ------------------------------------------

        S = np.vstack([
            sample_sources(
                [label],
                n_samples,
                rng=rng
            )[0]
            for _ in range(p)
        ])

        # standardize each source

        S -= S.mean(
            axis=1,
            keepdims=True
        )

        S /= S.std(
            axis=1,
            keepdims=True
        )

        # ------------------------------------------
        # Random mixing
        # ------------------------------------------

        while True:

            A = rng.normal(
                size=(p, p)
            )

            if np.linalg.cond(A) < 20:
                break

        X = A @ S

        # ------------------------------------------
        # Mixed
        # ------------------------------------------

        D = distance_correlation_matrix(X)

        stats = dcor_summary(D)

        rows.append({

            "distribution": label,
            "seed": seed,
            "method": "Mixed",

            **stats

        })

        # ------------------------------------------
        # Whitening methods
        # ------------------------------------------

        for method in methods[1:]:

            Z, W, Sigma = whiten(
                X,
                method=method
            )

            D = distance_correlation_matrix(Z)

            stats = dcor_summary(D)

            rows.append({

                "distribution": label,
                "seed": seed,
                "method": method,

                **stats

            })


# ==================================================
# Save detailed results
# ==================================================

df = pd.DataFrame(rows)

df.to_csv(
    "results/single_distribution_whitening.csv",
    index=False
)


# ==================================================
# Summary across seeds
# ==================================================

summary = (
    df.groupby(
        ["distribution", "method"]
    )
    .agg(
        mean_mean_dcor=("mean_dcor", "mean"),
        sd_mean_dcor=("mean_dcor", "std"),

        mean_p95_dcor=("p95_dcor", "mean"),
        mean_max_dcor=("max_dcor", "mean")
    )
)

summary.to_csv(
    "results/single_distribution_summary.csv"
)


# ==================================================
# Wide table:
# distributions × methods
# ==================================================

wide = (
    df.groupby(
        ["distribution", "method"]
    )["mean_dcor"]
    .mean()
    .unstack()
)

wide.to_csv(
    "results/single_distribution_matrix.csv"
)


# ==================================================
# Improvement table
# ==================================================

improvement = (
    wide["Mixed"].values.reshape(-1, 1)
    - wide
)

improvement.columns = wide.columns

improvement.to_csv(
    "results/single_distribution_improvement.csv"
)


# ==================================================
# Print
# ==================================================

print()
print("Average dCor matrix")
print("-------------------")
print(wide)

print()
print("Saved:")

print("results/single_distribution_whitening.csv")
print("results/single_distribution_summary.csv")
print("results/single_distribution_matrix.csv")
print("results/single_distribution_improvement.csv")
