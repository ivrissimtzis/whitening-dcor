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
# Experiment settings
# ==================================================

labels = list("abcdefghijklmnopqr")

p = len(labels)

n_samples = 5000

n_seeds = 20

experiment_name = "".join(labels)

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
# Main loop
# ==================================================

for seed in range(n_seeds):

    print()
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
    # Mixed data
    # ------------------------------------------

    print("  Mixed")

    D = distance_correlation_matrix(X)

    for i in range(p):
        for j in range(i + 1, p):

            rows.append({

                "seed": seed,
                "method": "Mixed",

                "pair":
                    f"{labels[i]}-{labels[j]}",

                "dist1": labels[i],
                "dist2": labels[j],

                "dcor": D[i, j]

            })

    # ------------------------------------------
    # Whitening methods
    # ------------------------------------------

    for method in methods[1:]:

        print(f"  {method}")

        Z, W, Sigma = whiten(
            X,
            method=method
        )

        D = distance_correlation_matrix(Z)

        for i in range(p):
            for j in range(i + 1, p):

                rows.append({

                    "seed": seed,
                    "method": method,

                    "pair":
                        f"{labels[i]}-{labels[j]}",

                    "dist1": labels[i],
                    "dist2": labels[j],

                    "dcor": D[i, j]

                })


# ==================================================
# Save long-format results
# ==================================================

df = pd.DataFrame(rows)

df.to_csv(
    f"results/{experiment_name}_allpairs.csv",
    index=False
)

print()
print("Saved:")
print(
    f"results/{experiment_name}_allpairs.csv"
)


# ==================================================
# Summary by method
# ==================================================

summary = df.groupby("method").agg(

    mean_dcor=("dcor", "mean"),

    sd_dcor=("dcor", "std"),

    p95_dcor=(
        "dcor",
        lambda x:
        np.percentile(x, 95)
    ),

    max_dcor=("dcor", "max")

)

summary.to_csv(
    f"results/{experiment_name}_summary.csv"
)

print()
print(summary.sort_values(
    "mean_dcor"
))

print()
print("Saved:")
print(
    f"results/{experiment_name}_summary.csv"
)
