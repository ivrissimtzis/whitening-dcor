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

labels = list("ghi")

p = len(labels)

n_samples = 1000

n_seeds = 100

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
    # Random Gaussian mixing
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

        print(f"  {method}")

        Z, W, Sigma = whiten(
            X,
            method=method
        )

        D = distance_correlation_matrix(Z)

        gh = D[0, 1]
        gi = D[0, 2]
        hi = D[1, 2]

        all_results.append({
            "seed": seed,
            "method": method,

            "gh": gh,
            "gi": gi,
            "hi": hi,

            "mean_dcor":
                np.mean([gh, gi, hi])
        })


# ==================================================
# Data frame
# ==================================================

df = pd.DataFrame(all_results)

print()
print("Detailed results")
print("----------------")
print(df.head())


# ==================================================
# Summary
# ==================================================

summary = df.groupby("method").agg(

    mean_gh=("gh", "mean"),
    sd_gh=("gh", "std"),

    mean_gi=("gi", "mean"),
    sd_gi=("gi", "std"),

    mean_hi=("hi", "mean"),
    sd_hi=("hi", "std"),

    mean_mean_dcor=("mean_dcor", "mean"),
    sd_mean_dcor=("mean_dcor", "std")

)

print()
print("Summary")
print("-------")
print(summary)


# ==================================================
# Rankings for each pair
# ==================================================

print()
print("Ranking for g-h")
print("---------------")

print(
    summary.sort_values("mean_gh")[
        ["mean_gh", "sd_gh"]
    ]
)

print()
print("Ranking for g-i")
print("---------------")

print(
    summary.sort_values("mean_gi")[
        ["mean_gi", "sd_gi"]
    ]
)

print()
print("Ranking for h-i")
print("---------------")

print(
    summary.sort_values("mean_hi")[
        ["mean_hi", "sd_hi"]
    ]
)

print()
print("Ranking for overall mean")
print("------------------------")

print(
    summary.sort_values("mean_mean_dcor")[
        ["mean_mean_dcor",
         "sd_mean_dcor"]
    ]
)


# ==================================================
# Save results
# ==================================================

df.to_csv(
    f"results/{experiment_name}_results.csv",
    index=False
)

summary.to_csv(
    f"results/{experiment_name}_summary.csv"
)

print()
print("Results saved:")
print(
    f"results/{experiment_name}_results.csv"
)
print(
    f"results/{experiment_name}_summary.csv"
)
