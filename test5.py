import numpy as np
import pandas as pd

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


def offdiag_values(M):

    mask = ~np.eye(
        M.shape[0],
        dtype=bool
    )

    return M[mask]


def mean_offdiag(M):

    return np.mean(
        offdiag_values(M)
    )


def std_offdiag(M):

    return np.std(
        offdiag_values(M)
    )


def percentile_offdiag(M, q=95):

    return np.percentile(
        offdiag_values(M),
        q
    )


def max_offdiag(M):

    return np.max(
        offdiag_values(M)
    )


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

        vals = offdiag_values(D)

        all_results.append({

            "seed": seed,
            "method": method,

            "mean_dcor":
                np.mean(vals),

            "sd_dcor":
                np.std(vals),

            "p95_dcor":
                np.percentile(vals, 95),

            "max_dcor":
                np.max(vals)

        })


# ==================================================
# Detailed results dataframe
# ==================================================

df = pd.DataFrame(all_results)

print()
print("Detailed results")
print("----------------")
print(df.head())


# ==================================================
# Summary statistics
# ==================================================

summary = df.groupby("method").agg(

    mean_mean_dcor=("mean_dcor", "mean"),
    sd_mean_dcor=("mean_dcor", "std"),

    mean_sd_dcor=("sd_dcor", "mean"),
    sd_sd_dcor=("sd_dcor", "std"),

    mean_p95_dcor=("p95_dcor", "mean"),
    sd_p95_dcor=("p95_dcor", "std"),

    mean_max_dcor=("max_dcor", "mean"),
    sd_max_dcor=("max_dcor", "std")

)

print()
print("Summary")
print("-------")
print(summary)


# ==================================================
# Rankings
# ==================================================

print()
print("Ranking by mean dCor")
print("--------------------")
print(
    summary.sort_values(
        "mean_mean_dcor"
    )
)

print()
print("Ranking by SD of dCor")
print("---------------------")
print(
    summary.sort_values(
        "mean_sd_dcor"
    )
)

print()
print("Ranking by 95th percentile")
print("--------------------------")
print(
    summary.sort_values(
        "mean_p95_dcor"
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

print()
print("Results saved.")
print(
    f"results/{experiment_name}_results.csv"
)
print(
    f"results/{experiment_name}_summary.csv"
)
