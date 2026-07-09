import numpy as np
import pandas as pd

from bach18 import sample_sources
from whitening import whiten
from dependence import distance_correlation


# ==================================================
# Utilities
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


def mean_pairwise_dcor(X):

    D = distance_correlation_matrix(X)

    iu = np.triu_indices_from(
        D,
        k=1
    )

    return np.mean(D[iu])


def random_orthogonal_matrix(
        p,
        rng):

    A = rng.normal(
        size=(p, p)
    )

    Q, R = np.linalg.qr(A)

    signs = np.sign(
        np.diag(R)
    )

    signs[signs == 0] = 1

    Q = Q @ np.diag(signs)

    return Q


# ==================================================
# Experiment settings
# ==================================================

distribution = "e"

method = "ZCA"

p = 10

n_samples = 2500

n_seeds = 20

n_rotations = 100


# ==================================================
# Storage
# ==================================================

summary_rows = []

rotation_rows = []


# ==================================================
# Main experiment
# ==================================================

for seed in range(n_seeds):

    print()
    print(
        f"Seed {seed}"
    )

    rng = np.random.default_rng(seed)

    # ------------------------------------------
    # Generate sources
    # ------------------------------------------

    S = np.vstack([

        sample_sources(
            [distribution],
            n_samples,
            rng=rng
        )[0]

        for _ in range(p)

    ])

    # standardise

    S -= S.mean(
        axis=1,
        keepdims=True
    )

    S /= S.std(
        axis=1,
        keepdims=True
    )

    # ------------------------------------------
    # Mixing
    # ------------------------------------------

    while True:

        A = rng.normal(
            size=(p, p)
        )

        if np.linalg.cond(A) < 20:
            break

    X = A @ S

    # ------------------------------------------
    # Whitening
    # ------------------------------------------

    Z, W, Sigma = whiten(
        X,
        method=method
    )

    base_dcor = mean_pairwise_dcor(Z)

    random_dcors = []

    # ------------------------------------------
    # Random rotations
    # ------------------------------------------

    for k in range(n_rotations):

        Q = random_orthogonal_matrix(
            p,
            rng
        )

        Z_rot = Q @ Z

        dcor = mean_pairwise_dcor(
            Z_rot
        )

        random_dcors.append(dcor)

        rotation_rows.append({

            "distribution":
                distribution,

            "method":
                method,

            "seed":
                seed,

            "rotation":
                k,

            "base_dcor":
                base_dcor,

            "rotation_dcor":
                dcor,

            "difference":
                dcor - base_dcor

        })

    random_dcors = np.asarray(
        random_dcors
    )

    summary_rows.append({

        "distribution":
            distribution,

        "method":
            method,

        "seed":
            seed,

        "base_dcor":
            base_dcor,

        "mean_random_dcor":
            np.mean(random_dcors),

        "sd_random_dcor":
            np.std(
                random_dcors,
                ddof=1
            ),

        "min_random_dcor":
            np.min(random_dcors),

        "max_random_dcor":
            np.max(random_dcors),

        "fraction_better":
            np.mean(
                random_dcors
                < base_dcor
            )

    })


# ==================================================
# Save detailed rotations
# ==================================================

rot_df = pd.DataFrame(
    rotation_rows
)

rot_df.to_csv(
    f"results/{distribution}_{method}_rotations.csv",
    index=False
)


# ==================================================
# Save per-seed summaries
# ==================================================

summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    f"results/{distribution}_{method}_summary.csv",
    index=False
)


# ==================================================
# Overall summary
# ==================================================

overall = summary_df.agg({

    "base_dcor":
        ["mean", "std"],

    "mean_random_dcor":
        ["mean", "std"],

    "fraction_better":
        ["mean", "std"]

})

overall.to_csv(

    f"results/"
    f"{distribution}_{method}_overall.csv"

)

print()
print("Overall summary")
print("----------------")
print(overall)

print()
print("Saved:")

print(
    f"results/{distribution}_{method}_rotations.csv"
)

print(
    f"results/{distribution}_{method}_summary.csv"
)

print(
    f"results/{distribution}_{method}_overall.csv"
)
