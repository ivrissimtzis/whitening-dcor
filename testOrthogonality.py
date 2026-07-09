import numpy as np
import pandas as pd

from bach18 import sample_sources
from whitening import whiten
from dependence import distance_correlation


# ==================================================
# Pairwise distance-correlation matrix
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
# Mean pairwise dCor
# ==================================================

def mean_pairwise_dcor(X):

    D = distance_correlation_matrix(X)

    iu = np.triu_indices_from(D, k=1)

    return np.mean(D[iu])


# ==================================================
# Random orthogonal matrix
# ==================================================

def random_orthogonal_matrix(p, rng):

    A = rng.normal(size=(p, p))

    Q, R = np.linalg.qr(A)

    # make orientation deterministic

    diag_signs = np.sign(np.diag(R))
    diag_signs[diag_signs == 0] = 1

    Q = Q @ np.diag(diag_signs)

    return Q


# ==================================================
# Experiment settings
# ==================================================

distribution = "e"

p = 10

n_samples = 5000

n_rotations = 100

seed = 123

rng = np.random.default_rng(seed)


# ==================================================
# Generate 10 independent copies
# of the same distribution
# ==================================================

S = np.vstack([

    sample_sources(
        [distribution],
        n_samples,
        rng=rng
    )[0]

    for _ in range(p)

])


# standardize

S -= S.mean(
    axis=1,
    keepdims=True
)

S /= S.std(
    axis=1,
    keepdims=True
)


# ==================================================
# Mixing
# ==================================================

while True:

    A = rng.normal(size=(p, p))

    if np.linalg.cond(A) < 20:
        break

X = A @ S


# ==================================================
# ZCA whitening
# ==================================================

Z, W, Sigma = whiten(
    X,
    method="PCA"
)


# ==================================================
# Reference value
# ==================================================

base_dcor = mean_pairwise_dcor(Z)

print()
print("Original PCA dCor")
print("-----------------")
print(base_dcor)


# ==================================================
# Random rotation experiment
# ==================================================

results = []

for k in range(n_rotations):

    Q = random_orthogonal_matrix(
        p,
        rng
    )

    Z_rot = Q @ Z

    dcor = mean_pairwise_dcor(
        Z_rot
    )

    results.append({
        "rotation": k,
        "mean_dcor": dcor
    })

    print(
        f"{k:3d} : {dcor:.6f}"
    )


# ==================================================
# Save
# ==================================================

df = pd.DataFrame(results)

df.to_csv(
    "results/random_rotation_test.csv",
    index=False
)


# ==================================================
# Summary
# ==================================================

print()
print("Summary")
print("-------")

print(
    df["mean_dcor"].describe()
)

print()
print("Difference from original")

diffs = df["mean_dcor"] - base_dcor

print(
    diffs.describe()
)

print()
print(
    "Maximum absolute difference:"
)

print(
    np.max(np.abs(diffs))
)

print()
print(
    "Saved: results/random_rotation_test.csv"
)
