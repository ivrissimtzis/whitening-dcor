from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import json


from bach18 import sample
from bach18 import sample_sources
from whitening import whiten
from mixing import sim_ortho
from dependence import distance_correlation

# Create results directory if necessary
Path("results").mkdir(exist_ok=True)


# Generate random seed 
SEED = 123
rng = np.random.default_rng(SEED)



# Generate all 18 source distributions
S = sample_sources(
    list("abcdefghijklmnopqr"),
    10000,
    rng=rng,
    standardize = True
)

metadata = {
    "seed": 123,
    "n_samples": 10000,
    "distributions": list("abcdefghijklmnopqr"),
    "standardized": True
}

with open("results/metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print(S.shape)



# Plot all sampled distributions
fig, axes = plt.subplots(
    3,
    6,
    figsize=(18, 9)
)

S = sample_sources(
    list("abcdefghijklmnopqr"),
    5000,
    rng=rng
)

for ax, (d, x) in zip(
        axes.ravel(),
        zip("abcdefghijklmnopqr", S)
):
    ax.hist(x, bins=50, density=True)
    ax.set_title(d)

plt.tight_layout()


# Save figure
plt.savefig("results/histograms.png")

# Save all sources
np.save(
    "results/sources.npy",
    S
)

plt.show()



# Whitening

from whitening import whiten

for method in [
    "ZCA",
    "PCA",
    "Cholesky",
    "ZCA-cor",
    "PCA-cor"
]:

    Z, W, Sigma = whiten(
        S,
        method=method
    )

    C = np.cov(Z)

    err = np.max(
        np.abs(
            C - np.eye(C.shape[0])
        )
    )

    print(
        f"{method:10s}: {err:.3e}"
    )


# Mixing 

from mixing import sim_ortho

A = sim_ortho(18, rng=rng)

print(A.shape)

print(
    np.max(
        np.abs(
            A.T @ A - np.eye(18)
        )
    )
)


rng = np.random.default_rng(123)

x = sample("a", 1000, rng)
y = sample("b", 1000, rng)



# dcor expects observations in rows
dcor_val = distance_correlation(
    x.reshape(-1, 1),
    y.reshape(-1, 1)
)

print(dcor_val)

x = sample("a", 1000, rng)

print(
    distance_correlation(x, x)
)


x = sample("a", 1000, rng)
y = x**2

print(distance_correlation(x, y))


def distance_correlation_matrix(S):

    p = S.shape[0]

    D = np.zeros((p, p))

    for i in range(p):
        for j in range(i, p):

            d = distance_correlation(
                S[i, :],
                S[j, :]
            )

            D[i, j] = d
            D[j, i] = d

    return D


D = distance_correlation_matrix(S)

print(D.shape)

print(np.round(D, 3))
