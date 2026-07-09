import pandas as pd


# ==========================================
# Read long-format results
# ==========================================

csv_file = "results/abcdefghijklmnopqr_allpairs.csv"

df = pd.read_csv(csv_file)


# ==========================================
# Average over seeds
# ==========================================

mean_df = (
    df.groupby(["method", "pair"])["dcor"]
      .mean()
      .reset_index()
)


# ==========================================
# Create 6 x 153 matrix
# ==========================================

table = mean_df.pivot(
    index="method",
    columns="pair",
    values="dcor"
)


# ==========================================
# Order methods
# ==========================================

method_order = [
    "Mixed",
    "ZCA",
    "PCA",
    "Cholesky",
    "ZCA-cor",
    "PCA-cor"
]

table = table.reindex(method_order)


# ==========================================
# Save
# ==========================================

outfile = "results/abcdefghijklmnopqr_pairs_matrix.csv"

table.to_csv(outfile)


# ==========================================
# Compute and save improvement
# ==========================================

improvement = (
    table.loc["Mixed"]
    .subtract(table)
)

improvement = improvement.drop(
    index="Mixed"
)

outfile = "results/abcdefghijklmnopqr_improvement_matrix.csv"
improvement.to_csv(outfile)


# ==========================================
# Print summary
# ==========================================

print()
print("Matrix shape:")
print(table.shape)

print()
print("First few columns:")
print(table.iloc[:, :10])

print()
print(f"Saved to: {outfile}")
