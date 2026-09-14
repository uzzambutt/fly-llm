import polars as pl
import numpy as np
from scipy.sparse import csr_matrix, save_npz
import os

input_file = "connections_princeton.csv.gz"
if not os.path.exists(input_file):
    raise FileNotFoundError(f"Cannot find {input_file} in {os.getcwd()}.")

print("[1/4] Reading and aggregating connections with Polars...")
df = (
    pl.read_csv(
        input_file,
        columns=["pre_root_id", "post_root_id", "syn_count"],
        schema_overrides={
            "pre_root_id": pl.UInt64,
            "post_root_id": pl.UInt64,
            "syn_count": pl.UInt32
        }
    )
    .group_by(["pre_root_id", "post_root_id"])
    .agg(pl.col("syn_count").sum())
)

pre_ids = df["pre_root_id"].to_numpy()
post_ids = df["post_root_id"].to_numpy()
syn_counts = df["syn_count"].to_numpy()

print("[2/4] Mapping neuron IDs to contiguous 0..N indices...")
unique_ids = np.unique(np.concatenate([pre_ids, post_ids]))
num_neurons = len(unique_ids)
print(f"Total functional neurons: {num_neurons:,}")

id_to_idx = {uid: idx for idx, uid in enumerate(unique_ids)}

print("[3/4] Indexing sparse arrays...")
src_idx = np.vectorize(id_to_idx.get, otypes=[np.int32])(pre_ids)
dst_idx = np.vectorize(id_to_idx.get, otypes=[np.int32])(post_ids)

# Scale raw synapse weights to control spectral radius
weights = (syn_counts.astype(np.float32)) / 50.0

print("[4/4] Writing Scipy CSR sparse matrix...")
adj_matrix = csr_matrix((weights, (dst_idx, src_idx)), shape=(num_neurons, num_neurons), dtype=np.float32)

output_file = "fly_connectome_csr.npz"
save_npz(output_file, adj_matrix)
print(f"Saved: {output_file} ({os.path.getsize(output_file) / (1024**2):.2f} MB)")