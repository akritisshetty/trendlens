import os
import numpy as np

# CLIP Embeddings
npy_one_file_path = "trendlens_outputs/embeddings.npy"

if os.path.exists(npy_one_file_path):
    data = np.load(npy_one_file_path)
    np.savetxt("trendlens_outputs/embeddings.csv", data, delimiter=",")

# UMAP 2d
npy_two_file_path = "trendlens_outputs/umap_2d.npy"

if os.path.exists(npy_two_file_path):
    data = np.load(npy_two_file_path)
    np.savetxt("trendlens_outputs/umap_2d.csv", data, delimiter=",")

# UMAP 10d    
npy_three_file_path = "trendlens_outputs/umap_10d.npy"

if os.path.exists(npy_three_file_path):
    data = np.load(npy_three_file_path)
    np.savetxt("trendlens_outputs/umap_10d.csv", data, delimiter=",")
