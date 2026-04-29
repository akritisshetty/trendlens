import os
import numpy as np

npy_file_path = "embeddings.npy"

if os.path.exists(npy_file_path):
    data = np.load(npy_file_path)
    np.savetxt("trendlens_outputs/embeddings.csv", data, delimiter=",")
