# trendlens

## Build Instructions (TrendLens Pipeline)

Run the following scripts in order:

```bash
python generate_metadata.py        # Generate metadata for each post
python generate_embeddings.py      # Generate CLIP embeddings for images + metadata
python generate_umap.py            # Perform dimensionality reduction (UMAP)
python convert_npy_to_csv.py       # Convert .npy files to .csv for readability
```

## Output

All outputs are stored in trendlens_outputs/:
```bash
trendlens_outputs/
├── embeddings.csv
├── embeddings.npy
├── embeddings_checkpoint.npy
├── metadata.csv
├── metadata_checkpoint.csv
├── smpd_metadata.json
├── umap_2d.npy
├── umap_10d.npy
├── umap_2d.csv
├── umap_10d.csv
└── umap_scatter.png
```
