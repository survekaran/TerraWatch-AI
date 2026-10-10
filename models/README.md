# Models Directory

TerraWatch uses pure baseline analysis:
- Root-Mean-Square (RMS) band differencing with adaptive (Otsu / percentile / fixed) thresholding
- Spectral index differencing (NDVI, NDBI, MNDWI, NDWI, ExG) for multispectral imagery

No learned weights or neural network checkpoints are required. All analysis operates transparently, deterministically, and 100% offline.
