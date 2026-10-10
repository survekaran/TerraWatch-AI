# Dataset setup guide

## Running without any dataset
Generate the built-in **synthetic** pair and run on it:

    python scripts/generate_synthetic_pair.py --out data/sample

This writes `before.tif`, `after.tif`, `reference_mask.tif` and `synthetic_pair_meta.json`.

**Synthetic data is not real satellite imagery.** It has arbitrary coordinates, flat-coloured
buildings and Gaussian noise. It proves the plumbing (areas, CRS, exports) against exact ground
truth. It says nothing about accuracy on real scenes, clouds, shadows, seasons or sensors.

## Real input requirements
Two GeoTIFFs that already:
- have a CRS and a valid affine transform,
- share the same pixel grid (same size, pixel size and origin) and the same band count,
- cover the same area.

Phase 2 includes an automated sub-pixel alignment gate and optional auto-correction.
Inputs must already share the same pixel grid and CRS; if your files differ in grid or CRS,
prepare them first, for example:

    gdalwarp -t_srs EPSG:32643 -tr 0.5 0.5 -te <xmin ymin xmax ymax> -r bilinear in.tif out.tif

Put prepared files in `data/input/` (git-ignored) and select them in the app or pass paths to the
pipeline. Preferred demo data: a real before/after pair from OpenAerialMap or Sentinel-2 imagery.

## Benchmark and test datasets
- **OSCD** (Sentinel-2): for multispectral coarse mode with band maps.
- **Sample pairs**: Place GeoTIFF pairs with projected metric CRS in `data/input/`.

