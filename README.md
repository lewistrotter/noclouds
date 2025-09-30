# noclouds

**noclouds** is a lightweight Python package for fast and scalable detection, removal, and filling of anomalous pixels in satellite imagery. It implements proven, peer-reviewed methods and leverages Xarray, Numba, and Dask for ease of use, speed, and scalability.

## Supported methods

### Cloud-filling
| Method | Citation | Reference | Target | Output |
|--------|----------|-----------|--------|--------|
| SSRF (Spatial-Spectral Random Forest) | Wang et al. (2022) | img_a.png | img_b.png | img_c.png |
