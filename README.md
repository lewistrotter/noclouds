# noclouds

**noclouds** is a lightweight Python package for fast and scalable detection, removal, and filling of anomalous pixels in satellite imagery. It implements proven, peer-reviewed methods and leverages Xarray, Numba, and Dask for ease of use, speed, and scalability.

## Supported methods

### Gap-filling
<table style="width:100%;">
  <tr>
    <th>Method</th>
    <th>Citation</th>
    <th>Numba</th>
    <th>Dask</th>
    <th>Example</th>
  </tr>
  <tr>
    <td style="width:1%;"><p align="left">SSRF (Spatial-Spectral Random Forest)</p></td>
    <td><p align="left">Wang et al. (2022)</p></td>
    <td><p align="center">✅</p></td>
    <td><p align="center">✅</p></td>
    <td><img src="docs/images/ssrf.png" width="100%"></td>
  </tr>
  <tr>
    <td style="width:1%;"><p align="left">Moving Window Regression (MWR)</p></td>
    <td><p align="left">Brooks et al. (2018)</p></td>
    <td><p align="center">✅</p></td>
    <td><p align="center">✅</p></td>
    <td><img src="docs/images/mwr.png" width="100%"></td>
  </tr>
</table>



### Anomoly detection
<table style="width:100%;">
  <tr>
    <th>Method</th>
    <th>Citation</th>
    <th>Numba</th>
    <th>Dask</th>
    <th>Example</th>
  </tr>
  <tr>
    <td style="width:1%;"><p align="left">Median Filter Spike Removal</p></td>
    <td><p align="left">Eklundh and Jönsson (2017)</p></td>
    <td><p align="center">✅</p></td>
    <td><p align="center">✅</p></td>
    <td><img src="docs/images/mfsr.png" width="100%"></td>
  </tr>
</table>


