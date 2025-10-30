
import os
import time
import xarray as xr

from lightgbm import early_stopping
from lightgbm import log_evaluation

from noclouds import ssrf

DATA_DIR = '../tests/data'

def _dev():
    """SSRF development only."""

    nc_cloud = os.path.join(DATA_DIR, '2018-02-21.nc')
    nc_clear = os.path.join(DATA_DIR, '2018-02-11.nc')

    ds_cloud = xr.open_dataset(
        nc_cloud,
        mask_and_scale=False,
        decode_coords='all',
    ).drop_vars('spatial_ref')

    ds_clear = xr.open_dataset(
        nc_clear,
        mask_and_scale=False,
        decode_coords='all',
    ).drop_vars('spatial_ref')

    nodata = -999
    n_total_samples = 2000000
    rand_seed = 40

    params = {
        'n_estimators': 500,
        'boosting_type': 'rf',
        'num_leaves': 32,
        'learning_rate': 0.01,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'feature_fraction': 0.8,
        'random_state': rand_seed
    }

    cbs = [
        early_stopping(10),
        log_evaluation(5)
    ]

    s = time.time()

    da_out = ssrf.run(
        da_ref=ds_clear.to_array(),  # inputs, features, predictors
        da_tar=ds_cloud.to_array(),  # target
        nodata=nodata,
        n_total_samples=n_total_samples,
        percent_train=0.9,
        predict_inplace=True,
        rand_seed=rand_seed,
        params=params,
        callbacks=cbs
    )

    da_out[[0, 1, 2]].rio.to_raster('ssrf_output.tif')

    e = time.time()
    print(f'Processing time: {e - s}')


if __name__ == '__main__':
    _dev()