
import os
import time
import pytest
import xarray as xr

from noclouds.ssrf import lazy as ssrf_lz

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

def test_ssrf_lazy_run_full():

    data_dir = r'../../tests/data'
    nc_cloud = os.path.join(data_dir, '2018-02-21.nc')
    nc_clear = os.path.join(data_dir, '2018-02-11.nc')

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
    n_samples = 2000000

    xgb_params = {
        'objective': 'reg:squarederror',
        'tree_method': 'hist',
        'learning_rate': 0.1,
        'max_depth': 8,
        'num_boost_round': 500,
        'percent_train': 0.9,
        'split_seed': None,
        'early_stopping_rounds': 10,
        'verbose_eval': 5,
        'device': 'gpu'
    }

    s = time.time()

    da_out = ssrf_lz.run(
        da_ref=ds_clear.to_array(),  # inputs, features, predictors
        da_tar=ds_cloud.to_array(),  # target
        nodata=nodata,
        n_samples=n_samples,
        rand_seed=0,
        predict_inplace=True,
        xgb_params=xgb_params
    )

    e = time.time()
    print(f'Processing time: {e - s}')