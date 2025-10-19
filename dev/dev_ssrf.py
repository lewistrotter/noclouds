
import os
import time
import xarray as xr

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
    n_samples = 2000000

    # xgb_params = {
    #     'objective': 'reg:squarederror',
    #     'tree_method': 'hist',
    #     'learning_rate': 0.1,
    #     'max_depth': 8,
    #     'num_boost_round': 500,
    #     'percent_train': 0.9,
    #     'split_seed': None,
    #     'early_stopping_rounds': 10,
    #     'verbose_eval': 5,
    #     'device': 'gpu'
    # }

    params = {
        'n_estimators': 500,
        'boosting_type': 'rf',
        'num_leaves': 32,
        'learning_rate': 0.01,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'feature_fraction': 0.8,
        'random_state': 0
    }

    s = time.time()

    da_out = ssrf.run(
        da_ref=ds_clear.to_array(),  # inputs, features, predictors
        da_tar=ds_cloud.to_array(),  # target
        nodata=nodata,
        n_total_samples=n_samples,
        percent_train=0.9,
        early_stopping_rounds=10,
        log_evaluation_periods=5,
        predict_inplace=True,
        rand_seed=0,
        params=params
    )

    e = time.time()
    print(f'Processing time: {e - s}')


if __name__ == '__main__':
    _dev()