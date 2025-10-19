
import os
import time
import xarray as xr

from dask.distributed import Client

from noclouds.ssrf import lazy as ssrf_lz

DATA_DIR = '../tests/data'

def _dev():
    """SSRF lazy development only."""

    client = Client(n_workers=4, threads_per_worker=1)
    #client = None

    nc_cloud = os.path.join(DATA_DIR, '2018-02-21.nc')
    nc_clear = os.path.join(DATA_DIR, '2018-02-11.nc')

    ds_cloud = xr.open_dataset(
        nc_cloud,
        mask_and_scale=False,
        decode_coords='all',
        chunks={}
    ).drop_vars('spatial_ref').to_array()

    ds_clear = xr.open_dataset(
        nc_clear,
        mask_and_scale=False,
        decode_coords='all',
        chunks={}
    ).drop_vars('spatial_ref').to_array()

    #ds_cloud = ds_cloud.chunk({'variable': -1})
    #ds_clear = ds_clear.chunk({'variable': -1})

    nodata = -999
    n_total_samples = 2000000

    xgb_params = {
        'objective': 'reg:squarederror',
        'tree_method': 'hist',
        'learning_rate': 0.1,
        'max_depth': 8,
        'num_boost_round': 1000,
        'percent_train': 0.9,
        'split_seed': None,
        'early_stopping_rounds': 10,
        'verbose_eval': 5,
        'device': 'cpu'
    }

    s = time.time()

    # da_out = ssrf_lz.run(
    #     da_ref=ds_clear,  # inputs, features, predictors
    #     da_tar=ds_cloud,  # target
    #     nodata=nodata,
    #     n_samples=n_samples,
    #     rand_seed=0,
    #     allow_persist=True,
    #     xgb_params=xgb_params,
    #     client=client
    # )

    # ssrf_lz.extract_train_set(
    #     da_ref=ds_clear,  # inputs, features, predictors
    #     da_tar=ds_cloud,  # target
    #     nodata=nodata,
    #     n_total_samples=n_total_samples,
    #     oversample_factor=1.5,
    #     rand_seed=0,
    # )

    ssrf_lz.run(
        da_ref=ds_clear,  # inputs, features, predictors
        da_tar=ds_cloud,  # target
        nodata=nodata,
        n_total_samples=n_total_samples,
        oversample_factor=1.5,
        percent_train=0.9,
        rand_seed=0,
        xgb_params=xgb_params
    )



    e = time.time()
    print(f'Processing time: {e - s}')

if __name__ == '__main__':
    _dev()