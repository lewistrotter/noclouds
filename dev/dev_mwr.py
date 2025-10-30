
import os
import time
import xarray as xr

from noclouds import mwr

DATA_DIR = '../tests/data'

def _dev():
    """MWR development only."""

    nc = os.path.join(DATA_DIR, 'mwr_ls_ts.nc')

    ds = xr.open_dataset(
        nc,
        mask_and_scale=False,
        decode_coords='all',
    )

    da_mask = (ds.to_array() != -999).any('variable')

    da = ((ds['nbart_nir'] - ds['nbart_red']) /
               (ds['nbart_nir'] + ds['nbart_red']))

    da = da * 10000
    da = da.where(da_mask, -999).squeeze()
    da = da.astype('int16')

    nodata = -999
    temporal_depth = 3 # 2   # 2 * 2 + 1
    space_depth = 3  # 3 * 2 + 1
    min_train_pairs = 2
    min_rsquared = 0.6
    max_value_range = (1, 10000)
    predict_inplace = True
    max_iters = 3

    s = time.time()

    da_out = mwr.run(
        da_ts=da,
        nodata=nodata,
        temporal_depth=temporal_depth,
        space_depth=space_depth,
        min_train_pairs=min_train_pairs,
        min_rsquared=min_rsquared,
        max_value_range=max_value_range,
        predict_inplace=predict_inplace,
        max_iters=max_iters
    )

    e = time.time()
    print(f'Processing time: {e - s}')

    # globals dirs
    # IN_NCS_DIR = r'E:\PMA Unmixing\data\storage\07_apply_masks'
    # IN_NCS_DIR = r'E:\PMA Unmixing\data\storage\08_gap_fill\ncs\iter_1'
    # OUT_NCS_DIR = r'E:\PMA Unmixing\data\storage\08_gap_fill\ncs\iter_2'

    # # global inputs  # iter 1, 2
    # time_size = 11  # 9, 11
    # win_size = 15  # 11, 15
    # m = 3  # 4, 3
    # v_min = 1
    # v_max = 10000
    # r2_min = 0.50  # 0.60, 0.5
    # max_iters = 1000
    #
    # # prepare inputs
    # t_cen = time_size // 2
    # r_cen = win_size // 2

    # # read corrected masked ncs (from 07) as sorted lazy xr
    # ds = ncs_tyx_to_lazy_xr(
    #     IN_NCS_DIR,
    #     dtype='int16',
    #     epsg=32750,
    #     strip_time=True,
    #     chunks={}
    # )
    #
    # # extract dates and pad (reflect 1, 2, 3 to -1, 2, 3, etc.)
    # dts = ds['time'].to_numpy()
    # dts_pad = np.pad(dts, pad_width=t_cen, mode='reflect')
    #
    # for dt_i, dt_v in enumerate(dts):
    #
    #     # create output filename and path
    #     dt = pd.to_datetime(dt_v).strftime('%Y-%m-%d')
    #     fn = f'{dt}.nc'
    #     fp = os.path.join(OUT_NCS_DIR, fn)
    #
    #     # skip prior download
    #     if os.path.exists(fp):
    #         print(f'NetCDF with date {dt} already exists, skipping.')
    #         continue
    #
    #     # select current date + edges
    #     dts_win = dts_pad[dt_i:dt_i + time_size]
    #     ds_sel = ds.sel(time=dts_win)
    #
    #     # set nodata -999 to nan
    #     ds_sel = ds_sel.astype('float32')
    #     ds_sel = ds_sel.where(ds_sel != -999)
    #
    #     # pad dataset for better windowing
    #     ds_sel = ds_sel.pad(
    #         y=(r_cen, r_cen),
    #         x=(r_cen, r_cen),
    #         constant_values=np.nan
    #     )
    #
    #     ds_out = {}
    #     for var in ds_sel.data_vars:
    #         # extract variable and compute
    #         da_var = ds_sel[var].squeeze()
    #         da_var = da_var.compute()
    #         arr_sel = da_var.to_numpy()
    #
    #         s = time.time()
    #
    #         arr_out = apply_winreg(
    #             arr_sel,
    #             t_cen=t_cen,
    #             r_cen=r_cen,
    #             m=m,
    #             v_min=v_min,
    #             v_max=v_max,
    #             r2_min=r2_min,
    #             max_iters=max_iters
    #         )
    #
    #         e = time.time()
    #         print(f'{dt} {var} finished in {e - s}.')
    #
    #         # convert target slice back to xr data array
    #         da_out = xr.DataArray(
    #             arr_out[t_cen, :, :],  # extra bracket to keep dim
    #             dims=('y', 'x'),
    #             coords={'y': da_var['y'], 'x': da_var['x']},
    #             attrs=da_var.attrs
    #         )
    #
    #         # remove padding
    #         da_out = da_out.isel(
    #             y=slice(r_cen, -r_cen),
    #             x=slice(r_cen, -r_cen)
    #         )
    #
    #         # set nan to -999 and cast to free up memory
    #         da_out = da_out.where(da_out.notnull(), -999)
    #         da_out = da_out.astype('int16')
    #
    #         # add to results
    #         ds_out[var] = da_out
    #
    #     # convert to dataset and pute it
    #     ds_out = xr.Dataset(ds_out)
    #
    #     # add time back on
    #     ds_out = ds_out.expand_dims({'time': [dt_v]})
    #
    #     # prepare and apply clamped nodata (true == invalid pix)
    #     da_nd_mask = get_nodata_mask(ds_out, nodata=-999, clamp=True)
    #     ds_out = ds_out.where(~da_nd_mask, -999)
    #
    #     # ensure nodata is properly set
    #     for var in ds_out.data_vars:
    #         ds_out[var] = ds_out[var].rio.write_nodata(-999)
    #
    #     # ensure crs, append
    #     ds_out = ds_out.rio.write_crs(ds_sel.rio.crs)
    #
    #     # build compression params
    #     cmp = {'zlib': True, 'complevel': 5}
    #     enc = {v: cmp for v in ds_out.data_vars}
    #
    #     # export netcdf
    #     ds_out.to_netcdf(fp, encoding=enc)
    #     # #ds_out.rio.to_raster(fp_nc, compress='zstd')


if __name__ == '__main__':
    _dev()