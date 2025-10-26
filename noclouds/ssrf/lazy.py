
import gc
import typing
import numpy as np
import numba as nb
import xarray as xr
import dask

from dask.array.overlap import overlap
from dask.array import map_overlap
from dask.array import map_blocks

from lightgbm.dask import DaskLGBMRegressor

from dask.diagnostics import ProgressBar
from dask.distributed import Client

from .core import _make_train_mask
from .core import _make_predict_mask
from .core import _extract_train_idx
from .core import _extract_predict_idx
from .core import _extract_x
from .core import _extract_y
from .core import _fill_y

from noclouds.utils.helpers import nodata_mask
from noclouds.utils.helpers import default_params


def _get_block_shapes(
        arr: dask.array.Array
) -> list:

    # get (y, x) chunk shapes (row-major)
    y_chunks, x_chunks = arr.chunks
    block_shapes = [
        (y, x)
        for y in y_chunks
        for x in x_chunks
    ]

    return block_shapes


def _guess_train_idx_size_per_block(
        arr: dask.array.Array,
        n_total_samples: int,
        oversample_factor: int | float
) -> np.ndarray:

    # get chunk sizes (row-major order)
    y_sizes, x_sizes = arr.chunks
    block_sizes = np.outer(y_sizes, x_sizes).ravel()

    # get total array size
    array_size = int(np.prod(arr.shape))

    # stratify to chunk sizes and oversample each
    strat_sizes = n_total_samples * (block_sizes / array_size)
    over_strat_sizes = strat_sizes * oversample_factor

    # calc candidate sizes, ensure within block sizes
    estimate_sizes = np.minimum(block_sizes, over_strat_sizes)
    estimate_sizes = np.ceil(estimate_sizes).astype(int)

    return estimate_sizes


def _refine_train_idx_size_per_block(
        arr: np.ndarray,
        n_total_samples: int
) -> np.ndarray:

    if np.any(arr < 0):
        raise ValueError('Input arr must not contain negative values.')

    total_oversample = arr.sum()
    if total_oversample == 0:
        raise ValueError('Cannot refine an all-zero array.')

    if n_total_samples >= total_oversample:
        return arr

    # proportionally rescale oversampled array
    arr_scaled = arr * (n_total_samples / total_oversample)

    # measure how far each value overshoots scaled value
    arr_baseline = np.ceil(arr_scaled)  # ceil prevent undershoot
    arr_remainder = arr_baseline - arr_scaled

    # calc num samples to remove overall to reach requested total
    arr_diff = arr_baseline.sum() - n_total_samples
    arr_diff = int(round(arr_diff))

    # sort idx by overshoot fraction (higher first)
    arr_excess_order = np.argsort(-arr_remainder)

    # correct rounding errors, remove from largest chunks first
    for i in arr_excess_order:
        if arr_diff <= 0:
            break
        if arr_baseline[i] > 0:
            arr_baseline[i] -= 1
            arr_diff -= 1

    arr_refined = arr_baseline.astype(int)

    return arr_refined


def _refine_train_idx(
        arr: np.ndarray,
        n_samples: int,
        rand_seed: int
) -> np.ndarray:

    if n_samples == 0:
        return np.array([], dtype=np.int32)

    if arr.size == 0:
        return np.array([], dtype=np.int32)

    n_valid = arr.size
    n_select = np.min([n_valid, n_samples])

    rng = np.random.default_rng(rand_seed)
    arr_i = rng.choice(
        arr,
        size=n_select,
        replace=False
    ).astype(np.int32)

    return arr_i


def _extract_train_idx_per_block(
        arr_mask: dask.array.Array,
        n_total_samples: int,
        oversample_factor: int | float,
        rand_seed: int
) -> list:

    # per-block estimates of oversizes train samples
    estimate_idx_sizes=_guess_train_idx_size_per_block(
        arr_mask,
        n_total_samples,
        oversample_factor
    )

    # get random idx from oversized estimates per-chunk
    idx_delays = [
        dask.delayed(_extract_train_idx)
        (b, n, rand_seed)
        for b, n in zip(
            arr_mask.to_delayed().ravel(),
            estimate_idx_sizes
        )
    ]

    with ProgressBar():
        idx_blocks = dask.compute(*idx_delays)

    # refine per-chunk estimates of oversizes train samples
    sampled_sizes = np.array([b.size for b in idx_blocks])
    refined_sizes = _refine_train_idx_size_per_block(
        sampled_sizes,
        n_total_samples
    )

    # refine random idx from oversized estimates per-chunk
    refined_idx_blocks = [
        _refine_train_idx(b, n, rand_seed)
        for b, n in zip(idx_blocks, refined_sizes)
    ]

    return refined_idx_blocks


def _extract_predict_idx_per_block(
        arr_mask: dask.array.Array
) -> list:

    idx_delays = [
        dask.delayed(_extract_predict_idx)(b)
        for b in arr_mask.to_delayed().ravel()
    ]

    with ProgressBar():
        idx_blocks = dask.compute(*idx_delays)

    return idx_blocks


def _split_idx_per_block(
        idx_blocks: list,
        percent_train: float,
        rand_seed: int
) -> tuple:

    rng = np.random.default_rng(rand_seed)

    idx_train, idx_eval = [], []
    for arr in idx_blocks:
        n_idx = len(arr)

        idx = np.arange(n_idx)
        rng.shuffle(idx)

        split_i = int(n_idx * percent_train)

        idx_train.append(arr[idx[:split_i]])
        idx_eval.append(arr[idx[split_i:]])

    return idx_train, idx_eval


def _pack_with_delayed_dummies(
        original: list,
        missing: list,
        template: typing.Any
) -> list:

    n_original = len(original)
    n_missing = len(missing)

    if n_original == n_missing:
        return missing

    packed = []

    j = 0
    for arr in original:
        if arr.size == 0:
            packed.append(template)
        else:
            packed.append(missing[j])
            j += 1

    return packed


def extract_train_set(
        da_ref: xr.DataArray,
        da_tar: xr.DataArray,
        nodata: int | float,
        n_total_samples: int = 100000,
        oversample_factor: int | float = 1.5,
        percent_train: float | None = 0.9,
        rand_seed: int = 0
) -> tuple:

    if not isinstance(da_ref, xr.DataArray):
        raise TypeError('Input da_ref must be type xr.DataArray.')
    if not isinstance(da_tar, xr.DataArray):
        raise TypeError('Input da_tar must be type xr.DataArray.')

    if da_ref.ndim not in (2, 3):
        raise ValueError('Input da_ref must be 2D (y, x) or 3D (b, y, x).')
    if da_tar.ndim not in (2, 3):
        raise ValueError('Input da_tar must be 2D (y, x) or 3D (b, y, x).')

    if da_ref.shape[1] != da_tar.shape[1]:
        raise ValueError('Inputs da_ref and da_tar must have same y size.')
    if da_ref.shape[2] != da_tar.shape[2]:
        raise ValueError('Inputs da_ref and da_tar must have same x size.')

    if da_ref.dtype != da_tar.dtype:
        raise TypeError('Inputs da_ref and da_tar must have same dtype.')

    if nodata is None:
        raise ValueError('Input nodata must be provided.')

    arr_ref = da_ref.data
    arr_tar = da_tar.data

    if not isinstance(arr_ref, dask.array.Array):
        raise TypeError('Input da_ref must be backed by a dask.array.Array.')
    if not isinstance(arr_tar, dask.array.Array):
        raise TypeError('Input arr_tar must be backed by a dask.array.Array.')

    # need (b, y, x) arrays
    if arr_ref.ndim == 2:
        arr_ref = np.expand_dims(arr_ref, axis=0)
    if arr_tar.ndim == 2:
        arr_tar = np.expand_dims(arr_tar, axis=0)

    # where (3, 3) x win all true and y true
    arr_mask = map_overlap(
        _make_train_mask,
        nodata_mask(arr_ref, nodata),
        nodata_mask(arr_tar, nodata),
        depth=(1, 1),
        boundary=True,  # nodata == true
        dtype=np.bool_
    )

    if rand_seed is None:
        rand_seed = np.random.randint(10000)

    if n_total_samples <= 0:
        raise ValueError('Input n_total_samples must be > 0.')

    # guess idx size per block, sample, then refine
    idx_blocks = _extract_train_idx_per_block(
        arr_mask,
        n_total_samples,
        oversample_factor,
        rand_seed
    )

    if len(idx_blocks) == 0:
        raise ValueError('No valid training data found.')

    # split idx into train / eval sets if requested
    if percent_train is not None:
        idx_blocks, idx_ev_blocks = _split_idx_per_block(
            idx_blocks,
            percent_train,
            rand_seed
        )

    # padding of (1, 1) so need offset
    offset = 1

    # TODO: reduce repeating code

    # extract train x samples at idx
    arr_x = []
    for var in range(arr_ref.shape[0]):
        x_blocks = overlap(
            arr_ref[var, :, :],
            depth=(1, 1),
            boundary=nodata
        )

        x_delays = [
            dask.array.from_delayed(
                dask.delayed(_extract_x)
                (i, b, offset),
                shape=(i.size, 9),  # always 9 pix win
                dtype=x_blocks.dtype
            )
            for i, b in zip(
                idx_blocks,
                x_blocks.to_delayed().ravel()
            )
        ]

        arr_x.append(np.vstack([*x_delays]))

    arr_x = np.hstack(arr_x)

    # extract train y samples at idx
    arr_y = []
    for var in range(arr_tar.shape[0]):
        y_blocks = overlap(
            arr_tar[var, :, :],
            depth=(1, 1),
            boundary=nodata
        )

        y_delays = [
            dask.array.from_delayed(
                dask.delayed(_extract_y)
                (i, b, offset),
                shape=(i.size, 1),  # always 1 var
                dtype=y_blocks.dtype
            )
            for i, b in zip(
                idx_blocks,
                y_blocks.to_delayed().ravel()
            )
        ]

        arr_y.append(np.vstack([*y_delays]))

    arr_y = np.hstack(arr_y)  # TODO ensure h and vstack ok

    if percent_train is None:
        return arr_x, arr_y, None, None

    # extract eval x samples at idx
    arr_x_ev = []
    for var in range(arr_ref.shape[0]):
        x_ev_blocks = overlap(
            arr_ref[var, :, :],
            depth=(1, 1),
            boundary=nodata
        )

        x_ev_delays = [
            dask.array.from_delayed(
                dask.delayed(_extract_x)
                (i, b, offset),
                shape=(i.size, 9),  # always 9 pix win
                dtype=x_ev_blocks.dtype
            )
            for i, b in zip(
                idx_ev_blocks,
                x_ev_blocks.to_delayed().ravel()
            )
        ]

        arr_x_ev.append(np.vstack([*x_ev_delays]))

    arr_x_ev = np.hstack(arr_x_ev)

    # extract eval y samples at idx
    arr_y_ev = []
    for var in range(arr_tar.shape[0]):
        y_ev_blocks = overlap(
            arr_tar[var, :, :],
            depth=(1, 1),
            boundary=nodata
        )

        y_ev_delays = [
            dask.array.from_delayed(
                dask.delayed(_extract_y)
                (i, b, offset),
                shape=(i.size, 1),  # always 1 var
                dtype=y_ev_blocks.dtype
            )
            for i, b in zip(
                idx_ev_blocks,
                y_ev_blocks.to_delayed().ravel()
            )
        ]

        arr_y_ev.append(np.vstack([*y_ev_delays]))

    arr_y_ev = np.hstack(arr_y_ev)

    return arr_x, arr_y, arr_x_ev, arr_y_ev


def calibrate_models(
        arr_x: dask.array.Array,
        arr_y: dask.array.Array,
        arr_x_ev: np.ndarray | None = None,
        arr_y_ev: np.ndarray | None = None,
        params: dict | None = None,
        callbacks: list | None = None,
        client: Client | None = None
) -> list:
    """TODO"""

    if not isinstance(arr_x, dask.array.Array):
        raise TypeError('Input arr_x must be type dask.array.Array.')
    if not isinstance(arr_y, dask.array.Array):
        raise TypeError('Input arr_y must be type dask.array.Array.')

    if arr_x.ndim != 2:
        raise TypeError('Input arr_x must be 2D (samples, n_vars).')
    if arr_y.ndim != 2:
        raise TypeError('Input arr_y must be 2D (samples, n_vars).')

    if arr_x.shape[0] != arr_y.shape[0]:
        raise ValueError('Inputs arr_x, arr_y must have equal sample sizes.')

    if arr_x.dtype != arr_y.dtype:
        raise TypeError('Inputs arr_x, arr_y must have same dtype.')

    # TODO: check chunk sizes same

    if (arr_x_ev is None) != (arr_y_ev is None):
        raise ValueError('Inputs arr_x_ev, arr_y_ev must both be provided or ignored.')

    if arr_x_ev is not None and arr_y_ev is not None:

        if not isinstance(arr_x_ev, dask.array.Array):
            raise TypeError('Input arr_x_ev must be type dask.array.Array.')
        if not isinstance(arr_y_ev, dask.array.Array):
            raise TypeError('Input arr_y_ev must be type dask.array.Array.')

        if arr_x_ev.ndim != 2:
            raise TypeError('Input arr_x_ev must be 2D (samples, n_vars).')
        if arr_y_ev.ndim != 2:
            raise TypeError('Input arr_y_ev must be 2D (samples, n_vars).')

        if arr_x_ev.shape[0] != arr_y_ev.shape[0]:
            raise ValueError('Inputs arr_x_ev, arr_y_ev must have equal sample sizes.')

        if arr_x_ev.dtype != arr_y_ev.dtype:
            raise TypeError('Inputs arr_x_ev, arr_y_ev must have same dtype.')

        # TODO: check chunk sizes same

    if client is None:
        raise ValueError('No dask client provided.')

    if params is None:
        params = default_params()
        callbacks = None

    x_chunks = {0: -1, 1: -1}  # TODO: do auto here then use chunk size for y
    y_chunks = {0: -1, 1: 1}

    arr_x = arr_x.rechunk(x_chunks)
    arr_y = arr_y.rechunk(y_chunks)

    if arr_x_ev is not None:
        arr_x_ev = arr_x_ev.rechunk(x_chunks)
        arr_y_ev = arr_y_ev.rechunk(y_chunks)

    models = []
    for i in range(arr_y.shape[1]):

        eval_set = None
        if arr_x_ev is not None and arr_y_ev is not None:
            eval_set = [(arr_x_ev, arr_y_ev[:, i])]

        model = DaskLGBMRegressor(client=client, **params)
        model.fit(
            arr_x, arr_y[:, i],
            eval_set=eval_set,
            eval_names=['valid'],  # ignored if no eval_set
            eval_metric='rmse',    # likewise
            callbacks=callbacks
        )

        models.append(model)

    return models


def predict_models(
        da_ref: xr.DataArray,
        da_tar: xr.DataArray,
        models: list,
        nodata: int | float,
        predict_inplace: bool = True,
        client: Client | None = None
) -> xr.DataArray:
    """TODO"""

    if not isinstance(da_ref, xr.DataArray):
        raise TypeError('Input da_ref must be type xr.DataArray.')
    if not isinstance(da_tar, xr.DataArray):
        raise TypeError('Input da_tar must be type xr.DataArray.')

    if da_ref.ndim not in (2, 3):
        raise ValueError('Input da_ref must be 2D (y, x) or 3D (b, y, x).')
    if da_tar.ndim not in (2, 3):
        raise ValueError('Input da_tar must be 2D (y, x) or 3D (b, y, x).')

    if da_ref.shape[1] != da_tar.shape[1]:
        raise ValueError('Inputs da_ref and da_tar must have same y size.')
    if da_ref.shape[2] != da_tar.shape[2]:
        raise ValueError('Inputs da_ref and da_tar must have same x size.')

    if da_ref.dtype != da_tar.dtype:
        raise TypeError('Inputs da_ref and da_tar must have same dtype.')

    # TODO: ensure chunk sizes same

    if nodata is None:
        raise ValueError('Input nodata must be provided.')

    if models is None:
        raise ValueError('Input models must be provided.')

    if client is None:
        raise ValueError('No dask client provided.')

    # offset for map overlaps of size (1, 1)
    offset = 1

    arr_ref = da_ref.data
    arr_tar = da_tar.data

    if not isinstance(arr_ref, dask.array.Array):
        raise TypeError('Input da_ref must be backed by a dask.array.Array.')
    if not isinstance(arr_tar, dask.array.Array):
        raise TypeError('Input arr_tar must be backed by a dask.array.Array.')

    # we always need (b, y, x) array
    if arr_ref.ndim == 2:
        arr_ref = np.expand_dims(arr_ref, axis=0)
    if arr_tar.ndim == 2:
        arr_tar = np.expand_dims(arr_tar, axis=0)

    if len(models) != arr_tar.shape[0]:
        raise ValueError('Inputs models and da_ref must have same num vars.')

    # where (3, 3) x win all true and y false
    arr_mask = map_overlap(
        _make_predict_mask,
        nodata_mask(arr_ref, nodata),
        nodata_mask(arr_tar, nodata),
        depth=(1, 1),
        boundary=True,  # nodata == true
        dtype=np.bool_
    )

    # extract idx of all missing y and valid x per block
    idx_blocks = _extract_predict_idx_per_block(arr_mask)

    del arr_mask
    gc.collect()

    # if arr_i.size == 0:  # TODO: adapt to list of arrs
    #     raise ValueError('No valid training data found.')

    # extract train x samples at idx
    arr_x = []
    for var in range(arr_ref.shape[0]):
        x_blocks = overlap(
            arr_ref[var, :, :],
            depth=(1, 1),
            boundary=nodata
        )

        x_delays = [
            dask.array.from_delayed(
                dask.delayed(_extract_x)
                (i, b, offset),
                shape=(i.size, 9),  # always 9 pix win
                dtype=x_blocks.dtype
            )
            for i, b in zip(
                idx_blocks,
                x_blocks.to_delayed().ravel()
            )
        ]

        arr_x.append(np.vstack([*x_delays]))

    arr_x = np.hstack(arr_x)

    # predict nodata for each y variable
    arr_y_pred = []
    for i, model in enumerate(models):
        print(f'Predicting variable {i + 1}.')
        arr_y_pred.append(
            model.predict(arr_x).astype(arr_x.dtype)
        )

    #arr_y_pred = np.stack(arr_y_pred, axis=1)
    arr_y_pred = dask.array.stack(arr_y_pred, axis=1)

    del arr_x
    gc.collect()

    # no longer need to consider padding
    offset = 0

    arr_y = []
    for var in range(arr_tar.shape[0]):

        block_shapes = _get_block_shapes(arr_tar[var, :, :])

        t_delays = arr_tar[var, :, :].to_delayed().ravel()
        p_delays = arr_y_pred[:, var].to_delayed().ravel()

        # vstack above removes empty arrays, add back in
        p_delays_full = _pack_with_delayed_dummies(
            idx_blocks,
            p_delays,
            dask.delayed(np.empty((0, ), dtype=da_tar.dtype))  # always 9 pix
        )

        y_delays = [
            dask.array.from_delayed(
                dask.delayed(_fill_y)
                (i, t, p, offset, predict_inplace),
                shape=s,
                dtype=arr_tar.dtype
            )
            for i, t, p, s in zip(
                idx_blocks,
                t_delays,
                p_delays_full,
                block_shapes
            )
        ]

        # ...
        n_y_chunks, n_x_chunks = arr_tar[var, :, :].numblocks
        arr_y_sel = dask.array.block([
            y_delays[i * n_y_chunks: (i + 1) * n_x_chunks]
            for i in range(n_y_chunks)
        ])

        arr_y.append(arr_y_sel)

    arr_y = np.stack(arr_y, axis=0)

    del arr_y_pred
    gc.collect()

    da_out = xr.DataArray(
        arr_y,
        dims=da_tar.dims,
        coords=da_tar.coords,
        attrs=da_tar.attrs
    )

    return da_out


def run(
        da_ref: xr.DataArray,
        da_tar: xr.DataArray,
        nodata: int | float,
        n_total_samples: int | None = 100000,
        oversample_factor: int | float = 1.5,
        percent_train: float | None = 0.9,
        predict_inplace: bool = True,
        rand_seed: int | None = 0,
        params: dict = None,
        callbacks: list | None = None,
        client: Client | None = None
) -> xr.DataArray:
    """TODO"""

    if not isinstance(da_ref, xr.DataArray):
        raise TypeError('Input da_ref must be type xr.DataArray.')
    if not isinstance(da_tar, xr.DataArray):
        raise TypeError('Input da_tar must be type xr.DataArray.')

    if da_ref.ndim not in (2, 3):
        raise ValueError('Input da_ref must be 2D (y, x) or 3D (b, y, x).')
    if da_tar.ndim not in (2, 3):
        raise ValueError('Input da_tar must be 2D (y, x) or 3D (b, y, x).')

    # TODO: ensure chunks same

    if rand_seed is None:
        rand_seed = np.random.randint(10000)

    arr_x, arr_y, arr_x_ev, arr_y_ev = extract_train_set(
        da_ref,
        da_tar,
        nodata,
        n_total_samples,
        oversample_factor,
        percent_train,
        rand_seed
    )

    if arr_x.size == 0 or arr_y.size == 0:
        raise ValueError('No training pixels were extracted.')

    models = calibrate_models(
        arr_x,
        arr_y,
        arr_x_ev,
        arr_y_ev,
        params,
        callbacks,
        client
    )

    del arr_x, arr_y, arr_x_ev, arr_y_ev
    gc.collect()

    da_y = predict_models(
        da_ref,
        da_tar,
        models,
        nodata,
        predict_inplace,
        client
    )

    return da_y
