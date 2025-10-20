
import os
import time
import gc
import dask
import dask.array as dk
import numpy as np
import numba as nb
import xarray as xr
import xgboost.dask as xgbd

from dask.array.overlap import overlap

from dask.diagnostics import ProgressBar
from dask.distributed import get_client

from dask_ml.model_selection import train_test_split

from noclouds.utils.helpers import nodata_mask
#from noclouds.utils.helpers import default_xgb_params
#from noclouds.utils.helpers import _prepare_xgb_params


# region Training data extraction

@nb.njit(parallel=True)
def _make_train_mask(
        arr_ref: np.ndarray,
        arr_tar: np.ndarray
) -> np.ndarray:

    y_size, x_size = arr_ref.shape
    arr_out = np.zeros((y_size, x_size), np.bool_)

    for yi in nb.prange(1, y_size - 1):
        for xi in range(1, x_size - 1):
            if not arr_tar[yi, xi]:
                if not np.any(arr_ref[yi - 1:yi + 2, xi - 1:xi + 2]):
                    arr_out[yi, xi] = True

    return arr_out


def _estimate_chunk_train_sizes(
        arr: dk.Array,
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


def _extract_chunk_train_idx(
        block: np.ndarray,
        n_samples: int,
        rand_seed: int
) -> np.ndarray:

    if n_samples == 0:
        return np.array([], dtype=np.int32)

    arr_i = np.flatnonzero(block)
    if arr_i.size == 0:
        return np.array([], dtype=np.int32)

    n_valid = arr_i.size
    n_select = np.min([n_valid, n_samples])

    rng = np.random.default_rng(rand_seed)
    arr_i = rng.choice(
        arr_i,
        size=n_select,
        replace=False
    ).astype(np.int32)

    return arr_i


def _refine_chunk_train_sizes(
        arr_oversample: np.ndarray,
        n_total_samples: int
) -> np.ndarray:

    if np.any(arr_oversample < 0):
        raise ValueError('Input arr_oversample must not contain negative values.')

    total_oversample = arr_oversample.sum()
    if total_oversample == 0:
        raise ValueError('Cannot refine an all-zero array.')
    if n_total_samples >= total_oversample:
        # TODO: warn user
        return arr_oversample

    # proportionally rescale oversampled array
    arr_scaled = arr_oversample * (n_total_samples / total_oversample)

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


def _refine_chunk_train_idx(
        block: np.ndarray,
        n_samples: int,
        rand_seed: int
) -> np.ndarray:

    if n_samples == 0:
        return np.array([], dtype=np.int32)

    if block.size == 0:
        return np.array([], dtype=np.int32)

    n_valid = block.size
    n_select = np.min([n_valid, n_samples])

    rng = np.random.default_rng(rand_seed)
    arr_i = rng.choice(
        block,
        size=n_select,
        replace=False
    )

    arr_i = arr_i.astype(np.int32)

    return arr_i


def _split_idx_blocks(
        idx_blocks: list[np.ndarray],
        train_percent: float,
        rand_seed: int
) -> tuple[list[np.ndarray], list[np.ndarray]]:

    idx_train_blocks = []
    idx_test_blocks = []
    for idx_block in idx_blocks:
        idx_train_block, idx_test_block = train_test_split(
            idx_block,
            train_size=train_percent,
            random_state=rand_seed,
            shuffle=True
        )

        idx_train_blocks.append(idx_train_block)
        idx_test_blocks.append(idx_test_block)

    # TODO: convert to this
    # np.random.seed(rand_seed)
    #
    # n_rows = block.shape[0]
    #
    # arr_i = np.arange(n_rows)
    # np.random.shuffle(arr_i)
    #
    # arr_train_i = arr_i[:n_train_samples]
    # arr_test_i = arr_i[n_train_samples:]
    #
    # block_train = block[arr_train_i, :]
    # block_test = block[arr_test_i, :]
    #
    # return block_train, block_test

    return idx_train_blocks, idx_test_blocks


@nb.njit(parallel=True)
def _extract_train_x(
        arr_i: np.ndarray,
        arr_ref: np.ndarray
):
    n_vars = 9  # 9 pix per var per win
    x_size = arr_ref.shape[1] - 2  # - 2 exclude pad

    n_idx = len(arr_i)
    if n_idx == 0:
        return np.empty((0, n_vars), dtype=arr_ref.dtype)

    arr_x = np.empty((n_idx, n_vars), arr_ref.dtype)

    for i in nb.prange(n_idx):
        j = arr_i[i]
        yi = (j // x_size) + 1  # + 1 to offset map_overlap
        xi = (j % x_size) + 1
        arr_x[i, :] = arr_ref[yi - 1:yi + 2, xi - 1:xi + 2].ravel()

    return arr_x


@nb.njit(parallel=True)
def _extract_train_y(
        arr_i: np.ndarray,
        arr_tar: np.ndarray
):
    n_vars = 1
    x_size = arr_tar.shape[1] - 2  # - 2 exclude pad

    n_idx = len(arr_i)
    if n_idx == 0:
        return np.empty((0, n_vars), dtype=arr_tar.dtype)

    arr_y = np.empty((n_idx, n_vars), arr_tar.dtype)

    for i in nb.prange(n_idx):
        j = arr_i[i]
        yi = (j // x_size) + 1  # + 1 to offset map_overlap
        xi = (j % x_size) + 1
        arr_y[i, :] = arr_tar[yi, xi]

    return arr_y


def extract_train_set(
        da_ref: xr.DataArray,
        da_tar: xr.DataArray,
        nodata: int | float,
        n_total_samples: int = 10000,
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

    # TODO: check n_samples, oversample?

    # TODO: percent train check?

    if rand_seed is None:
        rand_seed = np.random.randint(10000)

    arr_ref = da_ref.data
    arr_tar = da_tar.data

    # we always need (b, y, x) array
    if arr_ref.ndim == 2:
        arr_ref = np.expand_dims(arr_ref, axis=0)
    if arr_tar.ndim == 2:
        arr_tar = np.expand_dims(arr_tar, axis=0)

    # clamp each into (y, x) nodata masks
    arr_ref_nd = nodata_mask(arr_ref, nodata)
    arr_tar_nd = nodata_mask(arr_tar, nodata)

    # ...
    arr_mask = dk.map_overlap(
        _make_train_mask,
        arr_ref_nd,
        arr_tar_nd,
        depth=(1, 1),
        boundary=True,  # mask nodata == true
        dtype=np.bool_
    )

    # per-chunk estimates of oversizes train samples
    estimate_sizes = _estimate_chunk_train_sizes(
        arr_mask,
        n_total_samples,
        oversample_factor
    )

    # todo: determine max dtype

    # get random idx from oversized estimates per-chunk
    delays = [
        dask.delayed(_extract_chunk_train_idx)
        (b, n, rand_seed)
        for b, n in zip(
            arr_mask.to_delayed().ravel(),
            estimate_sizes
        )
    ]

    with ProgressBar():
        idx_blocks = dask.compute(*delays)

    # refine per-chunk estimates of oversizes train samples
    sampled_sizes = np.array([a.size for a in idx_blocks])
    refined_sizes = _refine_chunk_train_sizes(
        sampled_sizes,
        n_total_samples
    )

    # refine random idx from oversized estimates per-chunk
    idx_blocks = [
        _refine_chunk_train_idx(b, n, rand_seed)
        for b, n in zip(idx_blocks, refined_sizes)
    ]

    # TODO: ensure sum is == n_total_samples

    # TODO: could implement x, y split here via idx

    if percent_train:
        idx_blocks, idx_eval_blocks = _split_idx_blocks(
            idx_blocks,
            percent_train,
            rand_seed
        )

    # extract train x samples at idx
    arr_x = []
    for var in range(arr_ref.shape[0]):
        arr_var = arr_ref[var, :, :]

        blocks = overlap(
            arr_var,
            depth=(1, 1),
            boundary=nodata
        )

        delays = [
            dk.from_delayed(
                dask.delayed(_extract_train_x)
                (i, b),
                shape=(i.size, 9),  # always 9 pix win
                dtype=arr_var.dtype
            )
            for i, b in zip(
                idx_blocks,
                blocks.to_delayed().ravel()
            )
        ]

        arr_x.append(np.vstack([*delays]))

    arr_x = np.hstack(arr_x)

    # extract train y samples at idx
    arr_y = []
    for var in range(arr_tar.shape[0]):
        arr_var = arr_tar[var, :, :]

        blocks = overlap(
            arr_var,
            depth=(1, 1),
            boundary=nodata
        )

        delays = [
            dk.from_delayed(
                dask.delayed(_extract_train_y)
                (i, b),
                shape=(i.size, 1),  # always 1 var
                dtype=arr_var.dtype
            )
            for i, b in zip(
                idx_blocks,
                blocks.to_delayed().ravel()
            )
        ]

        arr_y.append(np.vstack([*delays]))

    arr_y = np.hstack(arr_y)

    if not percent_train:
        return arr_x, arr_y

    # extract eval x samples at idx
    arr_x_eval = []
    for var in range(arr_ref.shape[0]):
        arr_var = arr_ref[var, :, :]

        blocks = overlap(
            arr_var,
            depth=(1, 1),
            boundary=nodata
        )

        delays = [
            dk.from_delayed(
                dask.delayed(_extract_train_x)
                (i, b),
                shape=(i.size, 9),  # always 9 pix win
                dtype=arr_var.dtype
            )
            for i, b in zip(
                idx_eval_blocks,
                blocks.to_delayed().ravel()
            )
        ]

        arr_x_eval.append(np.vstack([*delays]))

    arr_x_eval = np.hstack(arr_x_eval)

    # extract eval y samples at idx
    arr_y_eval = []
    for var in range(arr_tar.shape[0]):
        arr_var = arr_tar[var, :, :]

        blocks = overlap(
            arr_var,
            depth=(1, 1),
            boundary=nodata
        )

        delays = [
            dk.from_delayed(
                dask.delayed(_extract_train_y)
                (i, b),
                shape=(i.size, 1),  # always 1 var
                dtype=arr_var.dtype
            )
            for i, b in zip(
                idx_eval_blocks,
                blocks.to_delayed().ravel()
            )
        ]

        arr_y_eval.append(np.vstack([*delays]))

    arr_y_eval = np.hstack(arr_y_eval)

    return arr_x, arr_y, arr_x_eval, arr_y_eval

# endregion


# region XGB modelling

def train_xgb_models(
        arr_x: dk.Array,
        arr_y: dk.Array,
        arr_x_eval: dk.Array,
        arr_y_eval: dk.Array,
        xgb_params: dict | None = None
) -> tuple:

    if not isinstance(arr_x, dk.Array):
        raise TypeError('Input arr_x must be type dask.array.Array.')
    if not isinstance(arr_y, dk.Array):
        raise TypeError('Input arr_y must be type dask.array.Array.')

    if arr_x.ndim != 2:
        raise TypeError('Input arr_x must be 2D (samples, n_vars).')
    if arr_y.ndim != 2:
        raise TypeError('Input arr_y must be 2D (samples, n_vars).')

    if arr_x.shape[0] != arr_y.shape[0]:
        raise ValueError('Inputs arr_x, arr_y must have equal sample sizes.')

    if arr_x.dtype != arr_y.dtype:
        raise TypeError('Inputs arr_x, arr_y must have same dtype.')

    # TODO: checks on evals

    if xgb_params is None:
        xgb_params = default_xgb_params()

    xgb_params = xgb_params.copy()  # prevent pop outside
    e_xgb_params = _prepare_xgb_params(xgb_params)

    # TODO: bit messy, clean up
    num_boost_round = e_xgb_params.get('num_boost_round')
    early_stopping_rounds = e_xgb_params.get('early_stopping_rounds')
    verbose_eval = e_xgb_params.get('verbose_eval')

    # must rechunk -1 across cols
    arr_x = arr_x.rechunk({0: None, 1: -1})
    if arr_x_eval is not None:
        arr_x_eval = arr_x_eval.rechunk({0: None, 1: -1})

    client = get_client()

    xgb_models = []
    for i in range(arr_y.shape[1]):
        print(f'Training variable {i + 1}.')

        dtrain = xgbd.DaskDMatrix(client, arr_x, arr_y[:, i])

        evals = None
        if arr_x_eval is not None and arr_y_eval is not None:
            deval = xgbd.DaskDMatrix(client, arr_x_eval, arr_y_eval[:, i])  # will cause 2 computes if not persisted early
            evals = [(dtrain, 'train'), (deval, 'eval')]

        xgb_model = xgbd.train(
            client,
            xgb_params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval
        )

        # TODO: official rand forest settings - test speed
        # params = {
        #     "colsample_bynode": 0.8,
        #     "learning_rate": 1,
        #     "max_depth": 5,
        #     "num_parallel_tree": 100,  # n_trees
        #     "objective": "reg:squarederror",
        #     "subsample": 0.8,
        #     "tree_method": "hist",
        #     "device": "cpu",
        # }
        #
        # xgb_model = xgbd.train(
        #     client,
        #     params,
        #     dtrain,
        #     num_boost_round=1,
        #     evals=evals,
        #     early_stopping_rounds=10,
        #     verbose_eval=5
        # )

        xgb_models.append(xgb_model)

    return tuple(xgb_models)

# endregion


@nb.njit(parallel=True)
def _make_predict_mask(
        arr_ref: np.ndarray,
        arr_tar: np.ndarray
) -> np.ndarray:

    y_size, x_size = arr_ref.shape
    arr_out = np.zeros((y_size, x_size), np.bool_)

    for yi in nb.prange(1, y_size - 1):
        for xi in range(1, x_size - 1):
            if arr_tar[yi, xi]:
                if not np.any(arr_ref[yi - 1:yi + 2, xi - 1:xi + 2]):
                    arr_out[yi, xi] = True

    return arr_out


def _extract_chunk_predict_idx(
        block: np.ndarray
) -> np.ndarray:

    arr_i = np.flatnonzero(block)
    if arr_i.size == 0:
        return np.array([], dtype=np.int32)

    arr_i = arr_i.astype(np.int32)

    return arr_i


# TODO: same as train_x func, merge
@nb.njit(parallel=True)
def _extract_predict_x(
        arr_i: np.ndarray,
        arr_ref: np.ndarray
):
    n_vars = 9  # 9 pix per var per win
    x_size = arr_ref.shape[1] - 2  # - 2 exclude pad

    n_idx = len(arr_i)
    if n_idx == 0:
        return np.empty((0, n_vars), dtype=arr_ref.dtype)

    arr_x = np.empty((n_idx, n_vars), arr_ref.dtype)

    for i in nb.prange(n_idx):
        j = arr_i[i]
        yi = (j // x_size) + 1  # + 1 to offset map_overlap
        xi = (j % x_size) + 1
        arr_x[i, :] = arr_ref[yi - 1:yi + 2, xi - 1:xi + 2].ravel()

    return arr_x


@nb.njit(parallel=True)
def _fill_predict_y(
        arr_i: np.ndarray,
        arr_y_pred: np.ndarray,
        arr_tar: np.ndarray,
        predict_inplace: bool
) -> np.ndarray:

    x_size = arr_tar.shape[1]  # - 2 if using pad

    if not predict_inplace:
        arr_tar = arr_tar.copy()  # TODO: check this funcs as expected

    n_idx = len(arr_i)
    if n_idx == 0:
        return arr_tar

    for i in nb.prange(n_idx):
        j = arr_i[i]
        yi = (j // x_size)  # + 1 if using pad
        xi = (j % x_size) # + 1
        arr_tar[yi, xi] = arr_y_pred[i]

    return arr_tar



def predict_xgb_models(
        da_ref: xr.DataArray,
        da_tar: xr.DataArray,
        nodata: int | float,
        xgb_models: tuple
) -> xr.DataArray:

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

    #if xgb_models is None:
        #raise ValueError('Input xgb_models must be provided.')

    arr_ref = da_ref.data
    arr_tar = da_tar.data

    # we always need (b, y, x) array
    if arr_ref.ndim == 2:
        arr_ref = np.expand_dims(arr_ref, axis=0)
    if arr_tar.ndim == 2:
        arr_tar = np.expand_dims(arr_tar, axis=0)

    #if len(xgb_models) != arr_ref.shape[0]:
        #raise ValueError('Inputs xgb_models and da_ref must have same num variables.')

    # clamp each into (y, x) nodata masks
    arr_ref_nd = nodata_mask(arr_ref, nodata)
    arr_tar_nd = nodata_mask(arr_tar, nodata)

    # ...
    arr_mask = dk.map_overlap(
        _make_predict_mask,
        arr_ref_nd,
        arr_tar_nd,
        depth=(1, 1),
        boundary=True,  # mask nodata == true
        dtype=np.bool_
    )

    # todo: determine max dtype

    # get predict idx per-chunk without pre-size estimate
    delays = [
        dask.delayed(_extract_chunk_predict_idx)(b)
        for b in arr_mask.to_delayed().ravel()
    ]

    with ProgressBar():
        idx_blocks = dask.compute(*delays)

    # extract train x samples at idx
    arr_x = []
    for var in range(arr_ref.shape[0]):
        arr_var = arr_ref[var, :, :]

        blocks = overlap(
            arr_var,
            depth=(1, 1),
            boundary=nodata
        )

        delays = [
            dk.from_delayed(
                dask.delayed(_extract_predict_x)
                (i, b),
                shape=(i.size, 9),  # always 9 pix win
                dtype=arr_var.dtype
            )
            for i, b in zip(
                idx_blocks,
                blocks.to_delayed().ravel()
            )
        ]

        arr_x.append(np.vstack([*delays]))

    arr_x = np.hstack(arr_x)

    # must rechunk -1 across cols
    arr_x = arr_x.rechunk({0: None, 1: -1})

    # TODO: try this for better scheduler handling
    # def process_batch(batch):
    #     return [_extract_train_idx(b, n, 123) for b, n in batch]
    #
    # # Create small groups of delayed inputs
    # grouped = [
    #     dask.delayed(process_batch)(list(zip(blocks, sizes)))
    #     for blocks, sizes in zip(
    #         np.array_split(arr_mask.to_delayed().ravel(), 100),
    #         np.array_split(estimate_idx_sizes, 100)
    #     )
    # ]
    #
    # # This creates 100 "batch tasks" instead of 10,000 tiny ones
    # with ProgressBar():
    #     results = dask.compute(*grouped)

    # TODO: or this
    # def process_batch(batch, rand_seed):
    #     return [_extract_train_idx(b, n, rand_seed) for b, n in batch]
    #
    # # group (b, n) pairs into batches of 100
    # bn_pairs = list(zip(arr_mask.to_delayed().ravel(), estimate_idx_sizes))
    # batch_size = 100
    # batches = [bn_pairs[i:i + batch_size] for i in range(0, len(bn_pairs), batch_size)]
    #
    # # make one delayed task per batch
    # batch_tasks = [dask.delayed(process_batch)(batch, rand_seed) for batch in batches]
    #
    # # compute all batches
    # with ProgressBar():
    #     results = dask.compute(*batch_tasks)

    client = get_client()

    arr_x = xgbd.DaskDMatrix(client, arr_x)

    arr_y_pred = []
    for i, model in enumerate(xgb_models):
        print(f'Predicting variable {i + 1}.')
        arr_out = xgbd.predict(client, model, arr_x)
        arr_out = arr_out.astype(arr_tar.dtype)
        arr_y_pred.append(arr_out)

    #arr_y_pred = np.hstack([*arr_y_pred])
    arr_y_pred = np.stack(arr_y_pred, axis=1)

    print(arr_y_pred.compute())

    # _fill_predict_y
    #
    # arr_i: np.ndarray,
    # arr_y_pred: np.ndarray,
    # arr_tar: np.ndarray,
    # predict_inplace: bool

    predict_inplace = True

    # ...
    arr_out = []
    for var in range(arr_tar.shape[0]):
        blocks = arr_tar[var, :, :]

        # blocks = overlap(
        #     arr_var,
        #     depth=(1, 1),
        #     boundary=nodata
        # )

        delays = [
            dk.from_delayed(
                dask.delayed(_fill_predict_y)
                (i, b, t, predict_inplace),
                shape=b.shape, #(i.size, 9),
                dtype=b.dtype
            )
            for i, b, t in zip(
                idx_blocks,
                blocks.to_delayed().ravel(),
                arr_tar.to_delayed().ravel()
            )
        ]

        arr_out.append(np.vstack([*delays]))

    arr_out = np.hstack(arr_out)

    with ProgressBar():
        arr_out = arr_out.compute()

    print(arr_out)


    return arr_out


def run(
        da_ref: xr.DataArray,
        da_tar: xr.DataArray,
        nodata: int | float,
        n_total_samples: int | None = 10000,
        oversample_factor: int | float = 1.5,
        percent_train: float | None = 0.9,
        rand_seed: int | None = 0,
        xgb_params: dict = None
) -> xr.DataArray:

    if not isinstance(da_ref, xr.DataArray):
        raise TypeError('Input da_ref must be type xr.DataArray.')
    if not isinstance(da_tar, xr.DataArray):
        raise TypeError('Input da_tar must be type xr.DataArray.')

    if da_ref.ndim not in (2, 3):
        raise ValueError('Input da_ref must be 2D (y, x) or 3D (b, y, x).')
    if da_tar.ndim not in (2, 3):
        raise ValueError('Input da_tar must be 2D (y, x) or 3D (b, y, x).')

    if rand_seed is None:
        rand_seed = np.random.randint(10000)

    # # ...
    # arr_x, arr_y, arr_x_eval, arr_y_eval = extract_train_set(
    #     da_ref,
    #     da_tar,
    #     nodata,
    #     n_total_samples,
    #     oversample_factor,
    #     percent_train,
    #     rand_seed
    # )
    #
    # if arr_x.size == 0 or arr_y.size == 0:
    #     raise ValueError('No training pixels could be extracted.')
    #
    #
    # xgb_models = train_xgb_models(
    #     arr_x,
    #     arr_y,
    #     arr_x_eval,
    #     arr_y_eval,
    #     xgb_params
    # )

    # TODO: pickle and play

    # Save each model as JSON (recommended format)
    # for i, model in enumerate(xgb_models):
    #     fn = f'model_{i}.json'
    #     fp = os.path.join(r'C:\Users\Lewis\Desktop\models', fn)
    #     model['booster'].save_model(fp)

    from pathlib import Path
    from glob import glob

    model_dir = Path(r'C:\Users\Lewis\Desktop\models')
    model_files = sorted(glob(str(model_dir / "*.json")))

    xgb_models = []
    for fp in model_files:
        booster = xgbd.Booster()
        booster.load_model(fp)
        xgb_models.append(booster)

    arr_y_pred = predict_xgb_models(
        da_ref,
        da_tar,
        nodata,
        xgb_models
    )






    # arr_i, arr_x = extract_predict_set(
    #     arr_ref,
    #     arr_tar,
    #     nodata
    # )
    #
    # if arr_i.size == 0 or arr_x.size == 0:
    #     raise ValueError('No prediction pixels could be extracted.')
    #
    # arr_y = predict_xgb_models(
    #     arr_i,
    #     arr_x,
    #     xgb_models
    # )
    #
    # arr_y = fill_predict_iy(
    #     arr_i,
    #     arr_y,
    #     arr_tar,
    #     predict_inplace
    # )
    #
    # da_out = xr.DataArray(
    #     arr_y,
    #     dims=da_tar.dims,
    #     coords=da_tar.coords,
    #     attrs=da_tar.attrs
    # )

    # with ProgressBar():
    #     arr_x, arr_y = dask.compute(arr_x, arr_y)

    return
