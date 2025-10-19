
import numpy as np
import numba as nb
import xarray as xr
import xgboost as xgb

from sklearn.model_selection import train_test_split

from noclouds.utils.helpers import has_nodata_1d
from noclouds.utils.helpers import has_nodata_3d
from noclouds.utils.helpers import default_xgb_params
from noclouds.utils.helpers import _prepare_xgb_params


# region Training data creation

@nb.njit(parallel=True)
def _make_train_mask(
        arr_ref: np.ndarray,
        arr_tar: np.ndarray,
        nodata: int | float
) -> np.ndarray:

    _, y_size, x_size = arr_ref.shape
    arr_out = np.zeros((y_size, x_size), np.bool_)

    for yi in nb.prange(1, y_size - 1):
        for xi in range(1, x_size - 1):

            if not has_nodata_1d(arr_tar[:, yi, xi], nodata):
                arr_sel = arr_ref[:, yi - 1:yi + 2, xi - 1:xi + 2]

                if not has_nodata_3d(arr_sel, nodata):
                    arr_out[yi, xi] = True

    return arr_out

@nb.njit(parallel=True)
def _extract_train_xy(
        arr_ref: np.ndarray,
        arr_tar: np.ndarray,
        nodata: int | float,
        n_samples: int,
        rand_seed: int
):

    n_x_vars = arr_ref.shape[0] * 9  # 9 pix per var per win
    n_y_vars = arr_tar.shape[0]

    arr_mask = _make_train_mask(arr_ref, arr_tar, nodata)

    n_valid = np.sum(arr_mask)
    if n_valid == 0:
        return (
            np.empty((0, n_x_vars), dtype=arr_ref.dtype),
            np.empty((0, n_y_vars), dtype=arr_tar.dtype),
        )

    arr_i = np.empty((n_valid, 2), dtype=np.int64)

    i = 0
    for yi in range(arr_ref.shape[1]):
        for xi in range(arr_ref.shape[2]):
            if arr_mask[yi, xi]:
                arr_i[i, 0], arr_i[i, 1] = yi, xi
                i += 1

    # only random sample if subset < total
    if n_samples < n_valid:
        np.random.seed(rand_seed)
        idx = np.arange(arr_i.shape[0])
        np.random.shuffle(idx)
        arr_i = arr_i[idx[:n_samples]]

    arr_x = np.empty((arr_i.shape[0], n_x_vars), arr_ref.dtype)
    arr_y = np.empty((arr_i.shape[0], n_y_vars), arr_tar.dtype)

    for i in nb.prange(arr_i.shape[0]):
        yi, xi = arr_i[i, 0], arr_i[i, 1]
        arr_y[i, :] = arr_tar[:, yi, xi]
        arr_x[i, :] = arr_ref[:, yi - 1:yi + 2, xi - 1:xi + 2].ravel()

    return arr_x, arr_y


def extract_train_set(
        arr_ref: np.ndarray,
        arr_tar: np.ndarray,
        nodata: int | float,
        n_samples: int | None = None,
        rand_seed: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """TODO"""

    if not isinstance(arr_ref, np.ndarray):
        raise TypeError('Input arr_ref must be type np.ndarray.')
    if not isinstance(arr_tar, np.ndarray):
        raise TypeError('Input arr_tar must be type np.ndarray.')

    if arr_ref.ndim == 2:
        arr_ref = np.expand_dims(arr_ref, axis=0)
    if arr_tar.ndim == 2:
        arr_tar = np.expand_dims(arr_tar, axis=0)

    if arr_ref.shape[1] != arr_tar.shape[1]:
        raise ValueError('Inputs arr_ref and arr_tar must have same y size.')
    if arr_ref.shape[2] != arr_tar.shape[2]:
        raise ValueError('Inputs arr_ref and arr_tar must have same x size.')

    if arr_ref.dtype != arr_tar.dtype:
        raise TypeError('Inputs arr_ref and arr_tar must have same dtype.')

    if nodata is None:
        raise ValueError('Input nodata must be provided.')

    if rand_seed is None:
        rand_seed = np.random.randint(10000)

    arr_x, arr_y = _extract_train_xy(
        arr_ref,
        arr_tar,
        nodata,
        n_samples,
        rand_seed
    )

    return arr_x, arr_y

# endregion

# region XGB modelling

def _split_train_test_xy(
        arr_x: np.ndarray,
        arr_y: np.ndarray,
        percent_train: float,
        split_seed: int | None
) -> tuple:

    if percent_train <= 0.0 or percent_train >= 1.0:
        raise ValueError('Input percent_train must be > 0.0 and < 1.0.')

    if split_seed is None:
        split_seed = np.random.randint(10000)

    x_train, x_test, y_train, y_test = train_test_split(
        arr_x,
        arr_y,
        train_size=percent_train,
        random_state=split_seed
    )

    return x_train, x_test, y_train, y_test


def build_xgb_models(
        arr_x: np.ndarray,
        arr_y: np.ndarray,
        xgb_params: dict | None = None
) -> tuple:
    """TODO"""

    if not isinstance(arr_x, np.ndarray):
        raise TypeError('Input arr_x must be type np.ndarray.')
    if not isinstance(arr_y, np.ndarray):
        raise TypeError('Input arr_y must be type np.ndarray.')

    if arr_x.ndim != 2:
        raise TypeError('Input arr_x must be 2D (samples, n_vars).')
    if arr_y.ndim != 2:
        raise TypeError('Input arr_y must be 2D (samples, n_vars).')

    if arr_x.shape[0] != arr_y.shape[0]:
        raise ValueError('Inputs arr_x, arr_y must have equal sample sizes.')

    if arr_x.dtype != arr_y.dtype:
        raise TypeError('Inputs arr_x, arr_y must have same dtype.')

    if xgb_params is None:
        xgb_params = default_xgb_params()

    xgb_params = xgb_params.copy()  # prevent pop outside
    e_xgb_params = _prepare_xgb_params(xgb_params)

    # TODO: bit messy, clean up
    num_boost_round = e_xgb_params.get('num_boost_round')
    percent_train = e_xgb_params.get('percent_train')
    split_seed = e_xgb_params.get('split_seed')
    early_stopping_rounds = e_xgb_params.get('early_stopping_rounds')
    verbose_eval = e_xgb_params.get('verbose_eval')

    arr_x_ev, arr_y_ev = None, None
    if percent_train:
        arr_x, arr_x_ev, arr_y, arr_y_ev = _split_train_test_xy(
            arr_x,
            arr_y,
            percent_train,
            split_seed
        )

    xgb_models = []
    for i in range(arr_y.shape[1]):
        print(f'Training variable {i + 1}.')

        dtrain = xgb.DMatrix(arr_x, arr_y[:, i])

        evals = None
        if percent_train:
            deval = xgb.DMatrix(arr_x_ev, arr_y_ev[:, i])
            evals = [(dtrain, 'train'), (deval, 'eval')]

        xgb_model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval
        )

        xgb_models.append(xgb_model)

    return tuple(xgb_models)

# endregion

# region Prediction data creation

@nb.njit(parallel=True)
def _make_predict_grid(
        arr_ref: np.ndarray,
        arr_tar: np.ndarray
) -> np.ndarray:

    y_size, x_size = arr_ref.shape
    arr_out = np.zeros((y_size, x_size), np.bool)

    for yi in nb.prange(1, y_size - 1):
        for xi in range(1, x_size - 1):
            if arr_tar[yi, xi]:
                if not np.any(arr_ref[yi - 1:yi + 2, xi - 1:xi + 2]):
                    arr_out[yi, xi] = True

    return arr_out


def _samp_predict_idx(
        arr_grid: np.ndarray
) -> np.ndarray:

    arr_i = np.argwhere(arr_grid)
    if arr_i.size == 0:
        return np.zeros(0, dtype=np.int64)

    return arr_i


@nb.njit(parallel=True)
def _extract_predict_x(
        arr_i: np.ndarray,
        arr_ref: np.ndarray
) -> np.ndarray:

    n_rows = arr_i.shape[0]
    n_x_vars = arr_ref.shape[0] * 9  # 9 pix per var per win

    arr_x = np.empty((n_rows, n_x_vars), arr_ref.dtype)

    for i in nb.prange(n_rows):
        yi, xi = arr_i[i, 0], arr_i[i, 1]
        arr_x[i, :] = arr_ref[:, yi - 1:yi + 2, xi - 1:xi + 2].ravel()

    return arr_x


def extract_predict_set(
        arr_ref: np.ndarray,
        arr_tar: np.ndarray,
        nodata: int | float
) -> tuple[np.ndarray, np.ndarray]:
    """TODO"""

    if not isinstance(arr_ref, np.ndarray):
        raise TypeError('Input arr_ref must be type np.ndarray.')
    if not isinstance(arr_tar, np.ndarray):
        raise TypeError('Input arr_tar must be type np.ndarray.')

    if arr_ref.ndim == 2:
        arr_ref = np.expand_dims(arr_ref, axis=0)
    if arr_tar.ndim == 2:
        arr_tar = np.expand_dims(arr_tar, axis=0)

    if arr_ref.shape[1] != arr_tar.shape[1]:
        raise ValueError('Inputs arr_ref and arr_tar must have same y size.')
    if arr_ref.shape[2] != arr_tar.shape[2]:
        raise ValueError('Inputs arr_ref and arr_tar must have same x size.')

    if arr_ref.dtype != arr_tar.dtype:
        raise TypeError('Inputs arr_ref and arr_tar must have same dtype.')

    if nodata is None:
        raise ValueError('Input nodata must be provided.')

    # clamp ref, tar to true where any nodata. build predict grid.
    # predict grid is tar = false and ref 3x3 win all = true.
    arr_grid = _make_predict_grid(
        nodata_mask(arr_ref, nodata),
        nodata_mask(arr_tar, nodata)
    )

    if not np.any(arr_grid):
        raise ValueError('No valid predict pixels detected.')

    # select all pixels to y, x idx arrays.
    arr_i = _samp_predict_idx(arr_grid)

    if arr_i.size == 0:
        raise ValueError('Sampling returned no predict pixels.')

    # extract predict x set using predict indices.
    arr_x = _extract_predict_x(
        arr_i,
        arr_ref
    )

    return arr_i, arr_x

# endregion

# region XGB prediction

def predict_xgb_models(
        arr_i: np.ndarray,
        arr_x: np.ndarray,
        xgb_models: tuple
) -> np.ndarray:
    """TODO"""

    if not isinstance(arr_i, np.ndarray):
        raise TypeError('Input arr_i must be type np.ndarray.')
    if not isinstance(arr_x, np.ndarray):
        raise TypeError('Input arr_x must be type np.ndarray.')

    if arr_i.ndim != 2:
        raise TypeError('Input arr_i must be 2D (samples, n_vars).')
    if arr_x.ndim != 2:
        raise TypeError('Input arr_x must be 2D (samples, n_vars).')

    if arr_i.shape[0] != arr_x.shape[0]:
        raise ValueError('Inputs arr_i, arr_x must have equal sample sizes.')

    if arr_i.dtype != np.int64:
        raise TypeError('Inputs arr_i must have dtype int64.')

    if not isinstance(xgb_models, tuple):
        raise TypeError('Input xgb_models must be type tuple.')

    if len(xgb_models) == 0:
        raise ValueError('Input xgb_models must have at least one model.')

    arr_x = xgb.DMatrix(arr_x)

    arr_y = []
    for i, model in enumerate(xgb_models):
        print(f'Predicting variable {i + 1}.')
        arr_y.append(model.predict(arr_x))

    arr_y = np.column_stack(arr_y)

    return arr_y


@nb.njit(parallel=True)
def fill_predict_iy(
        arr_i: np.ndarray,
        arr_y: np.ndarray,
        arr_tar: np.ndarray,
        predict_inplace: bool
) -> np.ndarray:

    if not predict_inplace:
        arr_tar = arr_tar.copy()  # TODO: check this funcs as expected

    n_rows = arr_i.shape[0]

    for i in nb.prange(n_rows):
        yi, xi = arr_i[i, 0], arr_i[i, 1]
        arr_tar[:, yi, xi] = arr_y[i]

    return arr_tar

# endregion


def run(
        da_ref: xr.DataArray,
        da_tar: xr.DataArray,
        nodata: int | float,
        n_samples: int | None = None,
        rand_seed: int = 0,
        predict_inplace: bool = True,
        xgb_params: dict = None
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

    arr_ref = da_ref.data
    arr_tar = da_tar.data

    arr_x, arr_y = extract_train_set(
        arr_ref,
        arr_tar,
        nodata,
        n_samples,
        rand_seed
    )

    if arr_x.size == 0 or arr_y.size == 0:
        raise ValueError('No training pixels could be extracted.')

    xgb_models = build_xgb_models(arr_x, arr_y, xgb_params)

    arr_i, arr_x = extract_predict_set(
        arr_ref,
        arr_tar,
        nodata
    )

    if arr_i.size == 0 or arr_x.size == 0:
        raise ValueError('No prediction pixels could be extracted.')

    arr_y = predict_xgb_models(
        arr_i,
        arr_x,
        xgb_models
    )

    arr_y = fill_predict_iy(
        arr_i,
        arr_y,
        arr_tar,
        predict_inplace
    )

    da_out = xr.DataArray(
        arr_y,
        dims=da_tar.dims,
        coords=da_tar.coords,
        attrs=da_tar.attrs
    )

    return da_out
