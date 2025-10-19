
import gc
import numpy as np
import numba as nb
import xarray as xr

from lightgbm import LGBMRegressor

from noclouds.utils.helpers import nodata_mask
from noclouds.utils.helpers import default_params


# TODO: exact same as dask, ref this one in lazy
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


# TODO: exact same as dask, ref this one in lazy
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


# TODO: exact same as dask, ref this one in lazy
def _extract_train_idx(
        arr: np.ndarray,
        n_samples: int,
        rand_seed: int
) -> np.ndarray:

    if n_samples == 0:
        return np.array([], dtype=np.int32)

    arr_i = np.flatnonzero(arr)
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


def _split_train_eval_idx(
        arr: np.ndarray,
        train_percent: float,
        rand_seed: int
) -> tuple[np.ndarray, np.ndarray]:

    if train_percent < 0 or train_percent > 1:
        raise ValueError('Input train_percent must be between 0 and 1')

    rng = np.random.default_rng(rand_seed)
    arr_i = rng.permutation(arr)
    split_i = int(len(arr_i) * train_percent)

    return arr_i[:split_i], arr_i[split_i:]


# TODO: exact same as dask, ref this one in lazy
def _extract_predict_idx(
        arr: np.ndarray
) -> np.ndarray:

    arr_i = np.flatnonzero(arr)
    if arr_i.size == 0:
        return np.array([], dtype=np.int32)

    arr_i = arr_i.astype(np.int32)

    return arr_i


# TODO: exact same as dask, ref this one in lazy
@nb.njit(parallel=True)
def _extract_x(
        arr_i: np.ndarray,
        arr_ref: np.ndarray,
        offset: int = 0
):
    n_vars = 9  # 9 pix per var per win
    x_size = arr_ref.shape[1] - (offset * 2)

    n_idx = len(arr_i)
    if n_idx == 0:
        return np.empty((0, n_vars), dtype=arr_ref.dtype)

    arr_x = np.empty((n_idx, n_vars), arr_ref.dtype)

    for i in nb.prange(n_idx):
        j = arr_i[i]
        yi = (j // x_size) + offset
        xi = (j % x_size)  + offset
        arr_x[i, :] = arr_ref[yi - 1:yi + 2, xi - 1:xi + 2].ravel()

    return arr_x


# TODO: exact same as dask, ref this one in lazy
@nb.njit(parallel=True)
def _extract_y(
        arr_i: np.ndarray,
        arr_tar: np.ndarray,
        offset: int = 0
):
    n_vars = 1
    x_size = arr_tar.shape[1] - (offset * 2)

    n_idx = len(arr_i)
    if n_idx == 0:
        return np.empty((0, n_vars), dtype=arr_tar.dtype)

    arr_y = np.empty((n_idx, n_vars), arr_tar.dtype)

    for i in nb.prange(n_idx):
        j = arr_i[i]
        yi = (j // x_size) + offset
        xi = (j % x_size)  + offset
        arr_y[i, :] = arr_tar[yi, xi]

    return arr_y


@nb.njit(parallel=True)
def _fill_y(
        arr_i: np.ndarray,
        arr_tar: np.ndarray,
        arr_y_pred: np.ndarray,
        offset: int = 0,
        predict_inplace: bool = True
) -> np.ndarray:

    x_size = arr_tar.shape[1] - (offset * 2) # remove pad

    if not predict_inplace:
        arr_tar = arr_tar.copy()

    n_idx = len(arr_i)
    if n_idx == 0:
        return arr_tar

    for i in nb.prange(n_idx):
        j = arr_i[i]
        yi = (j // x_size) + offset
        xi = (j % x_size)  + offset
        arr_tar[yi, xi] = arr_y_pred[i]

    return arr_tar


def extract_train_set(
        da_ref: xr.DataArray,
        da_tar: xr.DataArray,
        nodata: int | float,
        n_total_samples: int = 100000,
        percent_train: float | None = 0.9,
        rand_seed: int = 0
) -> tuple:
    """
    Standalone function to extract paired training and evaluation datasets
    from reference and target rasters. Implementation uses numba for very fast
    windowed pixel iteration.

    This function is based on the training / evaluation data logic discussed
    by Wang et al. (2022). Basically, the algorithm iterates over every pixel
    in the overlapping reference and target datasets. To be valid, the pixel
    in the target (y) must not be NoData, and the surrounding 9 pixels in the
    reference set (x) must also not have any NoData. If so, these pixels are
    flagged as valid training samples. Once complete, they're randomly sampled
    to n_total_samples size. The output are four numpy arrays: `arr_x`, `arr_y`.
    If percent_train is not None, a `arr_x_eval` and `arr_y_eval` set are also
    provided, else they are None, None.

    Parameters
    ----------
    da_ref : xr.DataArray
        Reference data (predictor variables), 2D or 3D (bands, y, x). Must
        have same y, x shape and NoData value as `da_tar`. Band size can differ.
    da_tar : xr.DataArray
        Target data (response variables), 2D or 3D (bands, y, x). Must
        have same y, x shape and NoData value as `da_ref`. Band size can differ.
    nodata : int or float
        NoData value identifying invalid pixels. Cloud or gap pixels should
        have this value. Must be provided.
    n_total_samples : int, default=100000
        Number of training samples to extract. If None, all valid pixels are used.
    percent_train : float or None, default=0.9
        Fraction of samples used for model training (rest for evaluation). If None,
        no evaluation will be performed.
    rand_seed : int or None, default=0
        Random seed for reproducibility. Used for random sampling.

    Returns
    -------
    tuple
        (arr_x, arr_y, arr_x_ev, arr_y_ev)
        - arr_x : np.ndarray
            Training feature matrix (samples * features).
        - arr_y : np.ndarray
            Training target values (samples * targets).
        - arr_x_ev : np.ndarray or None
            Evaluation feature matrix, or None if `percent_train` is None.
        - arr_y_ev : np.ndarray or None
            Evaluation target values, or None if `percent_train` is None.
    """

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

    # need (b, y, x) arrays
    if arr_ref.ndim == 2:
        arr_ref = np.expand_dims(arr_ref, axis=0)
    if arr_tar.ndim == 2:
        arr_tar = np.expand_dims(arr_tar, axis=0)

    # where (3, 3) x win all true and y true
    arr_mask = _make_train_mask(
        nodata_mask(arr_ref, nodata),
        nodata_mask(arr_tar, nodata)
    )

    if rand_seed is None:
        rand_seed = np.random.randint(10000)

    if n_total_samples <= 0:
        raise ValueError('Input n_total_samples must be > 0.')

    # extract random sample of valid training indices
    arr_i = _extract_train_idx(
        arr_mask,
        n_total_samples,
        rand_seed
    )

    del arr_mask
    gc.collect()

    if arr_i.size == 0:
        raise ValueError('No valid training data found.')

    # split idx into train / eval sets if requested
    if percent_train is not None:
        arr_i, arr_i_ev = _split_train_eval_idx(
            arr_i,
            percent_train,
            rand_seed
        )

    # no padding so no need for an offset
    offset = 0

    # make training x set
    arr_x = np.hstack([
        _extract_x(arr_i, arr_var, offset)
        for arr_var in arr_ref
    ])

    # make training y set
    arr_y = np.hstack([
        _extract_y(arr_i, arr_var, offset)
        for arr_var in arr_tar
    ])

    if percent_train is None:
        return arr_x, arr_y, None, None

    # make eval x set
    arr_x_ev = np.hstack([
        _extract_x(arr_i_ev, arr_var, offset)
        for arr_var in arr_ref
    ])

    # make eval y set
    arr_y_ev = np.hstack([
        _extract_y(arr_i_ev, arr_var, offset)
        for arr_var in arr_tar
    ])

    return arr_x, arr_y, arr_x_ev, arr_y_ev


def calibrate_models(
        arr_x: np.ndarray,
        arr_y: np.ndarray,
        arr_x_ev: np.ndarray | None = None,
        arr_y_ev: np.ndarray | None = None,
        params: dict | None = None,
        callbacks: list | None = None
) -> list:
    """
    Train one LightGBM regression model per target variable. Implemnts a
    random forest version of LightGBM based on the official documentation.
    Users can override this to any parameter setup by providing parameters
    and callbacks (e.g., for early stopping) as supported by the Scikit-Learn
    package implemntation of LightGBM.

    Parameters
    ----------
    arr_x : np.ndarray
        Training features of shape (n_samples, n_features). Must have same
        number of rows as `arr_y`.
    arr_y : np.ndarray
        Training targets of shape (n_samples, n_targets). Must have same
        number of rows as `arr_x`.
    arr_x_ev : np.ndarray or None, optional
        Evaluation features for validation. Must have same number of rows as
        `arr_y_ev`.
    arr_y_ev : np.ndarray or None, optional
        Evaluation targets for validation. Must have same number of rows as
        `arr_x_ev`.
    params : dict, optional
        LightGBM hyperparameters. If None, defaults from `default_params()`,
        which utilises a random forest version of LightGBM, are used.
    callbacks : list, optional
        Optional LightGBM callback functions (e.g., for early stopping).

    Returns
    -------
    list
        List of trained `LGBMRegressor` models, one per target (y) variable.

    Notes
    -----
    Each column in `arr_y` is treated as an independent regression target.
    If evaluation sets are provided, they are passed to LightGBM’s `eval_set`
    for monitoring during training.
    """

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

    if (arr_x_ev is None) != (arr_y_ev is None):
        raise ValueError('Inputs arr_x_ev, arr_y_ev must both be provided or ignored.')

    if arr_x_ev is not None and arr_y_ev is not None:

        if not isinstance(arr_x_ev, np.ndarray):
            raise TypeError('Input arr_x_ev must be type np.ndarray.')
        if not isinstance(arr_y_ev, np.ndarray):
            raise TypeError('Input arr_y_ev must be type np.ndarray.')

        if arr_x_ev.ndim != 2:
            raise TypeError('Input arr_x_ev must be 2D (samples, n_vars).')
        if arr_y_ev.ndim != 2:
            raise TypeError('Input arr_y_ev must be 2D (samples, n_vars).')

        if arr_x_ev.shape[0] != arr_y_ev.shape[0]:
            raise ValueError('Inputs arr_x_ev, arr_y_ev must have equal sample sizes.')

        if arr_x_ev.dtype != arr_y_ev.dtype:
            raise TypeError('Inputs arr_x_ev, arr_y_ev must have same dtype.')

    if params is None:
        params = default_params()
        callbacks = None

    models = []
    for i in range(arr_y.shape[1]):

        eval_set = None
        if arr_x_ev is not None and arr_y_ev is not None:
            eval_set = [(arr_x_ev, arr_y_ev[:, i])]

        model = LGBMRegressor(**params)
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
) -> xr.DataArray:
    """
    Apply trained LightGBM models to predict target values for all missing
    pixels in the target (`da_tar`) xarray data array.

    For each model, features are constructed from 3×3 neighborhoods of the
    reference array and predictions are written back into the target array
    where valid nodata conditions are met. This is based on the Wang et al.
    (2022) paper. Numba is used for speed.

    Parameters
    ----------
    da_ref : xr.DataArray
        Reference input data used for prediction, 2D or 3D (bands, y,  x).
    da_tar : xr.DataArray
        Target output data template, 2D or 3D (bands, y, x).
    models : list
        List of trained `LGBMRegressor` models (one per target (y) band).
    nodata : int or float
        NoData value marking invalid pixels in input arrays. Must be provided.
    predict_inplace : bool, default=True
        If True, predictions overwrite valid pixels in `da_tar` in place.
        Otherwise, a copy of `da_tar` is modified.

    Returns
    -------
    xr.DataArray
        Predicted target data array with the same shape, dims, and coords as `da_tar`.

    Notes
    -----
    Predictions are performed for all NoData target (y) pixels whose 3×3 reference
    window contains no nodata values. See Wang et al. (2022) for more details.
    """

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

    if models is None:
        raise ValueError('Input models must be provided.')

    # always use no offset (coz no map_overlap)
    offset = 0

    arr_ref = da_ref.data
    arr_tar = da_tar.data

    # we always need (b, y, x) array
    if arr_ref.ndim == 2:
        arr_ref = np.expand_dims(arr_ref, axis=0)
    if arr_tar.ndim == 2:
        arr_tar = np.expand_dims(arr_tar, axis=0)

    if len(models) != arr_tar.shape[0]:
        raise ValueError('Inputs models and da_ref must have same num vars.')

    arr_mask = _make_predict_mask(
        nodata_mask(arr_ref, nodata),
        nodata_mask(arr_tar, nodata)
    )

    arr_i = _extract_predict_idx(arr_mask)

    del arr_mask
    gc.collect()

    # make predict x set
    arr_x = np.hstack([
        _extract_x(arr_i, arr_var, offset)
        for arr_var in arr_ref
    ])

    # predict nodata for each y variable
    arr_y_pred = []
    for i, model in enumerate(models):
        print(f'Predicting variable {i + 1}.')
        arr_y_pred.append(
            model.predict(arr_x).astype(arr_x.dtype)
        )

    arr_y_pred = np.hstack([arr_y_pred])

    del arr_x
    gc.collect()

    # fill missing with predicted
    arr_y = np.stack([
        _fill_y(arr_i, t, p, offset, predict_inplace)
        for t, p in zip(arr_tar, arr_y_pred)
    ], axis=0)

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
        percent_train: float | None = 0.9,
        predict_inplace: bool = True,
        rand_seed: int | None = 0,
        params: dict = None,
        callbacks: list | None = None
) -> xr.DataArray:
    """
    End-to-end Spatial-Spectral Random Forest Regression technique
    for multi-temporal clouded and missing pixel filling. Algorithm is
    identical to Wang et al. (2022) except LightGBM is used to implement
    the random forest regressor.

    This function orchestrates training data extraction, model calibration,
    and prediction in a single call. It identifies valid training samples
    using numba for speed, fits LightGBM regressors per target variable, and
    applies them to produce a spatially complete predicted xarray data array.
    See the ssrf.lazy implementation for dask version.

    Parameters
    ----------
    da_ref : xr.DataArray
        Reference data (predictor variables), 2D or 3D (bands, y, x). Must
        have same y, x shape and NoData value as `da_tar`. Band size can differ.
    da_tar : xr.DataArray
        Target data (response variables), 2D or 3D (bands, y, x). Must
        have same y, x shape and NoData value as `da_ref`. Band size can differ.
    nodata : int or float
        NoData value identifying invalid pixels. Cloud or gap pixels should
        have this value. Must be provided.
    n_total_samples : int, default=100000
        Number of training samples to extract. If None, all valid pixels are used.
    percent_train : float or None, default=0.9
        Fraction of samples used for model training (rest for evaluation). If None,
        no evaluation will be performed.
    predict_inplace : bool, default=True
        Whether predictions overwrite the target array in place. If True, will
        modify the input `da_tar`. Using False will not, but `da_tar` will be
        copied and thus doubled in memory.
    rand_seed : int or None, default=0
        Random seed for reproducibility. Used for random sampling and model
        training.
    params : dict, optional
        LightGBM hyperparameter dictionary. Same as Scikit-Learn. If not
        provided, defaults to a LightGBM-based random forest setup.
    callbacks : list or None, optional
        Optional LightGBM callbacks (e.g., early stopping). See Scikit-Learn.

    Returns
    -------
    xr.DataArray
        Model-predicted target data with same dimensions as input `da_tar`.
    """

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

    arr_x, arr_y, arr_x_ev, arr_y_ev = extract_train_set(
        da_ref,
        da_tar,
        nodata,
        n_total_samples,
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
        callbacks
    )

    del arr_x, arr_y, arr_x_ev, arr_y_ev
    gc.collect()

    da_y = predict_models(
        da_ref,
        da_tar,
        models,
        nodata,
        predict_inplace
    )

    return da_y
