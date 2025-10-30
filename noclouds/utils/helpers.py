
import numpy as np
import numba as nb


def nodata_mask(
        arr: np.ndarray,
        nodata: int | float
) -> np.ndarray:

    if not np.isnan(nodata):
        return np.any(arr == nodata, axis=0)
    else:
        return np.any(np.isnan(arr), axis=0)


@nb.njit(inline='always')
def has_nodata_1d(
        arr: np.ndarray,
        nodata: int | float
):
    i_size = arr.shape[0]

    for i in range(i_size):
        v = arr[i]
        if v  == nodata or np.isnan(v):
            return True

    return False


@nb.njit(inline='always')
def has_nodata_3d(
        arr: np.ndarray,
        nodata: int | float
):
    b_size, y_size, x_size = arr.shape

    for b in range(b_size):
        for y in range(y_size):
            for x in range(x_size):
                v = arr[b, y, x]
                if v == nodata or np.isnan(v):
                    return True

    return False


def default_params() -> dict:

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

    return params
