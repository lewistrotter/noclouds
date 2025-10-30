

import os
import time
import numpy as np
import numba as nb

import xarray as xr
import rioxarray

from noclouds.utils.helpers import nodata_mask
from noclouds.utils.helpers import has_nodata_1d


@nb.njit(inline='always')
def _is_all_nan(arr: np.ndarray) -> bool:
    """Numba efficient check if all 1d array values are nan or not."""

    for i in range(arr.shape[0]):
        if not np.isnan(arr[i]):
            return False

    return True





@nb.njit
def winreg(
        win: np.ndarray,
        t_cen: int = 2,
        r_cen: int = 5,
        m: int = 2,
        v_min: int | float = 1,
        v_max: int | float = 10000,
        r2_min: float = 0.90
) -> float:

    # extract target vector
    arr_tgt = win[:, r_cen, r_cen]

    # check if target vector all nan
    if _is_all_nan(arr_tgt):
        return np.nan

    # extract win shape
    win_t_size, win_y_size, win_x_size = win.shape

    # find best matching candidate based on linreg absolute r-squared
    b_abs_r2 = 0.0
    b_yi = b_xi = -1
    for yi in nb.prange(win_y_size):
        for xi in nb.prange(win_x_size):

            # skip target yi, xi index
            if yi == r_cen and xi == r_cen:
                continue

            # extract candidate vector, skip if no target-time candidate
            arr_can = win[:, yi, xi]
            if np.isnan(arr_can[t_cen]):
                continue

            # calc sums of valid pixels, skip if inadequate num valid
            sums = _compute_sums(arr_tgt, arr_can)
            n_valid, sum_x, sum_y, sum_xx, sum_yy, sum_xy = sums
            if n_valid < m:
                continue

            # calc intercept and absolute r-squared, select if best yet
            abs_r2 = _compute_abs_r2(n_valid, sum_x, sum_y, sum_xx, sum_yy, sum_xy)
            if ~np.isnan(abs_r2) and abs_r2 > b_abs_r2:
                b_yi, b_xi = yi, xi
                b_abs_r2 = abs_r2

    # leave if best model was inadequate or no indices found
    if b_abs_r2 < r2_min or b_yi < 0 or b_xi < 0:
        return np.nan

    # extract the best candidate vector and target time pixel
    arr_can = win[:, b_yi, b_xi]
    y_tgt = arr_can[t_cen]

    # calc valid-only sums, slope, int, r2
    sums = _compute_sums(arr_can, arr_tgt)
    n_valid, sum_x, sum_y, sum_xx, sum_yy, sum_xy = sums
    slope = _compute_slope(n_valid, sum_x, sum_y, sum_xx, sum_xy)
    intercept = _compute_intercept(n_valid, sum_x, sum_y, slope)
    abs_r2 = _compute_abs_r2(n_valid, sum_x, sum_y, sum_xx, sum_yy, sum_xy)

    # model target using candidate value and model
    y_prd = slope * y_tgt + intercept

    # reject if prediction outside range
    if y_prd < v_min or y_prd > v_max:
        return np.nan

    return y_prd


@nb.njit(parallel=True)
def apply_winreg(
        arr: np.ndarray,
        t_cen: int = 2,
        r_cen: int = 5,
        m: int = 2,
        v_min: int | float = 1,
        v_max: int | float = 10000,
        r2_min: float = 0.90,
        max_iters: int = 10
) -> np.ndarray:

    _, y_size, x_size = arr.shape

    # calc initial total nans
    arr_total_nans = np.sum(np.isnan(arr[t_cen, :, :]))
    arr_remain_nans = arr_total_nans

    for _ in range(max_iters):
        for yi in nb.prange(r_cen, y_size - r_cen):
            for xi in range(r_cen, x_size - r_cen):

                # only apply algo if target is missing
                if np.isnan(arr[t_cen, yi, xi]):
                    win = arr[:, yi - r_cen:yi + r_cen + 1, xi - r_cen:xi + r_cen + 1]
                    arr[t_cen, yi, xi] = winreg(
                        win,
                        t_cen,
                        r_cen,
                        m,
                        v_min,
                        v_max,
                        r2_min
                    )

        # if nan count same, we've hit the end
        if arr_remain_nans == np.isnan(arr[t_cen, :, :]).sum():
            return arr

        # update nan count with latest iteration
        arr_remain_nans = np.sum(np.isnan(arr[t_cen, :, :]))

    return arr









# @nb.njit(inline='always')
def _compute_sums(
        arr_x: np.ndarray,
        arr_y: np.ndarray
) -> tuple:
    """
    Compute the required sums for linear regression.
    Parameters
    ----------
    x : 1D np.ndarray of floats
        Independent variable values
    y : 1D np.ndarray of floats
        Dependent variable values

    Returns
    -------
    sum_x : float
        Sum of x values
    sum_y : float
        Sum of y values
    sum_xx : float
        Sum of x squared
    sum_yy : float
        Sum of y squared
    sum_xy : float
        Sum of x*y
    """

    n_valid = 0
    sum_x = sum_y = sum_xx = sum_yy = sum_xy = 0.0

    for i in range(arr_x.shape[0]):
        if ~np.isnan(arr_x[i]) & ~np.isnan(arr_y[i]):
            x_val = arr_x[i]
            y_val = arr_y[i]

            sum_x += x_val           # add to Σ x
            sum_y += y_val           # add to Σ y
            sum_xx += x_val * x_val  # add to Σ x^2
            sum_yy += y_val * y_val  # add to Σ y^2
            sum_xy += x_val * y_val  # add to Σ (x*y)

            n_valid += 1

    return n_valid, sum_x, sum_y, sum_xx, sum_yy, sum_xy


# @nb.njit(inline='always')
def _compute_slope(
        n_samp: int,
        sum_x: float,
        sum_y: float,
        sum_xx: float,
        sum_xy: float
) -> float:
    """
    Compute slope of linear regression line.
    Parameters
    ----------
    n_samp : int
        Number of samples
    sum_x, sum_y, sum_xx, sum_xy : float
        Precomputed sums

    Returns
    -------
    slope : float
        Slope of the regression line. NaN if denominator is zero.
    """

    denominator = n_samp * sum_xx - sum_x * sum_x
    if denominator == 0.0:
        return np.nan

    slope = (n_samp * sum_xy - sum_x * sum_y) / denominator

    return slope


# @nb.njit(inline='always')
def _compute_intercept(
        n_samp: int,
        sum_x: float,
        sum_y: float,
        slope: float
) -> float:
    """
    Compute intercept of linear regression line.

    Parameters
    ----------
    n_samp : int
        Number of samples
    sum_x, sum_y : float
        Precomputed sums
    slope : float
        Regression slope

    Returns
    -------
    intercept : float
        Intercept of the regression line
    """

    intercept = (sum_y - slope * sum_x) / n_samp

    return intercept


# @nb.njit(inline='always')
def _compute_abs_r2(
        n_samp: int,
        sum_x: float,
        sum_y: float,
        sum_xx: float,
        sum_yy: float,
        sum_xy: float
) -> float:
    """
    Compute coefficient of determination (R^2) for linear regression.

    Parameters
    ----------
    n_samp : int
        Number of samples
    sum_x, sum_y, sum_xx, sum_yy, sum_xy : float
        Precomputed sums

    Returns
    -------
    r2 : float
        R^2 value, NaN if denominator is zero
    """

    r_numerator = n_samp * sum_xy - sum_x * sum_y

    r_denominator = np.sqrt(
        (n_samp * sum_xx - sum_x * sum_x) *
        (n_samp * sum_yy - sum_y * sum_y)
    )

    if r_denominator == 0.0:
        return np.nan

    r2 = (r_numerator / r_denominator) ** 2
    abs_r2 = abs(r2)

    return abs_r2


@nb.njit
def _temporal_slice(
    arr: np.ndarray,
    temporal_depth: int,
) -> np.ndarray:

    total_dates = arr.shape[0]

    ti_l = (total_dates // 2 - temporal_depth)
    ti_r = (total_dates // 2 + temporal_depth + 1)

    return arr[ti_l:ti_r]


#@nb.njit(parallel=True)
def _mwr_old(
        arr_ts: np.ndarray,
        nodata: int | float,
        temporal_depth: int,
        space_depth: int,
        min_train_pairs: int,
        min_rsquared: float,
        max_value_range: tuple,
        predict_inplace: bool
) -> np.ndarray:

    twi, swi = temporal_depth, space_depth
    v_min, v_max = max_value_range

    arr = _temporal_slice(arr_ts, twi)
    t_size, y_size, x_size = arr.shape

    arr_out = arr[twi, :, :]
    if not predict_inplace:
        arr_out = arr_out.copy()

    for yi in nb.prange(swi, y_size - swi):
        for xi in range(swi, x_size - swi):

            val = arr[twi, yi, xi]
            if (val != nodata) and (~np.isnan(val)):
                continue

            arr_win = arr[:, yi - swi:yi + swi + 1, xi - swi:xi + swi + 1]
            arr_mask = (arr_win == nodata) | np.isnan(arr_win)

            arr_mask_cen = arr_mask[:, swi, swi]
            if np.sum(~arr_mask_cen) < min_train_pairs:
                continue

            arr_cen = arr_win[:, swi, swi]

            _, win_y_size, win_x_size = arr_win.shape

            best_abs_r2 = 0.0
            best_wyi = best_wxi = -1
            for wyi in range(win_y_size):
                for wxi in range(win_x_size):

                    # skip target yi, xi index
                    if wyi == swi and wxi == swi:
                        continue

                    arr_can = arr_win[:, wyi, wxi]

                    val = arr_can[twi]
                    if (val == nodata) or (np.isnan(val)):
                        continue

                    # TODO: remove pairwise nodata

                    if not np.isnan(nodata):
                        arr_idx_mask = (arr_can != nodata) & (arr_cen != nodata)
                    else:
                        arr_idx_mask = (~np.isnan(arr_can)) & (~np.isnan(arr_cen))

                    arr_cen_sel = arr_cen[arr_idx_mask]
                    arr_can_sel = arr_can[arr_idx_mask]

                    # calc sums of valid pixels
                    sums = _compute_sums(arr_cen_sel, arr_can_sel)
                    n_valid, sum_x, sum_y, sum_xx, sum_yy, sum_xy = sums
                    if n_valid < min_train_pairs:
                        continue

                    # calc intercept and absolute r-squared, select if best yet
                    abs_r2 = _compute_abs_r2(n_valid, sum_x, sum_y, sum_xx, sum_yy, sum_xy)
                    if (abs_r2 > best_abs_r2) and (~np.isnan(abs_r2)):
                        best_wyi, best_wxi = wyi, wxi
                        best_abs_r2 = abs_r2

            # leave if best model was inadequate or no indices found
            if (best_abs_r2 < min_rsquared) or (best_wyi < 0) or (best_wxi < 0):
                continue

            # extract the best candidate vector and target time pixel
            arr_can = arr_win[:, best_wyi, best_wxi]
            y = arr_can[twi]


            if not np.isnan(nodata):
                arr_idx_mask = (arr_can != nodata) & (arr_cen != nodata)
            else:
                arr_idx_mask = (~np.isnan(arr_can)) & (~np.isnan(arr_cen))

            arr_cen_sel = arr_cen[arr_idx_mask]
            arr_can_sel = arr_can[arr_idx_mask]


            # calc valid-only sums, slope, int, r2
            sums = _compute_sums(arr_can_sel, arr_cen_sel)
            n_valid, sum_x, sum_y, sum_xx, sum_yy, sum_xy = sums
            slope = _compute_slope(n_valid, sum_x, sum_y, sum_xx, sum_xy)
            intercept = _compute_intercept(n_valid, sum_x, sum_y, slope)
            abs_r2 = _compute_abs_r2(n_valid, sum_x, sum_y, sum_xx, sum_yy, sum_xy)

            # model target using candidate value and model
            y_pred = slope * y + intercept

            # reject if prediction outside range
            if (y_pred < v_min) or (y_pred > v_max):
                continue

            arr_out[yi, xi] = y_pred

    return arr_out



@nb.njit
def linreg(
        x: np.ndarray,
        y: np.ndarray
):

    n = x.shape[0]

    # Cast to float for precision
    xf = x#.astype(np.float64)
    yf = y#.astype(np.float64)

    sum_x = 0.0
    sum_y = 0.0
    sum_xy = 0.0
    sum_x2 = 0.0
    sum_y2 = 0.0

    for i in range(n):
        sum_x += xf[i]
        sum_y += yf[i]
        sum_xy += xf[i] * yf[i]
        sum_x2 += xf[i] * xf[i]
        sum_y2 += yf[i] * yf[i]

    denom = n * sum_x2 - sum_x * sum_x
    if denom == 0:
        # all x identical - regression undefined
        return np.nan, np.nan, np.nan

    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n

    # compute r-squared
    ss_tot = 0.0
    ss_res = 0.0
    for i in range(n):
        y_pred = slope * xf[i] + intercept
        ss_res += (yf[i] - y_pred) ** 2
        ss_tot += (yf[i] - (sum_y / n)) ** 2

    r2 = 1.0 - (ss_res / ss_tot if ss_tot != 0 else np.nan)

    return slope, intercept, r2







@nb.njit(parallel=True)
def _mwr(
        arr_ts: np.ndarray,
        nodata: int | float,
        temporal_depth: int,
        space_depth: int,
        min_train_pairs: int,
        min_rsquared: float,
        max_value_range: tuple,
        predict_inplace: bool
) -> np.ndarray:

    twi, swi = temporal_depth, space_depth
    v_min, v_max = max_value_range

    arr = _temporal_slice(arr_ts, twi)
    t_size, y_size, x_size = arr.shape

    arr_out = arr[twi, :, :]
    if not predict_inplace:
        arr_out = arr_out.copy()

    for yi in nb.prange(swi, y_size - swi):
        for xi in range(swi, x_size - swi):

            val = arr[twi, yi, xi]
            if (val != nodata) and (~np.isnan(val)):
                continue

            arr_win = arr[:, yi - swi:yi + swi + 1, xi - swi:xi + swi + 1]
            arr_win_mask = (arr_win == nodata) | np.isnan(arr_win)

            arr_mask_cen = arr_win_mask[:, swi, swi]
            if np.sum(~arr_mask_cen) < min_train_pairs:
                continue

            arr_cen = arr_win[:, swi, swi]

            _, win_y_size, win_x_size = arr_win.shape

            best_abs_r2 = 0.0
            best_wyi = -1
            best_wxi = -1
            for wyi in range(win_y_size):
                for wxi in range(win_x_size):

                    if wyi == swi and wxi == swi:
                        continue  # skip target yi, xi index

                    arr_mask_can = arr_win_mask[:, wyi, wxi]
                    if arr_mask_can[twi]:
                        continue

                    arr_mask_valid_pairs = (~arr_mask_cen) & (~arr_mask_can)
                    if np.sum(arr_mask_valid_pairs) < min_train_pairs:
                        continue

                    arr_can = arr_win[:, wyi, wxi]

                    arr_cen_sel = arr_cen[arr_mask_valid_pairs]
                    arr_can_sel = arr_can[arr_mask_valid_pairs]

                    slope, intercept, r2 = linreg(arr_cen_sel, arr_can_sel)

                    abs_r2 = np.abs(r2)
                    if (abs_r2 > best_abs_r2) and (~np.isnan(abs_r2)):
                        best_wyi, best_wxi = wyi, wxi
                        best_abs_r2 = abs_r2

            # leave if best model was inadequate or no indices found
            if (best_abs_r2 < min_rsquared) or (best_wyi < 0) or (best_wxi < 0):
                continue

            # extract the best candidate vector and target time pixel

            arr_can = arr_win[:, best_wyi, best_wxi]
            arr_mask_can = arr_win_mask[:, best_wyi, best_wxi]

            arr_mask_valid_pairs = (~arr_mask_cen) & (~arr_mask_can)

            arr_cen_sel = arr_cen[arr_mask_valid_pairs]
            arr_can_sel = arr_can[arr_mask_valid_pairs]

            slope, intercept, r2 = linreg(arr_can_sel , arr_cen_sel)  # ORDER CORRECT?

            y = arr_can[twi]
            y_pred = slope * y + intercept

            # reject if prediction outside range
            if (y_pred < v_min) or (y_pred > v_max):
                continue

            arr_out[yi, xi] = y_pred

    return arr_out





def run(
    da_ts: xr.DataArray,
    nodata: int | float,
    temporal_depth: int | None = 5,  # None uses all time
    space_depth: int = 5,
    min_train_pairs: int = 2,
    min_rsquared: float = 0.6,
    max_value_range: tuple | None = None,  # use data itself?
    predict_inplace:bool = True,
    max_iters: int | None = 3  # None uses smart end using convergence
) -> xr.DataArray:

    if not isinstance(da_ts, xr.DataArray):
        raise TypeError('Input da_ts must be type xr.DataArray.')

    if da_ts.ndim != 3:
        raise ValueError('Input da_ts must be 3D (t, y, x).')

    # TODO: consider per-var loop

    # TODO: other checks

    if nodata is None:
        raise ValueError('Input nodata must be provided.')

    arr_ts = da_ts.data

    n_dts = arr_ts.shape[0]
    if n_dts < 3 :
        raise ValueError('Input da_ts must have >= 3 dates.')
    if n_dts % 2 != 1:
        raise ValueError('Input da_ts must have an odd number of dates.')

    if (2 * temporal_depth + 1) > n_dts:
        raise ValueError('Input temporal_depth > number of da_ts dates.')

    if temporal_depth is None:
        temporal_depth = n_dts // 2  # use all dates

    da_ts[[3], :, :].rio.to_raster('orig.tif')

    arr_out = _mwr(
        arr_ts,
        nodata,
        temporal_depth,
        space_depth,
        min_train_pairs,
        min_rsquared,
        max_value_range,
        predict_inplace
    )

    da_out = xr.DataArray(
        arr_out,
        dims=(da_ts.dims[1], da_ts.dims[2]),
        coords={'y': da_ts.coords['y'], 'x': da_ts.coords['x']},
        attrs=da_ts.attrs
    )

    da_out.rio.to_raster('mwr.tif')

    return

