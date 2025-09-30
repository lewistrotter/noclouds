
import numpy as np

def nodata_mask(
        arr: np.ndarray,
        nodata: int | float
) -> np.ndarray:

    if not np.isnan(nodata):
        return np.any(arr == nodata, axis=0)
    else:
        return np.any(np.isnan(arr), axis=0)


def _prepare_xgb_params(xgb_params: dict) -> dict:

    num_boost_round = xgb_params.pop('num_boost_round', None)

    if num_boost_round is None:
        num_boost_round = 100

    percent_train = xgb_params.pop('percent_train', None)
    split_seed = xgb_params.pop('split_seed', None)
    early_stopping_rounds = xgb_params.pop('early_stopping_rounds', None)
    verbose_eval = xgb_params.pop('verbose_eval', None)

    if percent_train is None:
        split_seed = None
        early_stopping_rounds = None
        verbose_eval = None

    extra_xgb_params = {
        'num_boost_round': num_boost_round,
        'percent_train': percent_train,
        'split_seed': split_seed,
        'early_stopping_rounds': early_stopping_rounds,
        'verbose_eval': verbose_eval
    }

    return extra_xgb_params


def default_xgb_params() -> dict:

    # TODO: determine optimal via optimum package

    return {
        'objective': 'reg:squarederror',
        'tree_method': 'hist',
        'learning_rate': 0.1,
        'max_depth': 8,
        'num_boost_round': 100,
        'percent_train': None,
        'split_seed': None,
        'early_stopping_rounds': None,
        'verbose_eval': None
    }

