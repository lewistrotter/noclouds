# region Training data creation

def _split_samples_per_chunk(
        arr: dk.Array,
        n_samples: int | None
) -> tuple:

    if arr.ndim != 3:
        raise ValueError('Input arr must be 3D.')

    y_chunks = np.array(arr.chunks[1])
    x_chunks = np.array(arr.chunks[2])

    # broadcast to cross product each y vs x
    chunk_sizes = (y_chunks[:, None] * x_chunks[None, :]).ravel()

    if n_samples is None:
        return tuple([int(c) for c in chunk_sizes])

    # proportion per chunk to total size
    chunk_fracs = chunk_sizes / np.sum(chunk_sizes)
    n_chunk_samples = np.floor(chunk_fracs * n_samples)
    n_chunk_samples = n_chunk_samples.astype(np.int64)

    # add any lost from floors to largest chunk
    remain = n_samples - np.sum(n_chunk_samples)
    if remain > 0:
        for i in np.argsort(-chunk_sizes)[:remain]:
            n_chunk_samples[i] += 1

    return tuple([int(c) for c in n_chunk_samples])


def _calc_valid_per_chunk(
        arr_ref: dk.Array,
        arr_tar: dk.Array,
        nodata: int | float
) -> int:

    arr_mask = _make_train_mask(
        arr_ref,
        arr_tar,
        nodata
    )

    return arr_mask.sum()


def _validate_chunk_samples(
        n_valid: tuple,
        n_total: tuple
) -> tuple:

    if len(n_valid) != len(n_total):
        raise ValueError('Inputs n_valid and n_total must be same size.')

    n_samples = []
    for v, t in zip(n_valid, n_total):
        if v > t:
            v = t
        n_samples.append(v)

    return tuple(n_samples)


def extract_train_set(
        arr_ref: dk.Array,
        arr_tar: dk.Array,
        nodata: int | float,
        n_samples: int | None = None,
        rand_seed: int | None = None
) -> tuple[dk.Array, dk.Array]:
    """TODO"""

    if not isinstance(arr_ref, dk.Array):
        raise TypeError('Input arr_ref must be type dask.array.Array.')
    if not isinstance(arr_tar, dk.Array):
        raise TypeError('Input arr_tar must be type dask.array.Array.')

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

    r_delays = overlap(
        arr_ref,
        depth=(0, 1, 1),
        #depth={1: 1, 2: 1},
        boundary=nodata
    ).to_delayed().ravel()

    t_delays = overlap(
        arr_tar,
        depth=(0, 1, 1),
        #depth={1: 1, 2: 1},
        boundary=nodata
    ).to_delayed().ravel()

    c_delays = [
        dask.delayed(_calc_valid_per_chunk)(r, t, nodata)
        for r, t in zip(r_delays, t_delays)
    ]

    with ProgressBar():

        #with performance_report(filename="dask-report-t8-norechunk.html"):
        n_valid_per_chunk = dask.compute(*c_delays)




    # stratify samples only if random sampling required, else all valid
    if n_samples is not None:
        n_total_per_chunk = _split_samples_per_chunk(
            arr_ref,
            n_samples
        )

        n_chunk_samples = _validate_chunk_samples(
            n_valid_per_chunk,
            n_total_per_chunk
        )
    else:
        n_chunk_samples = n_valid_per_chunk

    n_r_vars = arr_ref.shape[0] * 9  # 9 pix per var per win
    n_t_vars = arr_tar.shape[0]

    arr_x, arr_y = [], []
    for i, (r, t, c) in enumerate(zip(r_delays, t_delays, n_chunk_samples)):
        delay = dask.delayed(_extract_train_xy, name=f'etyx-{i}')(
            r, t, nodata, c, rand_seed
        )

        arr_x.append(
            dk.from_delayed(
                delay[0],
                shape=(c, n_r_vars),
                dtype=arr_ref.dtype
            )
        )

        # as above but for X output only
        arr_y.append(
            dk.from_delayed(
                delay[1],
                shape=(c, n_t_vars),
                dtype=arr_tar.dtype
            )
        )

    arr_x = dk.concatenate(arr_x, axis=0)
    arr_y = dk.concatenate(arr_y, axis=0)

    return arr_x, arr_y

# endregion

def run(
        da_ref: xr.DataArray,
        da_tar: xr.DataArray,
        nodata: int | float,
        n_samples: int | None = None,
        rand_seed: int = 0,
        allow_persist: bool = False,
        xgb_params: dict = None,
        client = None
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

    #if client is None:
        #raise ValueError('Must provide a dask client.')

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

    #if allow_persist:
        ...
    arr_x, arr_y = dask.compute(arr_x, arr_y)
    #wait([arr_x, arr_y])  # TODO: figure out why this fails in build_xgb_models. mem issues?

    #arr_x_out, arr_y_out = dask.persist(arr_x, arr_y)
    #wait([arr_x_out, arr_y_out])

    #xgb_models = build_xgb_models(arr_x, arr_y, xgb_params)

    # arr_i, arr_x = extract_predict_set(
    #     arr_ref,
    #     arr_tar,
    #     nodata
    # )

    # if arr_i.size == 0 or arr_x.size == 0:
    #     raise ValueError('No prediction pixels could be extracted.')

    # arr_y = predict_xgb_models(
    #     arr_i,
    #     arr_x,
    #     xgb_models
    # )

    # arr_y = fill_predict_iy(
    #     arr_i,
    #     arr_y,
    #     arr_tar,
    #     predict_inplace
    # )

    # da_out = xr.DataArray(
    #     arr_y,
    #     dims=da_tar.dims,
    #     coords=da_tar.coords,
    #     attrs=da_tar.attrs
    # )

    #return da_out

    return


def _working_train_test_split():
    # chunk_counts = arr_x.chunks[0]
    #
    # total_current = np.sum(chunk_counts)
    # total_target = int(total_current * percent_train)
    #
    # # Compute probabilities proportional to current counts
    # probs = chunk_counts / total_current
    #
    # # Sample new counts from a multinomial distribution
    # chunk_train_sizes = np.random.multinomial(total_target, probs)
    #
    # # get the test sizes now too
    # chunk_test_sizes = chunk_counts - chunk_train_sizes

    def _tmp(
            block: np.ndarray,
            n_train_samples: int | None = None,
            rand_seed: int = 0
    ):
        np.random.seed(rand_seed)

        n_rows = block.shape[0]

        arr_i = np.arange(n_rows)
        np.random.shuffle(arr_i)

        arr_train_i = arr_i[:n_train_samples]
        arr_test_i = arr_i[n_train_samples:]

        block_train = block[arr_train_i, :]
        block_test = block[arr_test_i, :]

        return block_train, block_test

    rand_seed = 0

    # delays = [
    #     dk.from_delayed(
    #         dask.delayed(_tmp)
    #         (b, n, rand_seed),
    #         shape=(n, 9),  # always 9 pix win
    #         dtype=arr_x.dtype
    #     )
    #     for b, n in zip(
    #         arr_x.to_delayed().ravel(),
    #         chunk_train_sizes,
    #     )
    # ]

    # delays = arr_x.to_delayed().ravel()
    #
    # delays_train = []
    # delays_test = []
    # for b, n_train, n_test in zip(delays, chunk_train_sizes, chunk_test_sizes):
    #     delay = dask.delayed(_tmp)(b, n_train, rand_seed)
    #
    #     delays_train.append(
    #         dk.from_delayed(
    #             delay[0],
    #             shape=(n_train, 9),
    #             dtype=arr_x.dtype
    #         )
    #     )
    #     delays_test.append(
    #         dk.from_delayed(
    #             delay[1],
    #             shape=(n_test, 9),
    #             dtype=arr_x.dtype
    #         )
    #     )
    #
    # arr_x_train = np.vstack([*delays_train])
    # arr_x_test = np.vstack([*delays_test])

    # with ProgressBar():
    #     arr_x_train, arr_x_test = dask.compute(arr_x_train, arr_x_test)

    # arr_x_ev, arr_y_ev = None, None
    # if percent_train:
    #     arr_x, arr_x_ev, arr_y, arr_y_ev = _split_train_test_xy(
    #         arr_x,
    #         arr_y,
    #         percent_train,
    #         split_seed
    #     )