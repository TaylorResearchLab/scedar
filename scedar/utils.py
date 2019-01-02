import pickle
import multiprocessing as mp
import numpy as np
import gzip
import os
import warnings


def _parmap_fun(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))


def parmap(f, X, nprocs=1):
    """
    parmap_fun() and parmap() are adapted from klaus se's post
    on stackoverflow. https://stackoverflow.com/a/16071616/4638182

    parmap allows map on lambda and class static functions.

    Fall back to serial map when nprocs=1.
    """
    if nprocs < 1:
        raise ValueError("nprocs should be >= 1. nprocs: {}".format(nprocs))

    nprocs = min(int(nprocs), mp.cpu_count())
    # exception handling f
    # simply ignore all exceptions. If exception occurs in parallel queue, the
    # process with exception will get stuck and not be able to process
    # following requests.

    def ehf(x):
        try:
            res = f(x)
        except Exception as e:
            res = e
        return res
    # fall back on serial
    if nprocs == 1:
        return list(map(ehf, X))

    q_in = mp.Queue(1)
    q_out = mp.Queue()
    proc = [mp.Process(target=_parmap_fun, args=(ehf, q_in, q_out))
            for _ in range(nprocs)]

    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]
    # maintain the order of X
    ordered_res = [x for i, x in sorted(res)]
    for i, x in enumerate(ordered_res):
        if isinstance(x, Exception):
            warnings.warn("{} encountered in parmap {}th arg {}".format(
                x, i, X[i]))
    return ordered_res


def save_obj(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_gz_obj(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def dict_str_key(d):
    """
    Get a hash key for a dictionary, usually used for `**kwargs`.


    Examples
    --------
    >>> dict_str_key({"a": 1, "b": 2})
    "[('a', 1), ('b', 2)]"
    >>> dict_str_key({"b": 2, "a": 1})
    "[('a', 1), ('b', 2)]"

    Notes
    -----
    Non-string keys will be converted to strings before sorting, but the
    original value is preserved in the generated key.
    """
    if type(d) != dict:
        raise ValueError("d must be dictionary. {}".format(d))
    key_str_pair = [(k, str(k)) for k in d.keys()]
    sorted_key_str_pair = sorted(key_str_pair, key=lambda p: p[1])
    sorted_keys = map(lambda p: p[0], sorted_key_str_pair)
    return str([(k, d[k]) for k in sorted_keys])


def remove_constant_features(sfm):
    """
    Remove features that are constant across all samples
    """
    # boolean matrix of whether x == first column (feature)
    x_not_equal_to_1st_row = sfm._x != sfm._x[0]
    non_const_f_bool_ind = x_not_equal_to_1st_row.sum(axis=0) >= 1
    return sfm.ind_x(selected_f_inds=non_const_f_bool_ind)
