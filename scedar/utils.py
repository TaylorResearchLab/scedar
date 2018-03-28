import pickle
import multiprocessing as mp
import numpy as np
import gzip
import os


def _parmap_fun(f, q_in, q_out):
    def ehf(x):
        try:
            res = f(x)
        except Exception as e:
            res = e
        return res
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, ehf(x)))

def parmap(f, X, nprocs=1):
    """
    parmap_fun() and parmap() are obtained from klaus se's post
    on stackoverflow. <https://stackoverflow.com/a/16071616/4638182>
    
    parmap allows map on lambda and class static functions.
    """
    if nprocs < 1:
        raise ValueError("nprocs should be >= 1. nprocs: {}".format(nprocs))

    nprocs = min(int(nprocs), mp.cpu_count())
    
    q_in = mp.Queue(1)
    q_out = mp.Queue()
    proc = [mp.Process(target=_parmap_fun, args=(f, q_in, q_out))
            for _ in range(nprocs)]

    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [x for i, x in sorted(res)]


def is_valid_full_cut_tree_mat(cmat):
    """
    Validate scipy hierarchical clustering cut tree
    Number of clusters should decrease from n to 1
    """
    col_unique_vals = [len(np.unique(x)) for x in cmat.T]
    return col_unique_vals == list(range(cmat.shape[0], 0, -1))


def save_obj(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_gz_obj(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def is_uniq_np1darr(x):
    """Test whether x is a 1D np array that only contains unique values."""
    if not isinstance(x, np.ndarray):
        return False

    if not x.ndim == 1:
        return False

    uniqx = np.unique(x)
    if not uniqx.shape[0] == x.shape[0]:
        return False

    return True
