import pickle
import multiprocessing as mp
import numpy as np
import gzip
import os

#
# !!! parmap_fun() and parmap() are obtained from klaus se's post 
# on stackoverflow. !!!
# <https://stackoverflow.com/a/16071616/4638182>
# parmap allows map on lambda and class static functions. 
# 
def _parmap_fun(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))


def parmap(f, X, nprocs=1):
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


# numerically correct dmat
def num_correct_dist_mat(dmat, upper_bound = None):
    assert dmat.shape[0] == dmat.shape[1]
    
    dmat[dmat < 0] = 0
    dmat[np.diag_indices(dmat.shape[0])] = 0
    if upper_bound:
        dmat[dmat > upper_bound] = upper_bound
    
    dmat[np.triu_indices_from(dmat)] = dmat.T[np.triu_indices_from(dmat)]
    return dmat


# Validate scipy hierarchical clustering cut tree
# Number of clusters should decrease from n to 1
def is_valid_full_cut_tree_mat(cmat):
    col_unique_vals = [len(np.unique(x)) for x in cmat.T]
    return col_unique_vals == list(range(cmat.shape[0], 0, -1))


def save_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_gz_obj(path):
    with gzip.open(path, 'rb') as f:
        return pickle.load(f)
