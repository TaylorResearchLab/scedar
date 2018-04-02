import numpy as np


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


def is_valid_full_cut_tree_mat(cmat):
    """
    Validate scipy hierarchical clustering cut tree
    Number of clusters should decrease from n to 1
    """
    col_unique_vals = [len(np.unique(x)) for x in cmat.T]
    return col_unique_vals == list(range(cmat.shape[0], 0, -1))


def is_valid_lab(lab):
    return (type(lab) == str) or (type(lab) == int)


def check_is_valid_labs(labs):
    if labs is None:
        raise ValueError("labs cannot be None")

    if type(labs) != list:
        raise ValueError("labs must be a homogenous list of int or str")

    n_uniq_types = len(set(map(type, labs)))
    if n_uniq_types > 1:
        raise ValueError("labs must be a homogenous list of int or str")
    elif n_uniq_types == 1:
        if not is_valid_lab(labs[0]):
            raise ValueError("labs must be a homogenous list of int or str")
    # At this point labs can either be an empty list or a list of ints/strs,
    # so it can only be 1d.
    labs = np.array(labs)


def is_valid_sfid(sfid):
    return (type(sfid) == str) or (type(sfid) == int)


def check_is_valid_sfids(sfids):
    if sfids is None:
        raise ValueError("[sf]ids cannot be None")

    if type(sfids) != list:
        raise ValueError("[sf]ids must be a homogenous list of int or str")

    n_uniq_types = len(set(map(type, sfids)))
    if n_uniq_types > 1:
        raise ValueError("[sf]ids must be a homogenous list of int or str")
    elif n_uniq_types == 1:
        if not is_valid_sfid(sfids[0]):
            raise ValueError("[sf]ids must be a homogenous list of int or str")
    # At this point sfids can either be an empty list or a list of ints/strs,
    # so it can only be 1d.
    sfids = np.array(sfids)
    if not is_uniq_np1darr(sfids):
        raise ValueError("[sf]ids must not contain duplicated values")
