import numpy as np
from .. import utils

def is_valid_lab(lab):
    return (type(lab) == str) or (type(lab) == int)


def check_is_valid_labs(labs):
    if labs is None:
        raise ValueError("labs cannot be None")

    if type(labs) != list:
        raise ValueError("labs must be a homogenous list of int or str")

    if len(labs) == 0:
        raise ValueError("labs cannot be empty")

    if len(set(map(type, labs))) != 1:
        raise ValueError("labs must be a homogenous list of int or str")

    if not is_valid_lab(labs[0]):
        raise ValueError("labs must be a homogenous list of int or str")

    labs = np.array(labs)
    assert labs.ndim == 1, "Labels must be 1D"
    assert labs.shape[0] > 0


def is_valid_sfid(sfid):
    return (type(sfid) == str) or (type(sfid) == int)


def check_is_valid_sfids(sfids):
    if sfids is None:
        raise ValueError("[sf]ids cannot be None")

    if type(sfids) != list:
        raise ValueError("[sf]ids must be a homogenous list of int or str")

    if len(sfids) == 0:
        raise ValueError("[sf]ids must have >= 1 values")

    sid_types = tuple(map(type, sfids))
    if len(set(sid_types)) != 1:
        raise ValueError("[sf]ids must be a homogenous list of int or str")

    if not is_valid_sfid(sfids[0]):
        raise ValueError("[sf]ids must be a homogenous list of int or str")

    sfids = np.array(sfids)
    assert sfids.ndim == 1
    assert sfids.shape[0] > 0
    if not utils.is_uniq_np1darr(sfids):
        raise ValueError("[sf]ids must not contain duplicated values")
