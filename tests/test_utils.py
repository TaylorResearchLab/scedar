import scxplit.utils as utils
import numpy as np
import os
import shutil
import gzip
import pytest


def test_parmap_tup():
    pm_res = utils.parmap(lambda x: x ** 2, (1, 2, 3))
    assert isinstance(pm_res, list)
    assert pm_res == [1, 4, 9]


def test_parmap_lst():
    pm_res = utils.parmap(lambda x: x ** 2, [1, 2, 3])
    assert isinstance(pm_res, list)
    assert pm_res == [1, 4, 9]


def test_parmap_gen():
    pm_res = utils.parmap(lambda x: x ** 2, range(1, 4))
    assert isinstance(pm_res, list)
    assert pm_res == [1, 4, 9]


def test_parmap_arr1d():
    pm_res = utils.parmap(lambda x: x ** 2, np.array([1, 2, 3]))
    assert isinstance(pm_res, list)
    assert pm_res == [1, 4, 9]


def test_parmap_arr2d():
    pm_res = utils.parmap(lambda x: x ** 2, np.array([1, 2, 3]).reshape(3, 1))
    assert isinstance(pm_res, list)
    assert pm_res == [1, 4, 9]


def test_parmap_arr2d():
    pm_res = utils.parmap(lambda x: x ** 2, 
                          np.array([[1, 2], [3, 4]]))
    assert isinstance(pm_res, list)
    assert np.all(pm_res[0] == np.array([1, 4]))
    assert np.all(pm_res[1] == np.array([9, 16]))


def test_parmap_gen_mp():
    n = 1000
    pm_res = utils.parmap(lambda x: x ** 2, range(n), nprocs=10)
    assert isinstance(pm_res, list)
    assert pm_res == list(map(lambda x: x**2, range(n)))


def test_save_obj(tmpdir):
    def test_helper(x, is_np_arr):
        p = os.path.join(tmpdir.strpath, 'tmp_obj.pkl')

        utils.save_obj(x, p)
        y = utils.load_obj(p)
        if is_np_arr:
            assert np.all(x == y)
        else:
            assert x == y

        gzp = p + '.gz'
        with open(p, 'rb') as f_in:
            with gzip.open(gzp, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        gzy = utils.load_gz_obj(gzp)
        if is_np_arr:
            assert np.all(x == gzy)
        else:
            assert x == gzy

    test_helper(np.arange(100), True)
    test_helper(np.arange(100).reshape(10, 10), True)
    test_helper(list(range(10)), False)
    test_helper(tuple(range(10)), True)
    test_helper(dict(zip(range(10), range(1, 11))), True)


def test_is_valid_full_cut_tree_mat():
    tfctm = np.array([[0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0],
                      [2, 1, 1, 1, 0],
                      [3, 2, 1, 1, 0],
                      [4, 3, 2, 0, 0]])
    assert utils.is_valid_full_cut_tree_mat(tfctm)
    tfctm_invalid = tfctm.copy()
    tfctm_invalid[4, 2] = 0
    assert not utils.is_valid_full_cut_tree_mat(tfctm_invalid)


def test_is_uniq_np1darr():
    assert not utils.is_uniq_np1darr([])
    assert not utils.is_uniq_np1darr([1])
    assert not utils.is_uniq_np1darr([1, 2])

    assert not utils.is_uniq_np1darr(())
    assert not utils.is_uniq_np1darr((1, 2))

    assert not utils.is_uniq_np1darr((1., 2))
    assert not utils.is_uniq_np1darr((1, 2))

    assert not utils.is_uniq_np1darr(np.array(['1', '1']))
    assert not utils.is_uniq_np1darr(np.array([1, 1]))
    assert not utils.is_uniq_np1darr(np.array([1, 1.]))
    assert not utils.is_uniq_np1darr(np.array([0, 1, 2, 1]))

    assert not utils.is_uniq_np1darr(np.array([1, 1]).reshape(2, 1))
    assert not utils.is_uniq_np1darr(np.array([[], []]))

    assert utils.is_uniq_np1darr(np.array([]))
    assert utils.is_uniq_np1darr(np.array([1, 2]))
    assert utils.is_uniq_np1darr(np.array([1, 3]))
    assert utils.is_uniq_np1darr(np.array(['1', '3']))
    assert utils.is_uniq_np1darr(np.array(['1']))
