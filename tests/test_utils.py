import scedar.utils as utils
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

def test_parmap_invalid_nprocs():
    with pytest.raises(ValueError) as excinfo:
        pm_res = utils.parmap(lambda x: x ** 2,
                              np.array([[1, 2], [3, 4]]), nprocs=0.5)

def test_parmap_gen_mp():
    n = 1000
    pm_res = utils.parmap(lambda x: x ** 2, range(n), nprocs=10)
    assert isinstance(pm_res, list)
    assert pm_res == list(map(lambda x: x**2, range(n)))

def test_parmap_exception_mp():
    n = 1000
    with pytest.warns(UserWarning, match='division by zero'):
        pm_res = utils.parmap(lambda x: x/0, range(n), nprocs=10)
    assert all(map(lambda x: isinstance(x, ZeroDivisionError), pm_res))


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


def test_dict_str_key():
    # empty dict should work
    assert utils.dict_str_key({}) == '[]'
    # general case
    assert utils.dict_str_key({"a": 1, "b": 2}) == "[('a', 1), ('b', 2)]"
    # same
    d1 = {'1': 2, '3': 'b', '5': [1,2,3]}
    d11 = {'1': 2, '3': 'b', '5': [1,2,3]}
    d2 = {'3': 'b', '1': 2, '5': [1,2,3]}
    assert utils.dict_str_key(d1) == utils.dict_str_key(d11)
    assert utils.dict_str_key(d1) == utils.dict_str_key(d2)
    # diff
    d3 = {'3': 'b', '1': 2, '5': (1,2,3)}
    d4 = {'3': 'b', '1': '2', '5': [1,2,3]}
    d5 = {'3': 'b', 1: 2, '5': [1,2,3]}
    assert utils.dict_str_key(d1) != utils.dict_str_key(d3)
    assert utils.dict_str_key(d1) != utils.dict_str_key(d4)
    assert utils.dict_str_key(d1) != utils.dict_str_key(d5)
    assert utils.dict_str_key(d4) != utils.dict_str_key(d5)
    assert utils.dict_str_key(d3) != utils.dict_str_key(d5)

def test_dict_str_key_wrong_arg():
    with pytest.raises(ValueError) as excinfo:
        utils.dict_str_key(1)
    with pytest.raises(ValueError) as excinfo:
        utils.dict_str_key('1')
    with pytest.raises(ValueError) as excinfo:
        utils.dict_str_key([1])
    with pytest.raises(ValueError) as excinfo:
        utils.dict_str_key((1, 2))
    with pytest.raises(ValueError) as excinfo:
        utils.dict_str_key(1.1)
