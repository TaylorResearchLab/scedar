import numpy as np
import scedar.eda as eda
import pytest


def test_is_valid_sfid():
    assert eda.mtype.is_valid_sfid('1')
    assert eda.mtype.is_valid_sfid(1)
    assert not eda.mtype.is_valid_sfid(np.array([1])[0])
    assert not eda.mtype.is_valid_sfid([])
    assert not eda.mtype.is_valid_sfid([1])
    assert not eda.mtype.is_valid_sfid(None)
    assert not eda.mtype.is_valid_sfid((1,))


def test_check_is_valid_sfids():
    with pytest.raises(Exception) as excinfo:
        eda.mtype.check_is_valid_sfids(np.arange(5))

    with pytest.raises(Exception) as excinfo:
        eda.mtype.check_is_valid_sfids([True, False])

    with pytest.raises(Exception) as excinfo:
        eda.mtype.check_is_valid_sfids(None)

    with pytest.raises(Exception) as excinfo:
        eda.mtype.check_is_valid_sfids([[1], [2]])

    with pytest.raises(Exception) as excinfo:
        eda.mtype.check_is_valid_sfids(['1', 2, 3])

    with pytest.raises(Exception) as excinfo:
        eda.mtype.check_is_valid_sfids(['1', '1', '3'])

    with pytest.raises(Exception) as excinfo:
        eda.mtype.check_is_valid_sfids([0, 0, 1])

    with pytest.raises(Exception) as excinfo:
        eda.mtype.check_is_valid_sfids(['1', 2, '3'])

    eda.mtype.check_is_valid_sfids([])
    eda.mtype.check_is_valid_sfids([1, 2])
    eda.mtype.check_is_valid_sfids(['1', '2'])
    eda.mtype.check_is_valid_sfids([1, 2, 3])


def test_is_valid_lab():
    assert eda.mtype.is_valid_lab('1')
    assert eda.mtype.is_valid_lab(1)
    assert not eda.mtype.is_valid_lab(np.array([1])[0])
    assert not eda.mtype.is_valid_lab([])
    assert not eda.mtype.is_valid_lab([1])
    assert not eda.mtype.is_valid_lab(None)
    assert not eda.mtype.is_valid_lab((1,))


def test_check_is_valid_labs():
    with pytest.raises(Exception) as excinfo:
        eda.mtype.check_is_valid_labs(np.arange(5))

    with pytest.raises(Exception) as excinfo:
        eda.mtype.check_is_valid_labs([True, False])

    with pytest.raises(Exception) as excinfo:
        eda.mtype.check_is_valid_labs(None)

    with pytest.raises(Exception) as excinfo:
        eda.mtype.check_is_valid_labs([[1], [2]])

    with pytest.raises(Exception) as excinfo:
        eda.mtype.check_is_valid_labs(['1', 2, 1])

    with pytest.raises(Exception) as excinfo:
        eda.mtype.check_is_valid_labs(['1', 2, '3'])

    eda.mtype.check_is_valid_labs([])
    eda.mtype.check_is_valid_labs([1])
    eda.mtype.check_is_valid_labs([1, 2])
    eda.mtype.check_is_valid_labs([1, 1, 3])
    eda.mtype.check_is_valid_labs(['1', '2', '3'])


def test_is_valid_full_cut_tree_mat():
    tfctm = np.array([[0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0],
                      [2, 1, 1, 1, 0],
                      [3, 2, 1, 1, 0],
                      [4, 3, 2, 0, 0]])
    assert eda.mtype.is_valid_full_cut_tree_mat(tfctm)
    tfctm_invalid = tfctm.copy()
    tfctm_invalid[4, 2] = 0
    assert not eda.mtype.is_valid_full_cut_tree_mat(tfctm_invalid)


def test_is_uniq_np1darr():
    assert not eda.mtype.is_uniq_np1darr([])
    assert not eda.mtype.is_uniq_np1darr([1])
    assert not eda.mtype.is_uniq_np1darr([1, 2])

    assert not eda.mtype.is_uniq_np1darr(())
    assert not eda.mtype.is_uniq_np1darr((1, 2))

    assert not eda.mtype.is_uniq_np1darr((1., 2))
    assert not eda.mtype.is_uniq_np1darr((1, 2))

    assert not eda.mtype.is_uniq_np1darr(np.array(['1', '1']))
    assert not eda.mtype.is_uniq_np1darr(np.array([1, 1]))
    assert not eda.mtype.is_uniq_np1darr(np.array([1, 1.]))
    assert not eda.mtype.is_uniq_np1darr(np.array([0, 1, 2, 1]))

    assert not eda.mtype.is_uniq_np1darr(np.array([1, 1]).reshape(2, 1))
    assert not eda.mtype.is_uniq_np1darr(np.array([[], []]))

    assert eda.mtype.is_uniq_np1darr(np.array([]))
    assert eda.mtype.is_uniq_np1darr(np.array([1, 2]))
    assert eda.mtype.is_uniq_np1darr(np.array([1, 3]))
    assert eda.mtype.is_uniq_np1darr(np.array(['1', '3']))
    assert eda.mtype.is_uniq_np1darr(np.array(['1']))
