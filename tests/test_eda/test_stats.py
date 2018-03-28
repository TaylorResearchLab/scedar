import numpy as np
import scedar.eda as eda
import pytest


def test_gc1d():
    assert eda.stats.gc1d([1]) == 0
    # order should not matter
    np.testing.assert_equal(eda.stats.gc1d([1, 0]), eda.stats.gc1d([0, 1]))
    np.testing.assert_equal(eda.stats.gc1d([1, 0, 0]), 
                            eda.stats.gc1d([0, 1, 0]))
    np.testing.assert_equal(eda.stats.gc1d([1, 0, 0]), 
                            eda.stats.gc1d([0, 0, 1]))
    # equal vals
    np.testing.assert_equal(eda.stats.gc1d([0, 0]), 0)
    np.testing.assert_equal(eda.stats.gc1d([0.1, 0.1]), 0)
    np.testing.assert_equal(eda.stats.gc1d([0.1, 0.1]), 0)
    np.testing.assert_equal(eda.stats.gc1d([0, 0, 0]), 0)
    np.testing.assert_equal(eda.stats.gc1d([1, 1]), 0)
    np.testing.assert_equal(eda.stats.gc1d([10, 10]), 0)
    # In the formula of unbiased estimator, [0, 0, ..., 0, n] will always 
    # get 1.
    np.testing.assert_equal(eda.stats.gc1d([0, 0, 10]), 1)
    np.testing.assert_equal(eda.stats.gc1d([0, 0, 5]), 1)
    assert eda.stats.gc1d([0, 0, 1]) > eda.stats.gc1d([0, 0, 0])
    # direction of change. More uneven -> higher gc
    assert eda.stats.gc1d([1, 1, 8]) > eda.stats.gc1d([1, 2, 7])
    assert eda.stats.gc1d([1, 1, 8]) < eda.stats.gc1d([1, 0, 9])
    with pytest.raises(ValueError) as excinfo:
        eda.stats.gc1d([])
    with pytest.raises(ValueError) as excinfo:
        eda.stats.gc1d(np.zeros((1, 1)))
