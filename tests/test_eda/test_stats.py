import numpy as np
import scedar.eda as eda
import pytest


def test_gc1d():
    assert np.isnan(eda.stats.gc1d([1]))
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


def test_multiple_testing_correction():
    pvals = [0.0, 0.01, 0.029, 0.03, 0.031, 0.05,
             0.069, 0.07, 0.071, 0.09, 0.1]
    bonf = eda.stats.multiple_testing_correction(pvals, 'Bonferroni')
    fdr = eda.stats.multiple_testing_correction(pvals, 'FDR')
    # values obtained using R p.adjust
    np.testing.assert_allclose(bonf, [0, 0.11, 0.319, 0.33, 0.341, 0.55,
                                      0.759, 0.77, 0.781, 0.99, 1])
    np.testing.assert_allclose(fdr, [0, 0.055, 0.0682, 0.0682, 0.0682,
                                     0.0867777777777778, 0.0867777777777778,
                                     0.0867777777777778, 0.0867777777777778,
                                     0.099, 0.1])
    with pytest.raises(ValueError) as excinfo:
        eda.stats.multiple_testing_correction(pvals, '123')


def test_bidir_ReLU():
    np.testing.assert_allclose(eda.stats.bidir_ReLU(0, 0, 10, 0, 1), 0)
    np.testing.assert_allclose(
        eda.stats.bidir_ReLU(-1, 0, 10, 0, 1), 0)
    np.testing.assert_allclose(
        eda.stats.bidir_ReLU(-10, 0, 10, 0, 1), 0)
    np.testing.assert_allclose(
        eda.stats.bidir_ReLU(1, 0, 10, 0, 1), 1/10)
    np.testing.assert_allclose(
        eda.stats.bidir_ReLU(5, 0, 10, 0, 1), 5/10)
    np.testing.assert_allclose(
        eda.stats.bidir_ReLU(10, 0, 10, 0, 1), 1)
    np.testing.assert_allclose(
        eda.stats.bidir_ReLU(15, 0, 10, 0, 1), 1)

    # test default ub lb
    np.testing.assert_allclose(eda.stats.bidir_ReLU(0, 0, 10), 0)
    np.testing.assert_allclose(eda.stats.bidir_ReLU(-1, 0, 10), 0)
    np.testing.assert_allclose(eda.stats.bidir_ReLU(-10, 0, 10), 0)
    np.testing.assert_allclose(eda.stats.bidir_ReLU(1, 0, 10), 1/10)
    np.testing.assert_allclose(eda.stats.bidir_ReLU(5, 0, 10), 5/10)
    np.testing.assert_allclose(eda.stats.bidir_ReLU(10, 0, 10), 1)
    np.testing.assert_allclose(eda.stats.bidir_ReLU(15, 0, 10), 1)

    # different ub lb
    np.testing.assert_allclose(
        eda.stats.bidir_ReLU(0, 0, 10, 10, 60), 10)
    np.testing.assert_allclose(
        eda.stats.bidir_ReLU(-1, 0, 10, 10, 60), 10)
    np.testing.assert_allclose(
        eda.stats.bidir_ReLU(-10, 0, 10, 10, 60), 10)
    np.testing.assert_allclose(
        eda.stats.bidir_ReLU(1, 0, 10, 10, 60), 15)
    np.testing.assert_allclose(
        eda.stats.bidir_ReLU(5, 0, 10, 10, 60), 35)
    np.testing.assert_allclose(
        eda.stats.bidir_ReLU(10, 0, 10, 10, 60), 60)
    np.testing.assert_allclose(
        eda.stats.bidir_ReLU(15, 0, 10, 10, 60), 60)

    # different start end
    np.testing.assert_allclose(
        eda.stats.bidir_ReLU(10, 10, 110, 10, 60), 10)
    np.testing.assert_allclose(
        eda.stats.bidir_ReLU(-1, 10, 110, 10, 60), 10)
    np.testing.assert_allclose(
        eda.stats.bidir_ReLU(-10, 10, 110, 10, 60), 10)
    np.testing.assert_allclose(
        eda.stats.bidir_ReLU(11, 10, 110, 10, 60), 10.5)
    np.testing.assert_allclose(
        eda.stats.bidir_ReLU(15, 10, 110, 10, 60), 12.5)
    np.testing.assert_allclose(
        eda.stats.bidir_ReLU(110, 10, 110, 10, 60), 60)
    np.testing.assert_allclose(
        eda.stats.bidir_ReLU(120, 10, 110, 10, 60), 60)

    np.testing.assert_allclose(
        eda.stats.bidir_ReLU(-10, 10, 110, 60, 60), 60)
    np.testing.assert_allclose(
        eda.stats.bidir_ReLU(10, 10, 110, 60, 60), 60)
    np.testing.assert_allclose(
        eda.stats.bidir_ReLU(11, 10, 110, 60, 60), 60)
    np.testing.assert_allclose(
        eda.stats.bidir_ReLU(110, 10, 110, 60, 60), 60)
    np.testing.assert_allclose(
        eda.stats.bidir_ReLU(120, 10, 110, 60, 60), 60)

    np.testing.assert_allclose(
        eda.stats.bidir_ReLU(-10, 10, 10, 10, 60), 10)
    np.testing.assert_allclose(
        eda.stats.bidir_ReLU(10, 10, 10, 10, 60), 60)
    np.testing.assert_allclose(
        eda.stats.bidir_ReLU(11, 10, 10, 10, 60), 60)
    np.testing.assert_allclose(
        eda.stats.bidir_ReLU(12, 10, 10, 10, 60), 60)


def test_bidir_ReLU_wrong_args():
    with pytest.raises(ValueError) as excinfo:
        eda.stats.bidir_ReLU(10, 0, -1, 10, 60)
    with pytest.raises(ValueError) as excinfo:
        eda.stats.bidir_ReLU(10, 0, 110, 70, 60)
    with pytest.raises(ValueError) as excinfo:
        eda.stats.bidir_ReLU(10, 0, 110, 0, -1)
