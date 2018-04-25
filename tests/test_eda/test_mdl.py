import pytest

import numpy as np

import scedar.eda as eda


def test_np_number_1d():
    # correct dtype
    eda.mdl.np_number_1d([1, 2, 3], int)
    eda.mdl.np_number_1d([1, 2, 3], float)
    # wrong type
    with pytest.raises(ValueError) as excinfo:
        eda.mdl.np_number_1d([1, 2, 3], str)

    with pytest.raises(ValueError) as excinfo:
        eda.mdl.np_number_1d([1, 2, 3], np.bool_)

    with pytest.raises(ValueError) as excinfo:
        eda.mdl.np_number_1d([1, 2, 3], bool)

    # wrong dim, i.e. non 1d
    with pytest.raises(ValueError) as excinfo:
        eda.mdl.np_number_1d([[1, 2], [3, 4]])
    with pytest.raises(ValueError) as excinfo:
        eda.mdl.np_number_1d(1)

    # test copy
    x = np.array([1.1, 2.2])
    xcp = eda.mdl.np_number_1d(x)
    assert xcp is not x
    xref = eda.mdl.np_number_1d(x, copy=False)
    assert xref is x


class TestMultinomialMdl(object):
    """docstring for TestMultinomialMdl"""

    def test_empty_x(self):
        mmdl = eda.MultinomialMdl([])
        assert mmdl.mdl == 0

    def test_single_level(self):
        mmdl = eda.MultinomialMdl([1]*10)
        np.testing.assert_allclose(mmdl.mdl, np.log(10))

    def test_multi_levels(self):
        x = [1]*10 + [2]*25
        _, uxcnt = np.unique(x, return_counts=True)
        mmdl = eda.MultinomialMdl(x)
        np.testing.assert_allclose(mmdl.mdl,
                                   (-np.log(uxcnt / len(x)) * uxcnt).sum())

    def test_getter(self):
        mmdl = eda.MultinomialMdl([])
        assert mmdl.x.tolist() == []
        mmdl2 = eda.MultinomialMdl([0, 0, 1, 1, 1])
        assert mmdl2.x.tolist() == [0, 0, 1, 1, 1]

    def test_encode(self):
        # retrospect
        np.testing.assert_allclose(eda.MultinomialMdl([0]).mdl,
                                   eda.MultinomialMdl([0]).mdl)

        np.testing.assert_allclose(eda.MultinomialMdl(np.arange(10)).mdl,
                                   eda.MultinomialMdl(np.arange(10)).mdl)

        np.testing.assert_allclose(eda.MultinomialMdl([0, 1, 2, 10, -10]).mdl,
                                   eda.MultinomialMdl([0, 1, 2, 10, -10]).mdl)

        np.testing.assert_allclose(
            eda.MultinomialMdl([0]).encode([0, 1, 5, 100, 200, -20],
                                           use_adjescent_when_absent=True),
            0)

        np.testing.assert_allclose(
            eda.MultinomialMdl([0]).encode([0, 1, 5, 100, 200, -20],
                                           use_adjescent_when_absent=False),
            np.log(200*2) * 5)

        np.testing.assert_allclose(
            eda.MultinomialMdl([]).encode(
                [0, 1, 5, 100, 200, -20],
                use_adjescent_when_absent=False),
            np.log(200*2) * 6)

        np.testing.assert_allclose(
            eda.MultinomialMdl([0]).encode(
                [0, 1, 5, 100, 200, -20]),
            np.log(200*2) * 5)

        assert eda.MultinomialMdl([0]).encode([]) == 0

        np.testing.assert_allclose(
            eda.MultinomialMdl([0, 0, 3, 3, 3]).encode(
                [-20, 0, 2, 5, 100, 200],
                use_adjescent_when_absent=True),
            -np.log(0.4) * 2 - np.log(0.6) * 4)

        np.testing.assert_allclose(
            eda.MultinomialMdl([0, 0, 3, 3, 3]).encode(
                [-20, 1, 1.5, 2, 100, 200],
                use_adjescent_when_absent=True),
            -np.log(0.4) * 2 - np.log(0.6) * 4)


class TestZeroIMultinomialMdl(object):
    def test_init(self):
        zimn = eda.ZeroIMultinomialMdl([0, 0, 1, 1, 3, 3, 3])
        np.testing.assert_allclose(
            zimn._zi_mdl, -np.log(2/7)*2 - np.log(5/7)*5 + np.log(3))
        np.testing.assert_allclose(
            zimn._mn_mdl, -np.log(2/5)*2 - np.log(3/5)*3)
        assert zimn.mdl == zimn._zi_mdl + zimn._mn_mdl

    def test_encode(self):
        zimn = eda.ZeroIMultinomialMdl([0, 0, 1, 1, 3, 3, 3])
        qx = [-20, 1, 1.5, 2, 100, 200]

        np.testing.assert_allclose(
            zimn.encode(qx, use_adjescent_when_absent=True),
            -np.log(0.4) * 3 - np.log(0.6) * 3 - np.log(5/7) * 6 + np.log(3))

        np.testing.assert_allclose(
            zimn.encode(qx, use_adjescent_when_absent=False),
            -np.log(0.4) * 1 + np.log(200*2) * 5 - np.log(5/7) * 6 + np.log(3))


class TestGKdeMdl(object):
    """docstring for TestKdeMdl"""
    def test_wrong_x_shape(self):
        with pytest.raises(ValueError) as excinfo:
            eda.GKdeMdl(np.arange(10).reshape(5, 2))

    def test_x(self):
        gkmdl = eda.GKdeMdl(np.arange(100))
        np.testing.assert_equal(gkmdl.x, np.arange(100))

    def test_2d_kde(self):
        logdens = eda.GKdeMdl.gaussian_kde_logdens(
            np.random.normal(size=50).reshape(10, 5))
        assert logdens.ndim == 1
        assert logdens.size == 10

    def test_wrong_kde_x_shape(self):
        with pytest.raises(ValueError) as excinfo:
            eda.GKdeMdl.gaussian_kde_logdens(
                np.reshape(np.arange(9), (3, 3, 1)))

    def test_encode(self):
        np.random.seed(1)
        x = np.random.normal(size=5000)
        gkmdl = eda.GKdeMdl(x)
        np.testing.assert_allclose(gkmdl.encode(x) + np.log(2), gkmdl.mdl)
        np.testing.assert_allclose(eda.GKdeMdl([]).encode([0, 1, 2]),
                                   np.log(2*2)*3)
        assert gkmdl.encode([]) == 0


class TestZeroIGKdeMdl(object):
    """docstring for TestZeroIGKdeMdl"""
    x = np.concatenate([np.repeat(0, 50),
                        np.random.uniform(1, 2, size=200),
                        np.repeat(0, 50)])
    x_all_zero = np.repeat(0, 100)
    x_one_nonzero = np.array([0]*99 + [1])
    x_all_non_zero = x[50:250]

    def test_std_usage(self):
        zikm = eda.ZeroIGKdeMdl(self.x)
        np.testing.assert_allclose(zikm.x, self.x)

        assert zikm.x is not self.x
        np.testing.assert_allclose(zikm.x_nonzero, self.x[50:250])

        np.testing.assert_allclose(zikm.mdl, zikm.zi_mdl + zikm.kde_mdl)

        np.testing.assert_allclose(
            zikm.zi_mdl, np.log(3) + eda.MultinomialMdl(self.x != 0).mdl)

        assert zikm._bw_method == "scott"

        assert zikm.bandwidth is not None

        # test > 0 value kde same
        zikm2 = eda.ZeroIGKdeMdl(self.x[50:250])
        np.testing.assert_allclose(zikm2.kde_mdl, zikm.kde_mdl)
        np.testing.assert_allclose(zikm2.bandwidth, zikm.bandwidth)

    def test_all_zero(self):
        zikm = eda.ZeroIGKdeMdl(self.x_all_zero)
        assert zikm.bandwidth is None
        np.testing.assert_allclose(zikm.zi_mdl,
                                   np.log(3) + np.log(len(self.x_all_zero)))
        assert zikm.x_nonzero.size == 0
        np.testing.assert_allclose(zikm.x, self.x_all_zero)
        np.testing.assert_allclose(zikm.kde_mdl, 0)
        np.testing.assert_allclose(zikm.mdl, zikm.zi_mdl + zikm.kde_mdl)

    def test_all_nonzero(self):
        zikm = eda.ZeroIGKdeMdl(self.x_all_non_zero)
        np.testing.assert_allclose(
            zikm.zi_mdl, np.log(3) + np.log(len(self.x_all_non_zero)))

    def test_one_nonzero(self):
        zikm = eda.ZeroIGKdeMdl(self.x_one_nonzero)
        assert zikm.bandwidth is None
        # 1 non-0, kde falls back on uniform
        np.testing.assert_allclose(
            zikm.kde_mdl, np.log(np.max(np.abs(self.x_one_nonzero))*2))

    def test_empty(self):
        zikm = eda.ZeroIGKdeMdl(np.array([]))
        assert zikm.mdl == np.log(3)
        assert zikm.zi_mdl == np.log(3)
        assert zikm.kde_mdl == 0

    def test_kde_bw(self):
        zikm = eda.ZeroIGKdeMdl(self.x)
        n = len(zikm.x_nonzero)
        d = 1
        # scott: n**(-1./(d+4))
        np.testing.assert_allclose(
            zikm.bandwidth,
            zikm.x_nonzero.std(ddof=1) * n**(-1/(d+4)))
        # silverman: (n * (d + 2) / 4.)**(-1. / (d + 4))
        zikm2 = eda.ZeroIGKdeMdl(self.x, "silverman")
        np.testing.assert_allclose(
            zikm2.bandwidth,
            zikm2.x_nonzero.std(ddof=1) * (n * (d + 2) / 4.)**(-1. / (d + 4)))
        zikm3 = eda.ZeroIGKdeMdl(self.x, 1)
        xnz_std = zikm.x_nonzero.std(ddof=1)
        np.testing.assert_allclose(1, zikm3.bandwidth / xnz_std)
        assert not np.allclose(zikm2.bandwidth, zikm3.bandwidth)

    def test_kde(self):
        zikm = eda.ZeroIGKdeMdl(self.x)
        kde = zikm.kde
        kde.logpdf([1, 2, 3])

        kde2 = eda.ZeroIGKdeMdl(self.x_all_zero).kde
        assert kde2 is None

        kde3 = eda.ZeroIGKdeMdl(self.x_one_nonzero).kde
        assert kde3 is None

    def test_wrong_x_shape(self):
        with pytest.raises(ValueError) as excinfo:
            eda.ZeroIGKdeMdl(np.arange(10).reshape(5, 2))

    def test_encode(self):
        zimdl = eda.ZeroIGKdeMdl(self.x)
        qx = [0, 0, 0] + [1, 2, 10, 20]
        emdl = zimdl.encode(qx)
        ezi = zimdl._zi_encoder.encode(qx)
        ekde = zimdl._kde_encoder.encode(list(filter(lambda x: x != 0, qx)))
        qp = zimdl._k / zimdl._n
        np.testing.assert_allclose(
            ezi, -np.log(qp) * 4 - np.log(1-qp) * 3 + np.log(3))
        np.testing.assert_allclose(emdl, ezi + ekde)
