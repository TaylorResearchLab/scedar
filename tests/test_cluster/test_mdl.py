import pytest

import numpy as np

import scedar.cluster as cluster


class TestMultinomialMdl(object):
    """docstring for TestMultinomialMdl"""

    def test_empty_x(self):
        mmdl = cluster.MultinomialMdl([])
        assert mmdl.mdl == 0

    def test_single_level(self):
        mmdl = cluster.MultinomialMdl(["a"]*10)
        np.testing.assert_allclose(mmdl.mdl, np.log(10))

    def test_multi_levels(self):
        x = ["a"]*10 + ["b"]*25
        ux, uxcnt = np.unique(x, return_counts=True)
        mmdl = cluster.MultinomialMdl(x)
        np.testing.assert_allclose(mmdl.mdl,
                                   (-np.log(uxcnt / len(x)) * uxcnt).sum())

    def test_wrong_x_shape(self):
        with pytest.raises(ValueError) as excinfo:
            cluster.MultinomialMdl(np.arange(6).reshape(3, 2))

    def test_getter(self):
        mmdl = cluster.MultinomialMdl([])
        assert mmdl.x == []
        mmdl2 = cluster.MultinomialMdl([0, 0, 1, 1, 1])
        assert mmdl2.x == [0, 0, 1, 1, 1]


class TestZeroIdcGKdeMdl(object):
    """docstring for TestZeroIdcGKdeMdl"""
    x = np.concatenate([np.repeat(0, 50),
                        np.random.uniform(1, 2, size=200),
                        np.repeat(0, 50)])
    x_all_zero = np.repeat(0, 100)
    x_one_nonzero = np.array([0]*99 + [1])
    x_all_non_zero = x[50:250]

    def test_std_usage(self):
        zikm = cluster.ZeroIdcGKdeMdl(self.x)
        np.testing.assert_allclose(zikm.x, self.x)

        assert zikm.x is not self.x
        np.testing.assert_allclose(zikm.x_nonzero, self.x[50:250])

        np.testing.assert_allclose(zikm.mdl, zikm.zi_mdl + zikm.kde_mdl)

        np.testing.assert_allclose(zikm.zi_mdl,
                                   np.log(3) + cluster.MultinomialMdl(self.x != 0).mdl)

        assert zikm._bw_method == "silverman"

        assert zikm.bandwidth is not None

        # test > 0 value kde same
        zikm2 = cluster.ZeroIdcGKdeMdl(self.x[50:250])
        np.testing.assert_allclose(zikm2.kde_mdl, zikm.kde_mdl)
        np.testing.assert_allclose(zikm2.bandwidth, zikm.bandwidth)

    def test_all_zero(self):
        zikm = cluster.ZeroIdcGKdeMdl(self.x_all_zero)
        assert zikm.bandwidth is None
        np.testing.assert_allclose(zikm.zi_mdl, np.log(3))
        assert zikm.x_nonzero.size == 0
        np.testing.assert_allclose(zikm.x, self.x_all_zero)
        np.testing.assert_allclose(zikm.kde_mdl, 0)
        np.testing.assert_allclose(zikm.mdl, zikm.zi_mdl + zikm.kde_mdl)

    def test_all_nonzero(self):
        zikm = cluster.ZeroIdcGKdeMdl(self.x_all_non_zero)
        np.testing.assert_allclose(zikm.zi_mdl, np.log(3))

    def test_one_nonzero(self):
        zikm = cluster.ZeroIdcGKdeMdl(self.x_one_nonzero)
        assert zikm.bandwidth is None
        np.testing.assert_allclose(zikm.kde_mdl, np.log(1))

    def test_empty(self):
        zikm = cluster.ZeroIdcGKdeMdl(np.array([]))
        assert zikm.mdl == 0
        assert zikm.zi_mdl == 0
        assert zikm.kde_mdl == 0

    def test_kde_bw(self):
        zikm = cluster.ZeroIdcGKdeMdl(self.x)
        zikm2 = cluster.ZeroIdcGKdeMdl(self.x, "scott")
        zikm3 = cluster.ZeroIdcGKdeMdl(self.x, 1)
        xnz_std = zikm.x_nonzero.std(ddof=1)
        np.testing.assert_allclose(1, zikm3.bandwidth / xnz_std)
        np.testing.assert_raises(AssertionError,
                                 np.testing.assert_allclose,
                                 zikm2.bandwidth, zikm3.bandwidth)

    def test_wrong_x_shape(self):
        with pytest.raises(ValueError) as excinfo:
            cluster.ZeroIdcGKdeMdl(np.arange(10).reshape(5, 2))

    def test_2d_kde(self):
        logdens = cluster.ZeroIdcGKdeMdl.gaussian_kde_logdens(
            np.random.normal(size=50).reshape(10, 5))
        assert logdens.ndim == 1
        assert logdens.size == 10

    def test_wrong_kde_x_shape(self):
        with pytest.raises(ValueError) as excinfo:
            cluster.ZeroIdcGKdeMdl.gaussian_kde_logdens(
                np.reshape(np.arange(9), (3, 3, 1)))


class TestMDLSampleDistanceMatrix(object):
    """docstring for TestMDLSampleDistanceMatrix"""
    np.random.seed(5009)
    x50x5 = np.vstack((np.zeros((30, 5)), np.random.ranf((20, 5))))
    labs50 = [0]*10 + [1]*35 + [2]*5

    def test_mdl_computation(self):
        mdl_sdm = cluster.MDLSampleDistanceMatrix(
            self.x50x5, labs=self.labs50, metric="euclidean")
        no_lab_mdl = mdl_sdm.no_lab_mdl()
        ulab_s_ind_l, ulab_cnt_l, ulab_mdl_l, cluster_mdl = mdl_sdm.lab_mdl()
        assert ulab_s_ind_l == [list(range(10)), list(range(10, 45)),
                                list(range(45, 50))]

        ulab_cnt_l = [10, 35, 5]

        for i in range(3):
            ci_mdl = cluster.MDLSampleDistanceMatrix(
                self.x50x5[ulab_s_ind_l[i], :],
                labs=[self.labs50[ii] for ii in ulab_s_ind_l[i]],
                metric="euclidean")

            np.testing.assert_allclose(
                ci_mdl.no_lab_mdl(),
                ulab_mdl_l[i] - cluster_mdl * ulab_cnt_l[i] / 50)

    def test_mdl_computation_mp(self):
        mdl_sdm = cluster.MDLSampleDistanceMatrix(
            self.x50x5, labs=self.labs50, metric="euclidean")
        no_lab_mdl = mdl_sdm.no_lab_mdl(nprocs=2)
        ulab_s_ind_l, ulab_cnt_l, ulab_mdl_l, cluster_mdl = mdl_sdm.lab_mdl(
            nprocs=2)
        assert ulab_s_ind_l == [list(range(10)), list(range(10, 45)),
                                list(range(45, 50))]

        ulab_cnt_l = [10, 35, 5]

        for i in range(3):
            ci_mdl = cluster.MDLSampleDistanceMatrix(
                self.x50x5[ulab_s_ind_l[i], :],
                labs=[self.labs50[ii] for ii in ulab_s_ind_l[i]],
                metric="euclidean")

            np.testing.assert_allclose(
                ci_mdl.no_lab_mdl(nprocs=5),
                ulab_mdl_l[i] - cluster_mdl * ulab_cnt_l[i] / 50)

    def test_mdl_ret_internal(self):
        mdl_sdm = cluster.MDLSampleDistanceMatrix(
            self.x50x5, labs=self.labs50, metric="euclidean")

        ulab_s_ind_l, ulab_cnt_l, ulab_mdl_l, cluster_mdl, mdl_l = mdl_sdm.lab_mdl(
            ret_internal=True)
        np.testing.assert_allclose(sum(mdl_l) + cluster_mdl,
                                   sum(ulab_mdl_l))

        ulab_s_ind_l2, ulab_cnt_l2, ulab_mdl_l2, cluster_mdl2 = mdl_sdm.lab_mdl()
        assert ulab_s_ind_l == ulab_s_ind_l2
        assert ulab_cnt_l == ulab_cnt_l2
        assert ulab_mdl_l == ulab_mdl_l2
        assert cluster_mdl == cluster_mdl2

    def test_per_column_zigkmdl_wrong_xshape(self):
        with pytest.raises(ValueError) as excinfo:
            cluster.MDLSampleDistanceMatrix.per_column_zigkmdl(np.zeros(10))

        with pytest.raises(ValueError) as excinfo:
            cluster.MDLSampleDistanceMatrix.per_column_zigkmdl(
                np.zeros((10, 10, 10)))

    def test_per_column_zigkmdl_ret_internal(self):
        mdl_sum = cluster.MDLSampleDistanceMatrix.per_column_zigkmdl(
            self.x50x5)
        mdl_sum, mdl_l = cluster.MDLSampleDistanceMatrix.per_column_zigkmdl(
            self.x50x5, ret_internal=True)
        np.testing.assert_allclose(mdl_sum, sum(map(lambda x: x.mdl, mdl_l)))
