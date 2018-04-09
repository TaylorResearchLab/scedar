import pytest

import numpy as np

import sklearn.datasets as skdset
from sklearn import metrics

import scedar.cluster as cluster
import scedar.eda as eda

import scipy.cluster.hierarchy as sch


class TestMIRAC(object):
    """docstring for TestMIRAC"""

    def test_bidir_ReLU(self):
        np.testing.assert_allclose(cluster.MIRAC.bidir_ReLU(0, 0, 10, 0, 1), 0)
        np.testing.assert_allclose(
            cluster.MIRAC.bidir_ReLU(-1, 0, 10, 0, 1), 0)
        np.testing.assert_allclose(
            cluster.MIRAC.bidir_ReLU(-10, 0, 10, 0, 1), 0)
        np.testing.assert_allclose(
            cluster.MIRAC.bidir_ReLU(1, 0, 10, 0, 1), 1/10)
        np.testing.assert_allclose(
            cluster.MIRAC.bidir_ReLU(5, 0, 10, 0, 1), 5/10)
        np.testing.assert_allclose(
            cluster.MIRAC.bidir_ReLU(10, 0, 10, 0, 1), 1)
        np.testing.assert_allclose(
            cluster.MIRAC.bidir_ReLU(15, 0, 10, 0, 1), 1)

        # test default ub lb
        np.testing.assert_allclose(cluster.MIRAC.bidir_ReLU(0, 0, 10), 0)
        np.testing.assert_allclose(cluster.MIRAC.bidir_ReLU(-1, 0, 10), 0)
        np.testing.assert_allclose(cluster.MIRAC.bidir_ReLU(-10, 0, 10), 0)
        np.testing.assert_allclose(cluster.MIRAC.bidir_ReLU(1, 0, 10), 1/10)
        np.testing.assert_allclose(cluster.MIRAC.bidir_ReLU(5, 0, 10), 5/10)
        np.testing.assert_allclose(cluster.MIRAC.bidir_ReLU(10, 0, 10), 1)
        np.testing.assert_allclose(cluster.MIRAC.bidir_ReLU(15, 0, 10), 1)

        # different ub lb
        np.testing.assert_allclose(
            cluster.MIRAC.bidir_ReLU(0, 0, 10, 10, 60), 10)
        np.testing.assert_allclose(
            cluster.MIRAC.bidir_ReLU(-1, 0, 10, 10, 60), 10)
        np.testing.assert_allclose(
            cluster.MIRAC.bidir_ReLU(-10, 0, 10, 10, 60), 10)
        np.testing.assert_allclose(
            cluster.MIRAC.bidir_ReLU(1, 0, 10, 10, 60), 15)
        np.testing.assert_allclose(
            cluster.MIRAC.bidir_ReLU(5, 0, 10, 10, 60), 35)
        np.testing.assert_allclose(
            cluster.MIRAC.bidir_ReLU(10, 0, 10, 10, 60), 60)
        np.testing.assert_allclose(
            cluster.MIRAC.bidir_ReLU(15, 0, 10, 10, 60), 60)

        # different start end
        np.testing.assert_allclose(
            cluster.MIRAC.bidir_ReLU(10, 10, 110, 10, 60), 10)
        np.testing.assert_allclose(
            cluster.MIRAC.bidir_ReLU(-1, 10, 110, 10, 60), 10)
        np.testing.assert_allclose(
            cluster.MIRAC.bidir_ReLU(-10, 10, 110, 10, 60), 10)
        np.testing.assert_allclose(
            cluster.MIRAC.bidir_ReLU(11, 10, 110, 10, 60), 10.5)
        np.testing.assert_allclose(
            cluster.MIRAC.bidir_ReLU(15, 10, 110, 10, 60), 12.5)
        np.testing.assert_allclose(
            cluster.MIRAC.bidir_ReLU(110, 10, 110, 10, 60), 60)
        np.testing.assert_allclose(
            cluster.MIRAC.bidir_ReLU(120, 10, 110, 10, 60), 60)

        np.testing.assert_allclose(
            cluster.MIRAC.bidir_ReLU(-10, 10, 110, 60, 60), 60)
        np.testing.assert_allclose(
            cluster.MIRAC.bidir_ReLU(10, 10, 110, 60, 60), 60)
        np.testing.assert_allclose(
            cluster.MIRAC.bidir_ReLU(11, 10, 110, 60, 60), 60)
        np.testing.assert_allclose(
            cluster.MIRAC.bidir_ReLU(110, 10, 110, 60, 60), 60)
        np.testing.assert_allclose(
            cluster.MIRAC.bidir_ReLU(120, 10, 110, 60, 60), 60)

        np.testing.assert_allclose(
            cluster.MIRAC.bidir_ReLU(-10, 10, 10, 10, 60), 10)
        np.testing.assert_allclose(
            cluster.MIRAC.bidir_ReLU(10, 10, 10, 10, 60), 60)
        np.testing.assert_allclose(
            cluster.MIRAC.bidir_ReLU(11, 10, 10, 10, 60), 60)
        np.testing.assert_allclose(
            cluster.MIRAC.bidir_ReLU(12, 10, 10, 10, 60), 60)

    def test_bidir_ReLU_wrong_args(self):
        with pytest.raises(ValueError) as excinfo:
            cluster.MIRAC.bidir_ReLU(10, 0, -1, 10, 60)
        with pytest.raises(ValueError) as excinfo:
            cluster.MIRAC.bidir_ReLU(10, 0, 110, 70, 60)
        with pytest.raises(ValueError) as excinfo:
            cluster.MIRAC.bidir_ReLU(10, 0, 110, 0, -1)

    @staticmethod
    def raw_red_ratio(n1, n):
        return n1 / n * (1 - 1/n1)

    def test_spl_mdl_ratio_pos_mdl(self):
        np.testing.assert_allclose(cluster.MIRAC.spl_mdl_ratio(50, 100, 1, 50),
                                   self.raw_red_ratio(50, 100))

        assert cluster.MIRAC.spl_mdl_ratio(
            60, 100, 1, 25) < self.raw_red_ratio(60, 100)
        assert cluster.MIRAC.spl_mdl_ratio(
            40, 100, 1, 25) > self.raw_red_ratio(40, 100)
        assert cluster.MIRAC.spl_mdl_ratio(
            51, 100, 1, 25) < self.raw_red_ratio(51, 100)
        assert cluster.MIRAC.spl_mdl_ratio(
            49, 100, 1, 25) > self.raw_red_ratio(49, 100)

        assert cluster.MIRAC.spl_mdl_ratio(1, 100, 1, 25) >= 0
        assert cluster.MIRAC.spl_mdl_ratio(99, 100, 1, 25) < 1
        for n1 in range(1, 1000):
            cluster.MIRAC.spl_mdl_ratio(n1, 1000, 1, 1)

    def test_spl_mdl_ratio_neg_mdl(self):
        np.testing.assert_allclose(cluster.MIRAC.spl_mdl_ratio(50, 100, -1, 50),
                                   self.raw_red_ratio(50, 100))

        assert cluster.MIRAC.spl_mdl_ratio(
            60, 100, -1, 25) > self.raw_red_ratio(60, 100)
        assert cluster.MIRAC.spl_mdl_ratio(
            40, 100, -1, 25) < self.raw_red_ratio(40, 100)
        assert cluster.MIRAC.spl_mdl_ratio(
            51, 100, -1, 25) > self.raw_red_ratio(51, 100)
        assert cluster.MIRAC.spl_mdl_ratio(
            49, 100, -1, 25) < self.raw_red_ratio(49, 100)

        assert cluster.MIRAC.spl_mdl_ratio(1, 100, -1, 25) >= 0
        assert cluster.MIRAC.spl_mdl_ratio(99, 100, -1, 25) < 1

        for n1 in range(1, 1000):
            cluster.MIRAC.spl_mdl_ratio(n1, 1000, -1, 1)

    def test_spl_mdl_ratio_shrink_factor(self):
        # small shrink factor reduces the amonut of correction
        assert (cluster.MIRAC.spl_mdl_ratio(60, 100, 1, 25)
                > cluster.MIRAC.spl_mdl_ratio(60, 100, 1, 10))

        assert (cluster.MIRAC.spl_mdl_ratio(40, 100, 1, 25)
                < cluster.MIRAC.spl_mdl_ratio(40, 100, 1, 10))

        assert (cluster.MIRAC.spl_mdl_ratio(60, 100, -1, 25)
                < cluster.MIRAC.spl_mdl_ratio(60, 100, -1, 10))

        assert (cluster.MIRAC.spl_mdl_ratio(40, 100, -1, 25)
                > cluster.MIRAC.spl_mdl_ratio(40, 100, -1, 10))

    def test_spl_mdl_ratio_wrong_args(self):
        with pytest.raises(ValueError) as excinfo:
            cluster.MIRAC.spl_mdl_ratio(60, 100, 1, -1)
        with pytest.raises(ValueError) as excinfo:
            cluster.MIRAC.spl_mdl_ratio(60, 59, 1, 10)
        with pytest.raises(ValueError) as excinfo:
            cluster.MIRAC.spl_mdl_ratio(0, 100, 1, 10)
        with pytest.raises(ValueError) as excinfo:
            cluster.MIRAC.spl_mdl_ratio(-1, 100, 1, 10)
        with pytest.raises(ValueError) as excinfo:
            cluster.MIRAC.spl_mdl_ratio(-2, -1, 1, 10)

    def test_bi_split_compensation_factor(self):
        assert (cluster.MIRAC.bi_split_compensation_factor(100, 500, 25, 100)
                < cluster.MIRAC.bi_split_compensation_factor(101, 500, 25, 100))

        cluster.MIRAC.bi_split_compensation_factor(500, 500, 25, 100)

        np.testing.assert_allclose(
            cluster.MIRAC.bi_split_compensation_factor(25, 500, 25, 25),
            cluster.MIRAC.bi_split_compensation_factor(100, 2000, 25, 50))

        np.testing.assert_allclose(
            cluster.MIRAC.bi_split_compensation_factor(24, 500, 25, 25),
            cluster.MIRAC.bi_split_compensation_factor(24, 500, 25, 50))

        assert (cluster.MIRAC.bi_split_compensation_factor(100, 500, 25, 50)
                > cluster.MIRAC.bi_split_compensation_factor(99, 500, 25, 50))

        np.testing.assert_allclose(
            cluster.MIRAC.bi_split_compensation_factor(100, 500, 25, 100),
            cluster.MIRAC.bi_split_compensation_factor(100, 500, 50, 250))

        # when minimax becomes larger, complexity decreases, need less
        # compensation. The ratio affects factor value,
        # but not the absolute count.
        assert (cluster.MIRAC.bi_split_compensation_factor(100, 500, 25, 75)
                > cluster.MIRAC.bi_split_compensation_factor(100, 500, 26, 75 * 26/25))

        assert (cluster.MIRAC.bi_split_compensation_factor(100, 500, 25, 99)
                > cluster.MIRAC.bi_split_compensation_factor(100, 500, 25, 250))

        assert (cluster.MIRAC.bi_split_compensation_factor(100, 500, 25, 75)
                > cluster.MIRAC.bi_split_compensation_factor(100, 500, 25 * (76/75), 76))

        for i in range(1, 1001):
            x = cluster.MIRAC.bi_split_compensation_factor(i, 1000, 25, 250)
            assert 0 <= x <= 0.5

    def test_bi_split_compensation_factor_wrong_args(self):
        # 0 < stlc < n
        with pytest.raises(ValueError) as excinfo:
            cluster.MIRAC.bi_split_compensation_factor(0, 500, 25, 99)
        with pytest.raises(ValueError) as excinfo:
            cluster.MIRAC.bi_split_compensation_factor(-1, 500, 25, 99)
        with pytest.raises(ValueError) as excinfo:
            cluster.MIRAC.bi_split_compensation_factor(100, 99, 25, 99)
        with pytest.raises(ValueError) as excinfo:
            cluster.MIRAC.bi_split_compensation_factor(0, 1, 25, 99)

        # n > 0
        with pytest.raises(ValueError) as excinfo:
            cluster.MIRAC.bi_split_compensation_factor(100, -1, 25, 99)
        with pytest.raises(ValueError) as excinfo:
            cluster.MIRAC.bi_split_compensation_factor(-2, -1, 25, 99)

        # 0 < minimax < maxmini
        with pytest.raises(ValueError) as excinfo:
            cluster.MIRAC.bi_split_compensation_factor(100, 500, 0, 99)
        with pytest.raises(ValueError) as excinfo:
            cluster.MIRAC.bi_split_compensation_factor(100, 500, -1, 99)
        with pytest.raises(ValueError) as excinfo:
            cluster.MIRAC.bi_split_compensation_factor(100, 500, -2, -1)
        with pytest.raises(ValueError) as excinfo:
            cluster.MIRAC.bi_split_compensation_factor(100, 500, 2, -1)
        with pytest.raises(ValueError) as excinfo:
            cluster.MIRAC.bi_split_compensation_factor(100, 500, 2, 1)

    def test_mirac_wrong_args(self):
        x = np.zeros((10, 10))
        with pytest.raises(ValueError) as excinfo:
            cluster.MIRAC(x, metric='euclidean', minimax_n=-0.1)
        with pytest.raises(ValueError) as excinfo:
            cluster.MIRAC(x, metric='euclidean', minimax_n=25, maxmini_n=24)
        with pytest.raises(ValueError) as excinfo:
            cluster.MIRAC(x, metric='euclidean', minimax_n=-0.1)
        with pytest.raises(ValueError) as excinfo:
            cluster.MIRAC(x, metric='euclidean', cl_mdl_scale_factor=-0.1)
        # hac tree n_leaves different from n_samples
        z = sch.linkage([[0], [5], [6], [8], [9], [12]],
                        method='single', optimal_ordering=True)
        hct = eda.HClustTree(sch.to_tree(z))
        with pytest.raises(ValueError) as excinfo:
            cluster.MIRAC(x, metric='euclidean', hac_tree=hct)

    # no specific purpose. Just to exaust the coverage
    def test_mirac_cover_tests(self):
        # make all non-neg
        x = np.zeros((10, 10))
        cluster.MIRAC(x, metric='euclidean', minimax_n=25, maxmini_n=250)
        cluster.MIRAC(x, metric='euclidean', minimax_n=25)
        cluster.MIRAC(x, metric='euclidean', minimax_n=25, maxmini_n=250,
                      verbose=True)

        tx, tlab = skdset.make_blobs(n_samples=500, n_features=2,
                                     centers=10, random_state=8927)
        tx = tx - tx.min()
        sdm = eda.SampleDistanceMatrix(tx, metric='euclidean')
        m = cluster.MIRAC(sdm._x, sdm._d, metric='euclidean', linkage='ward',
                          minimax_n=35, maxmini_n=100,
                          cl_mdl_scale_factor=1, verbose=True)
        assert len(m.labs) == 500
        hct = eda.HClustTree.hclust_tree(sdm._d, linkage='ward')
        m2 = cluster.MIRAC(sdm._x, hac_tree=hct,
                          minimax_n=35, maxmini_n=100,
                          cl_mdl_scale_factor=1, verbose=True)
        assert m.labs == m2.labs
        assert m2._sdm._lazy_load_d is None

        tx2, tlab2 = skdset.make_blobs(n_samples=500, n_features=5,
                                       centers=100, cluster_std=1,
                                       random_state=8927)
        tx2 = tx2 - tx2.min()
        cluster.MIRAC(tx2, metric='correlation', minimax_n=1, maxmini_n=2,
                          cl_mdl_scale_factor=0, verbose=True)

