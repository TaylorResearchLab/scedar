import pytest

import numpy as np

import sklearn.datasets as skdset

import scxplit.cluster as cluster
import scxplit.eda as eda

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
        np.testing.assert_allclose(zikm.kde_mdl, np.log(2))

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
            cluster.ZeroIdcGKdeMdl(np.arange(10).reshape(5,2))
    
    def test_2d_kde(self):
        logdens = cluster.ZeroIdcGKdeMdl.gaussian_kde_logdens(
            np.random.normal(size=50).reshape(10, 5))
        assert logdens.ndim == 1
        assert logdens.size == 10

    def test_wrong_kde_x_shape(self):
        with pytest.raises(ValueError) as excinfo:
            cluster.ZeroIdcGKdeMdl.gaussian_kde_logdens(np.reshape(np.arange(9), (3, 3, 1)))


class TestHClustTree(object):
    """docstring for TestHClustTree"""
    sdm_5x2 = eda.SampleDistanceMatrix([[0,0], 
                                        [100, 100], 
                                        [1,1], 
                                        [101, 101], 
                                        [80, 80]],
                                        metric="euclidean")
    # This tree should be
    #   _______|_____
    #   |       ____|___
    # __|___    |    __|___
    # |    |    |    |    |
    # 0    2    4    1    3
    # Leaves are in optimal order. 
    hct = cluster.HClustTree.hclust_tree(sdm_5x2.d, linkage="auto")
    
    def test_hclust_tree_args(self):
        cluster.HClustTree.hclust_tree(self.sdm_5x2.d, linkage="auto",
                                    n_eval_rounds=-1, is_euc_dist=True, 
                                    verbose=True)

    def test_hclust_tree(self):
        assert self.hct.prev is None

        assert self.hct.left_count() == 2
        assert self.hct.right_count() == 3
        assert self.hct.count() == 5

        assert len(self.hct.leaf_ids()) == 5
        assert self.hct.leaf_ids() == [0, 2, 4, 1, 3]

        assert len(self.hct.left_leaf_ids()) == 2
        assert self.hct.left_leaf_ids() == [0, 2]

        assert len(self.hct.right_leaf_ids()) == 3
        assert self.hct.right_leaf_ids() == [4, 1, 3]

        assert self.hct.left().left().left().count() == 0
        assert self.hct.left().left().left().leaf_ids() is None
        assert self.hct.left().left().left_leaf_ids() is None
        assert self.hct.left().left().right().count() == 0
        
    def test_hclust_tree_invalid_dmat(self):
        with pytest.raises(ValueError) as excinfo:
            cluster.HClustTree.hclust_tree(np.arange(5))

        with pytest.raises(ValueError) as excinfo:
            cluster.HClustTree.hclust_tree(np.arange(10).reshape(2, 5))

    def test_bi_partition(self):
        # return subtrees False
        labs1, sids1 = self.hct.bi_partition()

        # return subtrees True
        labs2, sids2, lst, rst = self.hct.bi_partition(return_subtrees=True)

        np.testing.assert_equal(labs1, [0, 0, 1, 1, 1])
        np.testing.assert_equal(sids1, [0, 2, 4, 1, 3])
        np.testing.assert_equal(sids1, self.hct.leaf_ids())
        
        assert labs1 == labs2
        assert sids1 == sids2

        assert lst.count() == 2
        assert lst.left_count() == 1
        assert lst.left_leaf_ids() == [0]
        assert lst.right_leaf_ids() == [2]
        assert lst.leaf_ids() == [0, 2]

        assert rst.leaf_ids() == [4, 1, 3]
        assert rst.right_leaf_ids() == [1, 3]
        assert rst.left_leaf_ids() == [4]

    def test_cluster_id_list_to_lab_array_wrong_id_list_type(self):
        with pytest.raises(ValueError) as excinfo:
            cluster.HClustTree.cluster_id_list_to_lab_array(
                np.array([[0, 1, 2], [3,4]]), [0, 1, 2, 3, 4])

    def test_cluster_id_list_to_lab_array_mismatched_ids_sids(self):
        with pytest.raises(ValueError) as excinfo:
            cluster.HClustTree.cluster_id_list_to_lab_array(
                [[0, 1, 2], [3,4]], [0, 1, 2, 3, 5])

    def test_cluster_id_list_to_lab_array_empty_cluster(self):
        with pytest.raises(ValueError) as excinfo:
            cluster.HClustTree.cluster_id_list_to_lab_array(
                [[], [0, 1, 2, 3, 4]], [0, 1, 2, 3, 4])

class TestMDLSampleDistanceMatrix(object):
    """docstring for TestMDLSampleDistanceMatrix"""
    np.random.seed(5009)
    x50x5 = np.vstack(( np.zeros((30,5)), np.random.ranf((20, 5)) ))
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
        mdl_sum = cluster.MDLSampleDistanceMatrix.per_column_zigkmdl(self.x50x5)
        mdl_sum, mdl_l = cluster.MDLSampleDistanceMatrix.per_column_zigkmdl(
            self.x50x5, ret_internal=True)
        np.testing.assert_allclose(mdl_sum, sum(map(lambda x: x.mdl, mdl_l)))


class TestMIRCH(object):
    """docstring for TestMIRCH"""

    def test_bidir_ReLU(self):
        np.testing.assert_allclose(cluster.MIRCH.bidir_ReLU(0, 0, 10, 0, 1), 0)
        np.testing.assert_allclose(cluster.MIRCH.bidir_ReLU(-1, 0, 10, 0, 1), 0)
        np.testing.assert_allclose(cluster.MIRCH.bidir_ReLU(-10, 0, 10, 0, 1), 0)
        np.testing.assert_allclose(cluster.MIRCH.bidir_ReLU(1, 0, 10, 0, 1), 1/10)
        np.testing.assert_allclose(cluster.MIRCH.bidir_ReLU(5, 0, 10, 0, 1), 5/10)
        np.testing.assert_allclose(cluster.MIRCH.bidir_ReLU(10, 0, 10, 0, 1), 1)
        np.testing.assert_allclose(cluster.MIRCH.bidir_ReLU(15, 0, 10, 0, 1), 1)

        # test default ub lb
        np.testing.assert_allclose(cluster.MIRCH.bidir_ReLU(0, 0, 10), 0)
        np.testing.assert_allclose(cluster.MIRCH.bidir_ReLU(-1, 0, 10), 0)
        np.testing.assert_allclose(cluster.MIRCH.bidir_ReLU(-10, 0, 10), 0)
        np.testing.assert_allclose(cluster.MIRCH.bidir_ReLU(1, 0, 10), 1/10)
        np.testing.assert_allclose(cluster.MIRCH.bidir_ReLU(5, 0, 10), 5/10)
        np.testing.assert_allclose(cluster.MIRCH.bidir_ReLU(10, 0, 10), 1)
        np.testing.assert_allclose(cluster.MIRCH.bidir_ReLU(15, 0, 10), 1)

        # different ub lb
        np.testing.assert_allclose(cluster.MIRCH.bidir_ReLU(0, 0, 10, 10, 60), 10)
        np.testing.assert_allclose(cluster.MIRCH.bidir_ReLU(-1, 0, 10, 10, 60), 10)
        np.testing.assert_allclose(cluster.MIRCH.bidir_ReLU(-10, 0, 10, 10, 60), 10)
        np.testing.assert_allclose(cluster.MIRCH.bidir_ReLU(1, 0, 10, 10, 60), 15)
        np.testing.assert_allclose(cluster.MIRCH.bidir_ReLU(5, 0, 10, 10, 60), 35)
        np.testing.assert_allclose(cluster.MIRCH.bidir_ReLU(10, 0, 10, 10, 60), 60)
        np.testing.assert_allclose(cluster.MIRCH.bidir_ReLU(15, 0, 10, 10, 60), 60)

        # different start end
        np.testing.assert_allclose(cluster.MIRCH.bidir_ReLU(10, 10, 110, 10, 60), 10)
        np.testing.assert_allclose(cluster.MIRCH.bidir_ReLU(-1, 10, 110, 10, 60), 10)
        np.testing.assert_allclose(cluster.MIRCH.bidir_ReLU(-10, 10, 110, 10, 60), 10)
        np.testing.assert_allclose(cluster.MIRCH.bidir_ReLU(11, 10, 110, 10, 60), 10.5)
        np.testing.assert_allclose(cluster.MIRCH.bidir_ReLU(15, 10, 110, 10, 60), 12.5)
        np.testing.assert_allclose(cluster.MIRCH.bidir_ReLU(110, 10, 110, 10, 60), 60)
        np.testing.assert_allclose(cluster.MIRCH.bidir_ReLU(120, 10, 110, 10, 60), 60)

        np.testing.assert_allclose(cluster.MIRCH.bidir_ReLU(-10, 10, 110, 60, 60), 60)
        np.testing.assert_allclose(cluster.MIRCH.bidir_ReLU(10, 10, 110, 60, 60), 60)
        np.testing.assert_allclose(cluster.MIRCH.bidir_ReLU(11, 10, 110, 60, 60), 60)
        np.testing.assert_allclose(cluster.MIRCH.bidir_ReLU(110, 10, 110, 60, 60), 60)
        np.testing.assert_allclose(cluster.MIRCH.bidir_ReLU(120, 10, 110, 60, 60), 60)

        np.testing.assert_allclose(cluster.MIRCH.bidir_ReLU(-10, 10, 10, 10, 60), 10)
        np.testing.assert_allclose(cluster.MIRCH.bidir_ReLU(10, 10, 10, 10, 60), 60)
        np.testing.assert_allclose(cluster.MIRCH.bidir_ReLU(11, 10, 10, 10, 60), 60)
        np.testing.assert_allclose(cluster.MIRCH.bidir_ReLU(12, 10, 10, 10, 60), 60)

    def test_bidir_ReLU_wrong_args(self):
        with pytest.raises(ValueError) as excinfo:
            cluster.MIRCH.bidir_ReLU(10, 0, -1, 10, 60)
        with pytest.raises(ValueError) as excinfo:
            cluster.MIRCH.bidir_ReLU(10, 0, 110, 70, 60)
        with pytest.raises(ValueError) as excinfo:
            cluster.MIRCH.bidir_ReLU(10, 0, 110, 0, -1)

    @staticmethod
    def raw_red_ratio(n1, n):
        return n1 / n * (1 - 1/n1)

    def test_spl_mdl_ratio_pos_mdl(self):
        np.testing.assert_allclose(cluster.MIRCH.spl_mdl_ratio(50, 100, 1, 50), 
                                   self.raw_red_ratio(50, 100))

        assert cluster.MIRCH.spl_mdl_ratio(60, 100, 1, 25) < self.raw_red_ratio(60, 100)
        assert cluster.MIRCH.spl_mdl_ratio(40, 100, 1, 25) > self.raw_red_ratio(40, 100)
        assert cluster.MIRCH.spl_mdl_ratio(51, 100, 1, 25) < self.raw_red_ratio(51, 100)
        assert cluster.MIRCH.spl_mdl_ratio(49, 100, 1, 25) > self.raw_red_ratio(49, 100)

        assert cluster.MIRCH.spl_mdl_ratio(1, 100, 1, 25) >= 0
        assert cluster.MIRCH.spl_mdl_ratio(99, 100, 1, 25) < 1
        for n1 in range(1, 1000):
            cluster.MIRCH.spl_mdl_ratio(n1, 1000, 1, 1)

    def test_spl_mdl_ratio_neg_mdl(self):
        np.testing.assert_allclose(cluster.MIRCH.spl_mdl_ratio(50, 100, -1, 50), 
                                   self.raw_red_ratio(50, 100))

        assert cluster.MIRCH.spl_mdl_ratio(60, 100, -1, 25) > self.raw_red_ratio(60, 100)
        assert cluster.MIRCH.spl_mdl_ratio(40, 100, -1, 25) < self.raw_red_ratio(40, 100)
        assert cluster.MIRCH.spl_mdl_ratio(51, 100, -1, 25) > self.raw_red_ratio(51, 100)
        assert cluster.MIRCH.spl_mdl_ratio(49, 100, -1, 25) < self.raw_red_ratio(49, 100)

        assert cluster.MIRCH.spl_mdl_ratio(1, 100, -1, 25) >= 0
        assert cluster.MIRCH.spl_mdl_ratio(99, 100, -1, 25) < 1

        for n1 in range(1, 1000):
            cluster.MIRCH.spl_mdl_ratio(n1, 1000, -1, 1)

    def test_spl_mdl_ratio_shrink_factor(self):
        # small shrink factor reduces the amonut of correction
        assert (cluster.MIRCH.spl_mdl_ratio(60, 100, 1, 25)
                > cluster.MIRCH.spl_mdl_ratio(60, 100, 1, 10))

        assert (cluster.MIRCH.spl_mdl_ratio(40, 100, 1, 25)
                < cluster.MIRCH.spl_mdl_ratio(40, 100, 1, 10))

        assert (cluster.MIRCH.spl_mdl_ratio(60, 100, -1, 25)
                < cluster.MIRCH.spl_mdl_ratio(60, 100, -1, 10))

        assert (cluster.MIRCH.spl_mdl_ratio(40, 100, -1, 25)
                > cluster.MIRCH.spl_mdl_ratio(40, 100, -1, 10))

    def test_spl_mdl_ratio_wrong_args(self):
        with pytest.raises(ValueError) as excinfo:
            cluster.MIRCH.spl_mdl_ratio(60, 100, 1, -1)
        with pytest.raises(ValueError) as excinfo:
            cluster.MIRCH.spl_mdl_ratio(60, 59, 1, 10)
        with pytest.raises(ValueError) as excinfo:
            cluster.MIRCH.spl_mdl_ratio(0, 100, 1, 10)
        with pytest.raises(ValueError) as excinfo:
            cluster.MIRCH.spl_mdl_ratio(-1, 100, 1, 10)
        with pytest.raises(ValueError) as excinfo:
            cluster.MIRCH.spl_mdl_ratio(-2, -1, 1, 10)

    def test_bi_split_compensation_factor(self):
        assert (cluster.MIRCH.bi_split_compensation_factor(100, 500, 25, 100)
                < cluster.MIRCH.bi_split_compensation_factor(101, 500, 25, 100))

        cluster.MIRCH.bi_split_compensation_factor(500, 500, 25, 100)
        
        np.testing.assert_allclose(
            cluster.MIRCH.bi_split_compensation_factor(25, 500, 25, 25), 
            cluster.MIRCH.bi_split_compensation_factor(100, 2000, 25, 50))

        np.testing.assert_allclose(
            cluster.MIRCH.bi_split_compensation_factor(24, 500, 25, 25), 
            cluster.MIRCH.bi_split_compensation_factor(24, 500, 25, 50))
        
        assert (cluster.MIRCH.bi_split_compensation_factor(100, 500, 25, 50)
                > cluster.MIRCH.bi_split_compensation_factor(99, 500, 25, 50))

        np.testing.assert_allclose(
            cluster.MIRCH.bi_split_compensation_factor(100, 500, 25, 100), 
            cluster.MIRCH.bi_split_compensation_factor(100, 500, 50, 250))

        # when minimax becomes larger, complexity decreases, need less 
        # compensation. The ratio affects factor value, 
        # but not the absolute count. 
        assert (cluster.MIRCH.bi_split_compensation_factor(100, 500, 25, 75)
                > cluster.MIRCH.bi_split_compensation_factor(100, 500, 26, 75 * 26/25))

        assert (cluster.MIRCH.bi_split_compensation_factor(100, 500, 25, 99)
                > cluster.MIRCH.bi_split_compensation_factor(100, 500, 25, 250))

        assert (cluster.MIRCH.bi_split_compensation_factor(100, 500, 25, 75)
                > cluster.MIRCH.bi_split_compensation_factor(100, 500, 25 * (76/75), 76))

        for i in range(1, 1001):
            x = cluster.MIRCH.bi_split_compensation_factor(i, 1000, 25, 250)
            assert 0 <= x <= 0.5

    def test_bi_split_compensation_factor_wrong_args(self):
        # 0 < stlc < n
        with pytest.raises(ValueError) as excinfo:
            cluster.MIRCH.bi_split_compensation_factor(0, 500, 25, 99)
        with pytest.raises(ValueError) as excinfo:
            cluster.MIRCH.bi_split_compensation_factor(-1, 500, 25, 99)
        with pytest.raises(ValueError) as excinfo:
            cluster.MIRCH.bi_split_compensation_factor(100, 99, 25, 99)
        with pytest.raises(ValueError) as excinfo:
            cluster.MIRCH.bi_split_compensation_factor(0, 1, 25, 99)

        # n > 0
        with pytest.raises(ValueError) as excinfo:
            cluster.MIRCH.bi_split_compensation_factor(100, -1, 25, 99)
        with pytest.raises(ValueError) as excinfo:
            cluster.MIRCH.bi_split_compensation_factor(-2, -1, 25, 99)

        # 0 < minimax < maxmini
        with pytest.raises(ValueError) as excinfo:
            cluster.MIRCH.bi_split_compensation_factor(100, 500, 0, 99)
        with pytest.raises(ValueError) as excinfo:
            cluster.MIRCH.bi_split_compensation_factor(100, 500, -1, 99)
        with pytest.raises(ValueError) as excinfo:
            cluster.MIRCH.bi_split_compensation_factor(100, 500, -2, -1)
        with pytest.raises(ValueError) as excinfo:
            cluster.MIRCH.bi_split_compensation_factor(100, 500, 2, -1)
        with pytest.raises(ValueError) as excinfo:
            cluster.MIRCH.bi_split_compensation_factor(100, 500, 2, 1)

    def test_mirch_wrong_args(self):
        x = np.zeros((10, 10))
        with pytest.raises(ValueError) as excinfo:
            cluster.MIRCH(x, metric="euclidean", minimax_n=-0.1)
        with pytest.raises(ValueError) as excinfo:
            cluster.MIRCH(x, metric="euclidean", minimax_n=25, maxmini_n=24)
        with pytest.raises(ValueError) as excinfo:
            cluster.MIRCH(x, metric="euclidean", minimax_n=-0.1)
        with pytest.raises(ValueError) as excinfo:
            cluster.MIRCH(x, metric="euclidean", cl_mdl_scale_factor=-0.1)

    # no specific purpose. Just to exaust the coverage
    def test_mirch_cover_tests(self):
        # make all non-neg
        x = np.zeros((10, 10))
        cluster.MIRCH(x, metric="euclidean", minimax_n=25, maxmini_n=250)
        cluster.MIRCH(x, metric="euclidean", minimax_n=25)
        cluster.MIRCH(x, metric="euclidean", minimax_n=25, maxmini_n=250, 
                      verbose=True)

        tx2, tlab2 = skdset.make_blobs(n_samples=500, n_features=5, 
                                       centers=100, cluster_std=1,
                                       random_state=8927)
        tx2 = tx2 - tx2.min()
        cluster.MIRCH(tx2, metric="correlation", minimax_n=1, maxmini_n=2,
                      cl_mdl_scale_factor=0)

