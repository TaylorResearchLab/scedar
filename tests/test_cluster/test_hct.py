import pytest

import numpy as np

import scedar.cluster as cluster
import scedar.eda as eda

class TestHClustTree(object):
    """docstring for TestHClustTree"""
    sdm_5x2 = eda.SampleDistanceMatrix([[0, 0],
                                        [100, 100],
                                        [1, 1],
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

    def test_cluster_id_to_lab_list_wrong_id_list_type(self):
        with pytest.raises(ValueError) as excinfo:
            cluster.HClustTree.cluster_id_to_lab_list(
                np.array([[0, 1, 2], [3, 4]]), [0, 1, 2, 3, 4])

    def test_cluster_id_to_lab_list_mismatched_ids_sids(self):
        with pytest.raises(ValueError) as excinfo:
            cluster.HClustTree.cluster_id_to_lab_list(
                [[0, 1, 2], [3, 4]], [0, 1, 2, 3, 5])

    def test_cluster_id_to_lab_list_empty_cluster(self):
        with pytest.raises(ValueError) as excinfo:
            cluster.HClustTree.cluster_id_to_lab_list(
                [[], [0, 1, 2, 3, 4]], [0, 1, 2, 3, 4])
