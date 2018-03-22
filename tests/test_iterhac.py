import pytest

import numpy as np

import scxplit.iterhac as ihac
import scxplit.eda as eda

class TestMultinomialMdl(object):
    """docstring for TestMultinomialMdl"""
    def test_empty_x(self):
        mmdl = ihac.MultinomialMdl([])
        assert mmdl.mdl == 0

    def test_single_level(self):
        mmdl = ihac.MultinomialMdl(["a"]*10)
        np.testing.assert_allclose(mmdl.mdl, np.log(10))

    def test_multi_levels(self):
        x = ["a"]*10 + ["b"]*25
        ux, uxcnt = np.unique(x, return_counts=True)
        mmdl = ihac.MultinomialMdl(x)
        np.testing.assert_allclose(mmdl.mdl, 
            (-np.log(uxcnt / len(x)) * uxcnt).sum())

    def test_wrong_x_shape(self):
        with pytest.raises(ValueError) as excinfo:
            ihac.MultinomialMdl(np.arange(6).reshape(3, 2))

    def test_getter(self):
        mmdl = ihac.MultinomialMdl([])
        assert mmdl.x == []
        mmdl2 = ihac.MultinomialMdl([0, 0, 1, 1, 1])
        assert mmdl2.x == [0, 0, 1, 1, 1]



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
    hct = ihac.HClustTree.hclust_tree(sdm_5x2.d, linkage="auto")
    
    def test_hclust_tree_args(self):
        ihac.HClustTree.hclust_tree(self.sdm_5x2.d, linkage="auto",
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
            ihac.HClustTree.hclust_tree(np.arange(5))

        with pytest.raises(ValueError) as excinfo:
            ihac.HClustTree.hclust_tree(np.arange(10).reshape(2, 5))

    def test_bi_partition(self):
        # return subtrees False
        clabs1, cids1 = self.hct.bi_partition()

        # return subtrees True
        clasb2, cids2, lst, rst = self.hct.bi_partition(return_subtrees=True)

        assert clabs1 == clasb2
        assert cids1 == cids2

        assert lst.count() == 2
        assert lst.left_count() == 1
        assert lst.left_leaf_ids() == [0]
        assert lst.right_leaf_ids() == [2]
        assert lst.leaf_ids() == [0, 2]

        assert rst.leaf_ids() == [4, 1, 3]
        assert rst.right_leaf_ids() == [1, 3]
        assert rst.left_leaf_ids() == [4]

    def test_cluster_id_list_to_clab_array_wrong_id_list_type(self):
        with pytest.raises(ValueError) as excinfo:
            ihac.HClustTree.cluster_id_list_to_clab_array(
                np.array([[0, 1, 2], [3,4]]), [0, 1, 2, 3, 4])

    def test_cluster_id_list_to_clab_array_mismatched_ids_cids(self):
        with pytest.raises(ValueError) as excinfo:
            ihac.HClustTree.cluster_id_list_to_clab_array(
                [[0, 1, 2], [3,4]], [0, 1, 2, 3, 5])


