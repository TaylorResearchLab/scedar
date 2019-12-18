import pytest

import numpy as np

import scipy.sparse as spsp

import scedar.cluster as cluster
import scedar.eda as eda

class TestCommunityMIRAC(object):
    '''docstring for TestCommunityMIRAC'''
    np.random.seed(123)
    x_20x5 = np.random.uniform(size=(20, 5))

    def test_array_run(self):
        cm_mirac = cluster.CommunityMIRAC(self.x_20x5)
        cm_mirac.run()
        cm_mirac.labs
        cluster.CommunityMIRAC(self.x_20x5, verbose=True).run()

    def test_csr_run(self):
        cm_mirac = cluster.CommunityMIRAC(spsp.csr_matrix(self.x_20x5))
        cm_mirac.run()
        cm_mirac.labs
        cluster.CommunityMIRAC(
            spsp.csr_matrix(self.x_20x5), verbose=True).run()

    def test_collapse_clusters(self):
        data = np.array([[ 1,  8],
                         [ 3,  7],
                         [ 2,  4],
                         [ 0,  3],
                         [ 5, 17]])
        labs = [0, 2, 0, 1, 1]
        ref_res = np.array([[1.5, 6],
                            [2.5, 10],
                            [3, 7]])
        m_res = cluster.CommunityMIRAC.collapse_clusters(
            data, labs)
        np.testing.assert_equal(m_res, ref_res)

    def test_collapse_clusters_wrong_labs(self):
        data = np.array([[ 1,  8],
                         [ 0,  3],
                         [ 2,  4],
                         [ 3,  7],
                         [ 5, 17]])
        labs = [0, 1, 0, 3, 1]
        with pytest.raises(ValueError):
            cluster.CommunityMIRAC.collapse_clusters(data, labs)

    def test_collapse_clusters_wrong_exec_order(self):
        with pytest.raises(ValueError):
            cluster.CommunityMIRAC(self.x_20x5).run_mirac()
