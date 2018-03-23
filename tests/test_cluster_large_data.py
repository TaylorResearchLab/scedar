import pytest

import numpy as np

import sklearn.datasets as skdset

import scxplit.cluster as cluster
import scxplit.eda as eda

class TestMIRCH(object):
    """docstring for TestMIRCH"""
    x300x1000, labs300 = skdset.make_blobs(n_samples=300, n_features=1000, 
                                           centers=5, random_state=8927)
    x300x1000 = x300x1000 - x300x1000.min()

    x100x500, labs100 = skdset.make_blobs(n_samples=100, n_features=500, 
                                           centers=2, random_state=8927)
    x100x500 = x100x500 - x100x500.min()

    def test_mirch_run(self):
        mirch_res = cluster.MIRCH(self.x300x1000, metric="correlation",
                                  nprocs=1)
        ulabs, ulab_cnts = np.unique(mirch_res._cluster_labs, 
                                     return_counts=True)
        np.testing.assert_equal(ulab_cnts, [60]*5)
        mirch_res_ulab_ind = [mirch_res._cluster_s_ind[mirch_res._cluster_labs == ulab]
                              for ulab in ulabs]
        for i in mirch_res_ulab_ind:
            assert len(np.unique(self.labs300[i])) == 1
        assert sorted(mirch_res._cluster_s_ind.tolist()) == list(range(300))


    def test_mirch_run(self):
        mirch_res = cluster.MIRCH(self.x100x500, metric="correlation",
                                  nprocs=1)
        ulabs, ulab_cnts = np.unique(mirch_res._cluster_labs, 
                                     return_counts=True)
        np.testing.assert_equal(ulab_cnts, [50]*2)
        mirch_res_ulab_ind = [mirch_res._cluster_s_ind[mirch_res._cluster_labs == ulab]
                              for ulab in ulabs]
        for i in mirch_res_ulab_ind:
            assert len(np.unique(self.labs100[i])) == 1
        assert sorted(mirch_res._cluster_s_ind.tolist()) == list(range(100))
