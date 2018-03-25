import pytest

import numpy as np

import sklearn.datasets as skdset

import scedar.cluster as cluster
import scedar.eda as eda


class TestLargeDataMIRAC(object):
    """docstring for TestMIRAC"""
    x300x1000, labs300 = skdset.make_blobs(n_samples=300, n_features=1000,
                                           centers=5, random_state=8927)
    x300x1000 = ((x300x1000 - x300x1000.min()) * 100).astype(int)

    x100x500, labs100 = skdset.make_blobs(n_samples=100, n_features=500,
                                          centers=2, random_state=8927)
    x100x500 = ((x100x500 - x100x500.min()) * 100).astype(int)

    def test_mirac_run_300x1000(self):
        mirac_res = cluster.MIRAC(self.x300x1000, metric="correlation",
                                  nprocs=3)
        ulabs, ulab_cnts = np.unique(mirac_res._cluster_labs,
                                     return_counts=True)
        np.testing.assert_equal(ulab_cnts, [60]*5)
        mirac_res_ulab_ind = [mirac_res._cluster_s_ind[mirac_res._cluster_labs == ulab]
                              for ulab in ulabs]
        for i in mirac_res_ulab_ind:
            assert len(np.unique(self.labs300[i])) == 1
        assert sorted(mirac_res._cluster_s_ind.tolist()) == list(range(300))

    def test_mirac_run_100x500(self):
        mirac_res = cluster.MIRAC(self.x100x500, metric="correlation",
                                  nprocs=1)
        ulabs, ulab_cnts = np.unique(mirac_res._cluster_labs,
                                     return_counts=True)
        np.testing.assert_equal(ulab_cnts, [50]*2)
        mirac_res_ulab_ind = [mirac_res._cluster_s_ind[mirac_res._cluster_labs == ulab]
                              for ulab in ulabs]
        for i in mirac_res_ulab_ind:
            assert len(np.unique(self.labs100[i])) == 1
        assert sorted(mirac_res._cluster_s_ind.tolist()) == list(range(100))
