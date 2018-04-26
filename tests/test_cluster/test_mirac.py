import pytest

import numpy as np

import sklearn.datasets as skdset
from sklearn import metrics

import scedar.cluster as cluster
import scedar.eda as eda

import scipy.cluster.hierarchy as sch


class TestMIRAC(object):
    """docstring for TestMIRAC"""

    def test_mirac_wrong_args(self):
        x = np.zeros((10, 10))
        with pytest.raises(ValueError) as excinfo:
            cluster.MIRAC(x, metric='euclidean', min_cl_n=-0.1)

        with pytest.raises(ValueError) as excinfo:
            cluster.MIRAC(x, metric='euclidean', min_cl_n=-0.1)

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
        cluster.MIRAC(x, metric='euclidean', min_cl_n=25)
        cluster.MIRAC(x, metric='euclidean', min_cl_n=25)
        cluster.MIRAC(x, metric='euclidean', min_cl_n=25, verbose=True)

        tx, tlab = skdset.make_blobs(n_samples=100, n_features=2,
                                     centers=10, random_state=8927)
        tx = tx - tx.min()
        sdm = eda.SampleDistanceMatrix(tx, metric='euclidean')
        m = cluster.MIRAC(sdm._x, sdm._d, metric='euclidean', linkage='ward',
                          min_cl_n=35, cl_mdl_scale_factor=1, verbose=True)
        assert len(m.labs) == 100
        hct = eda.HClustTree.hclust_tree(sdm._d, linkage='ward',
                                         optimal_ordering=True)
        m2 = cluster.MIRAC(sdm._x, hac_tree=hct, min_cl_n=35,
                           cl_mdl_scale_factor=1, verbose=True)
        assert m.labs == m2.labs
        assert m2._sdm._lazy_load_d is None

        tx2, tlab2 = skdset.make_blobs(n_samples=500, n_features=5,
                                       centers=100, cluster_std=1,
                                       random_state=8927)
        tx2 = tx2 - tx2.min()
        cluster.MIRAC(tx2, metric='correlation', min_cl_n=3,
                      optimal_ordering=False, cl_mdl_scale_factor=0,
                      verbose=True)
