import pytest

import numpy as np

import sklearn.datasets as skdset
from sklearn import metrics

import scedar.cluster as cluster
import scedar.eda as eda

import scipy.cluster.hierarchy as sch


class TestMIRAC(object):
    '''docstring for TestMIRAC'''

    def test_mirac_wrong_args(self):
        x = np.zeros((10, 10))
        # wrong min_cl_n
        with pytest.raises(ValueError) as excinfo:
            cluster.MIRAC(x, metric='euclidean', min_cl_n=-0.1)

        with pytest.raises(ValueError) as excinfo:
            cluster.MIRAC(x, metric='euclidean', min_cl_n=-0.1)
        # wrong cl_mdl_scale_factor
        with pytest.raises(ValueError) as excinfo:
            cluster.MIRAC(x, metric='euclidean', cl_mdl_scale_factor=-0.1)
        # wrong encode type
        with pytest.raises(ValueError) as excinfo:
            cluster.MIRAC(x, metric='euclidean', encode_type='1')

        with pytest.raises(ValueError) as excinfo:
            cluster.MIRAC(x, metric='euclidean', encode_type=1)

        with pytest.raises(ValueError) as excinfo:
            cluster.MIRAC(x, metric='euclidean', dim_reduct_method='NONN')

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
                           cl_mdl_scale_factor=1, encode_type='data',
                           mdl_method=eda.mdl.ZeroIGKdeMdl, verbose=False)
        assert m.labs == m2.labs
        assert m2._sdm._lazy_load_d is None

        tx2, tlab2 = skdset.make_blobs(n_samples=500, n_features=5,
                                       centers=5, cluster_std=1,
                                       random_state=8927)
        tx2 = tx2 - tx2.min()
        cluster.MIRAC(tx2, metric='euclidean', min_cl_n=15,
                      optimal_ordering=False, cl_mdl_scale_factor=0,
                      verbose=True)
        # auto to encode distance
        tx3, tlab3 = skdset.make_blobs(n_samples=100, n_features=101,
                                       cluster_std=0.01, centers=5,
                                       random_state=8927)
        tx3 = tx3 - tx3.min()
        sdm = eda.SampleDistanceMatrix(tx3, metric='euclidean')
        m = cluster.MIRAC(sdm._x, sdm._d, metric='euclidean', linkage='ward',
                          min_cl_n=5, cl_mdl_scale_factor=1, verbose=True)
        assert m._encode_type == 'distance'
        # empty
        sdm = eda.SampleDistanceMatrix([[], []], metric='euclidean')
        m = cluster.MIRAC(sdm._x, sdm._d, metric='euclidean', linkage='ward',
                          min_cl_n=35, cl_mdl_scale_factor=1, verbose=True)


    def test_mirac_dim_reduction(self):
        tx, tlab = skdset.make_blobs(n_samples=100, n_features=2,
                                     centers=10, random_state=8927)
        tx = tx - tx.min()
        sdm = eda.SampleDistanceMatrix(tx, metric='euclidean')
        m = cluster.MIRAC(sdm._x, sdm._d, metric='euclidean', linkage='ward',
                          min_cl_n=35, cl_mdl_scale_factor=1,
                          dim_reduct_method='PCA', verbose=True)
        assert len(m.labs) == 100
        hct = eda.HClustTree.hclust_tree(sdm._d, linkage='ward',
                                         optimal_ordering=True)
        m2 = cluster.MIRAC(sdm._x, hac_tree=hct, min_cl_n=35,
                           cl_mdl_scale_factor=1, encode_type='data',
                           mdl_method=eda.mdl.ZeroIGKdeMdl,
                           dim_reduct_method='t-SNE', verbose=False)
        assert m.labs == m2.labs
        assert m2._sdm._lazy_load_d is None

        tx2, tlab2 = skdset.make_blobs(n_samples=500, n_features=50,
                                       centers=5, cluster_std=15,
                                       random_state=8927)
        tx2 = tx2 - tx2.min()
        cluster.MIRAC(tx2, metric='euclidean', min_cl_n=15,
                      optimal_ordering=False, cl_mdl_scale_factor=0,
                      dim_reduct_method='UMAP', verbose=True)
        # auto to encode distance
        tx3, tlab3 = skdset.make_blobs(n_samples=100, n_features=101,
                                       cluster_std=0.01, centers=5,
                                       random_state=8927)
        tx3 = tx3 - tx3.min()
        sdm = eda.SampleDistanceMatrix(tx3, metric='euclidean')
        m = cluster.MIRAC(sdm._x, sdm._d, metric='euclidean', linkage='ward',
                          min_cl_n=5, cl_mdl_scale_factor=1,
                          dim_reduct_method='t-SNE', verbose=True)
        # empty
        sdm = eda.SampleDistanceMatrix([[], []], metric='euclidean')

        with pytest.raises(Exception) as excinfo:
            m = cluster.MIRAC(sdm._x, sdm._d, metric='euclidean',
                              linkage='ward',
                              min_cl_n=35, cl_mdl_scale_factor=1,
                              dim_reduct_method='PCA', verbose=True)

    def test_mirac_rerun(self):
        tx, tlab = skdset.make_blobs(n_samples=100, n_features=2,
                                     centers=10, random_state=8927)
        tx = tx - tx.min()
        m = cluster.MIRAC(tx, min_cl_n=35,
                           cl_mdl_scale_factor=1, encode_type='data',
                           mdl_method=eda.mdl.ZeroIGKdeMdl,
                           dim_reduct_method='t-SNE', verbose=False)
        m_tsne = m._sdm._x
        m.tune_parameters(0, 5, 0.1, 1)
        assert m_tsne is m._sdm._x

    @pytest.mark.mpl_image_compare
    def test_mirac_dmat_heatmap(self):
        # create a dummy MIRAC object
        sdm = eda.SampleDistanceMatrix([[], []], metric='euclidean')
        m = cluster.MIRAC(sdm._x, sdm._d, metric='euclidean', linkage='ward',
                          min_cl_n=35, cl_mdl_scale_factor=1, verbose=True)
        # empty tree
        m._hac_tree = eda.HClustTree(None)
        assert m.dmat_heatmap() is None
        # normal tree and d
        sdm_5x2 = eda.SampleDistanceMatrix([[0, 0],
                                            [100, 100],
                                            [1, 1],
                                            [101, 101],
                                            [80, 80]],
                                           metric='euclidean')
        # This tree should be
        #   _______|_____
        #   |       ____|___
        # __|___    |    __|___
        # |    |    |    |    |
        # 0    2    4    1    3
        # Leaves are in optimal order.
        hct = eda.HClustTree.hclust_tree(sdm_5x2.d, linkage='auto')
        m._hac_tree = hct
        m._sdm = sdm_5x2

        # invalid labels
        # leaf order: [0, 2, 4, 1, 3]
        # leaf labs : [1, 1, 2, 3, 2]
        m._labs = [1, 3, 1, 2, 2]
        with pytest.raises(ValueError) as excinfo:
            m.dmat_heatmap()
        m._labs = ['1', '3', '1', '2', '2']
        with pytest.raises(ValueError) as excinfo:
            m.dmat_heatmap()
        # leaf order: [0, 2, 4, 1, 3]
        # leaf labs : [5, 1, 2, 1, 2]
        m._labs = [5, 1, 1, 2, 2]
        with pytest.raises(ValueError) as excinfo:
            m.dmat_heatmap()
        # leaf order: [0, 2, 4, 1, 3]
        # leaf labs : [5, 1, 2, 3, 2]
        m._labs = [5, 3, 1, 2, 2]
        with pytest.raises(ValueError) as excinfo:
            m.dmat_heatmap()
        # leaf order: [0, 2, 4, 1, 3]
        # leaf labs : [5, 10, 20, 3, 5]
        m._labs = [5, 3, 10, 5, 20]
        with pytest.raises(ValueError) as excinfo:
            m.dmat_heatmap()

        # valid labels
        # leaf order: [0, 2, 4, 1, 3]
        # leaf labs : [2, 2, 1, 1, 0]
        m._labs = [2, 1, 2, 0, 1]
        m.dmat_heatmap()
        m._labs = ['2', '1', '2', '0', '1']
        m.dmat_heatmap()
        # leaf order: [0, 2, 4, 1, 3]
        # leaf labs : [0, 2, 4, 1, 3]
        m._labs = [0, 1, 2, 3, 4]
        m.dmat_heatmap()
        # leaf order: [0, 2, 4, 1, 3]
        # leaf labs : [0, 0, 0, 0, 0]
        m._labs = [0]*5
        m.dmat_heatmap()

        # have a figure
        # leaf order: [ 0 ,  2 ,  4 ,  1 ,  3 ]
        # leaf labs : ['b', 'l', 'r', 'e', 'o']
        m._labs = ['beginning', 'end', 'l', 'out', 'r']
        fig = m.dmat_heatmap(
            selected_labels=['beginning', 'end', 'l', 'r'])
        assert fig is not None
        return fig
