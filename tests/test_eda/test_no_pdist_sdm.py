import numpy as np

import matplotlib as mpl
mpl.use("agg", warn=False)  # noqa
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn.metrics.pairwise

import scipy.cluster.hierarchy as sch
import scipy.sparse as spsp

import scedar.eda as eda

import pytest



class TestNoPdistSampleDistanceMatrix(object):
    """docstring for TestSampleDistanceMatrix"""
    x_3x2 = spsp.csr_matrix([[0, 0], [1, 1], [2, 2]])
    x_2x4_spsp = spsp.csr_matrix(np.array([[0, 1, 2, 3], [1, 2, 0, 6]]))
    x_2x4_arr = np.array([[0, 1, 2, 3], [1, 2, 0, 6]])

    def test_valid_init(self):
        sdm = eda.SampleDistanceMatrix(self.x_3x2, metric='euclidean',
                                       use_pdist=False)
        with pytest.raises(ValueError) as excinfo:
            sdm.d

        with pytest.raises(ValueError) as excinfo:
            sdm3 = eda.SampleDistanceMatrix(
                self.x_2x4_spsp, metric='correlation', use_pdist=False,
                nprocs=5).d

        dist_mat = np.array([[0, np.sqrt(2), np.sqrt(8)],
                             [np.sqrt(2), 0, np.sqrt(2)],
                             [np.sqrt(8), np.sqrt(2), 0]])

        with pytest.raises(ValueError) as excinfo:
            sdm4 = eda.SampleDistanceMatrix(
                self.x_3x2, dist_mat, use_pdist=False)
        sdm5 = eda.SampleDistanceMatrix([[1, 2]], metric='euclidean',
                                        use_pdist=False)
        assert sdm5.tsne(n_iter=250).shape == (1, 2)

    def test_empty_init(self):
        with pytest.raises(ValueError) as excinfo:
            eda.SampleDistanceMatrix(np.empty(0), metric='euclidean')
        sdm = eda.SampleDistanceMatrix(
            np.empty((0, 0)), metric='euclidean', use_pdist=False)
        assert len(sdm.sids) == 0
        assert len(sdm.fids) == 0
        assert sdm._x.shape == (0, 0)
        with pytest.raises(ValueError) as excinfo:
            assert sdm._d.shape == (0, 0)
        with pytest.raises(ValueError) as excinfo:
            assert sdm._col_sorted_d.shape == (0, 0)
        with pytest.raises(ValueError) as excinfo:
            assert sdm._col_argsorted_d.shape == (0, 0)
        assert sdm.tsne(n_iter=250).shape == (0, 0)

    def test_init_wrong_metric(self):
        # when d is None, metric cannot be precomputed
        with pytest.raises(Exception) as excinfo:
            eda.SampleDistanceMatrix(
                self.x_3x2, metric='precomputed', use_pdist=False)

        # lazy load d
        eda.SampleDistanceMatrix(self.x_3x2, metric='unknown', use_pdist=False)
        with pytest.raises(Exception) as excinfo:
            eda.SampleDistanceMatrix(
                self.x_3x2, metric='unknown', use_pdist=False).d

        eda.SampleDistanceMatrix(self.x_3x2, metric=1, use_pdist=False)
        with pytest.raises(Exception) as excinfo:
            eda.SampleDistanceMatrix(self.x_3x2, metric=1, use_pdist=False).d

        eda.SampleDistanceMatrix(self.x_3x2, metric=1., use_pdist=False)
        with pytest.raises(Exception) as excinfo:
            eda.SampleDistanceMatrix(self.x_3x2, metric=1., use_pdist=False).d

        eda.SampleDistanceMatrix(self.x_3x2, metric=('euclidean', ),
                                 use_pdist=False)
        with pytest.raises(Exception) as excinfo:
            eda.SampleDistanceMatrix(
                self.x_3x2, metric=('euclidean', ), use_pdist=False).d

        eda.SampleDistanceMatrix(self.x_3x2, metric=['euclidean'],
                                 use_pdist=False)
        with pytest.raises(Exception) as excinfo:
            eda.SampleDistanceMatrix(self.x_3x2, metric=['euclidean'],
                                     use_pdist=False).d

    def test_sort_features(self):
        x = np.array([[0, 2, 30, 10],
                      [1, 2, 30, 10],
                      [0, 3, 33, 10],
                      [2, 5, 30, 7],
                      [2, 5, 30, 9]])
        x = spsp.csr_matrix(x)
        sdm = eda.SampleDistanceMatrix(
            x, metric='euclidean', use_pdist=False)
        sdm2 = eda.SampleDistanceMatrix(
            x, metric='euclidean', use_pdist=False)
        sdm2.sort_features(fdist_metric='euclidean')
        assert sdm2.fids == [2, 3, 1, 0]

    def test_get_tsne_kv(self):
        tmet = 'euclidean'
        sdm = eda.SampleDistanceMatrix(
            self.x_3x2, metric=tmet, use_pdist=False)
        assert sdm.get_tsne_kv(1) is None
        assert sdm.get_tsne_kv(1) is None
        assert sdm.get_tsne_kv(0) is None
        assert sdm.get_tsne_kv(2) is None

    def test_get_tsne_kv_wrong_args(self):
        tmet = 'euclidean'
        sdm = eda.SampleDistanceMatrix(
            self.x_3x2, metric=tmet, use_pdist=False)
        with pytest.raises(ValueError) as excinfo:
            sdm.get_tsne_kv([1, 2, 3])
        with pytest.raises(ValueError) as excinfo:
            sdm.get_tsne_kv({1: 2})

    def test_put_tsne_wrong_args(self):
        tmet = 'euclidean'
        sdm = eda.SampleDistanceMatrix(
            self.x_3x2, metric=tmet, use_pdist=False)
        with pytest.raises(ValueError) as excinfo:
            sdm.put_tsne(1, [1, 2, 3])
        with pytest.raises(ValueError) as excinfo:
            sdm.put_tsne({1: 2}, [1, 2, 3])

    def test_tsne(self):
        tmet = 'euclidean'
        tsne_kwargs = {'metric': tmet, 'n_iter': 250,
                       'random_state': 123}
        ref_tsne = eda.tsne(self.x_3x2.toarray(), **tsne_kwargs)
        sdm = eda.SampleDistanceMatrix(
            self.x_3x2, metric=tmet, use_pdist=False)

        assert sdm.tsne_lut == {}
        tsne1 = sdm.tsne(n_iter=250, random_state=123)
        np.testing.assert_allclose(ref_tsne, tsne1)
        np.testing.assert_allclose(ref_tsne, sdm._last_tsne)
        assert tsne1.shape == (3, 2)
        assert len(sdm.tsne_lut) == 1

        tsne2 = sdm.tsne(store_res=False, **tsne_kwargs)
        np.testing.assert_allclose(ref_tsne, tsne2)
        assert len(sdm.tsne_lut) == 1

        with pytest.raises(Exception) as excinfo:
            wrong_metric_kwargs = tsne_kwargs.copy()
            wrong_metric_kwargs['metric'] = 'correlation'
            sdm.tsne(**wrong_metric_kwargs)
        assert len(sdm.tsne_lut) == 1

        with pytest.raises(Exception) as excinfo:
            sdm.tsne(metric='precomputed')

        tsne3 = sdm.tsne(store_res=True, **tsne_kwargs)
        np.testing.assert_allclose(ref_tsne, tsne3)
        # (param, ind) as key, so same params get an extra entry.
        assert len(sdm.tsne_lut) == 2

        np.testing.assert_allclose(tsne1, sdm.get_tsne_kv(1)[1])
        np.testing.assert_allclose(tsne3, sdm.get_tsne_kv(2)[1])
        assert tsne1 is not sdm.get_tsne_kv(1)[1]
        assert tsne3 is not sdm.get_tsne_kv(2)[1]

        tsne4 = sdm.tsne(store_res=True, n_iter=250, random_state=123)
        np.testing.assert_allclose(ref_tsne, tsne4)
        np.testing.assert_allclose(sdm.get_tsne_kv(3)[1], tsne4)
        assert len(sdm.tsne_lut) == 3

        tsne5 = sdm.tsne(store_res=True, n_iter=251, random_state=123)
        tsne6 = sdm.tsne(store_res=True, n_iter=251, random_state=123)
        np.testing.assert_allclose(tsne6, tsne5)
        np.testing.assert_allclose(tsne5, sdm.get_tsne_kv(4)[1])
        np.testing.assert_allclose(tsne6, sdm.get_tsne_kv(5)[1])
        assert len(sdm.tsne_lut) == 5

    def test_par_tsne(self):
        tmet = 'euclidean'
        param_list = [{'metric': tmet, 'n_iter': 250, 'random_state': 123},
                      {'metric': tmet, 'n_iter': 250, 'random_state': 125},
                      {'metric': tmet, 'n_iter': 250, 'random_state': 123}]
        ref_tsne = eda.tsne(self.x_3x2.toarray(), **param_list[0])
        sdm = eda.SampleDistanceMatrix(
            self.x_3x2, metric=tmet, use_pdist=False)
        # If not store, should not update lut
        sdm.par_tsne(param_list, store_res=False)
        assert sdm._lazy_load_last_tsne is None
        assert sdm.tsne_lut == {}
        # store results
        tsne1, tsne2, tsne3 = sdm.par_tsne(param_list)
        np.testing.assert_allclose(ref_tsne, tsne1)
        np.testing.assert_allclose(ref_tsne, tsne3)
        np.testing.assert_allclose(ref_tsne, sdm._last_tsne)
        assert tsne1.shape == (3, 2)
        assert len(sdm.tsne_lut) == 3

        np.testing.assert_allclose(tsne1, sdm.get_tsne_kv(1)[1])
        np.testing.assert_allclose(tsne2, sdm.get_tsne_kv(2)[1])
        np.testing.assert_allclose(tsne3, sdm.get_tsne_kv(3)[1])
        np.testing.assert_allclose(tsne3, sdm.get_tsne_kv(1)[1])

    def test_par_tsne_mp(self):
        tmet = 'euclidean'
        param_list = [{'metric': tmet, 'n_iter': 250, 'random_state': 123},
                      {'metric': tmet, 'n_iter': 250, 'random_state': 125},
                      {'metric': tmet, 'n_iter': 250, 'random_state': 123}]
        ref_tsne = eda.tsne(self.x_3x2.toarray(), **param_list[0])
        sdm = eda.SampleDistanceMatrix(
            self.x_3x2, metric=tmet, use_pdist=False)
        # If not store, should not update lut
        sdm.par_tsne(param_list, store_res=False, nprocs=3)
        assert sdm._lazy_load_last_tsne is None
        assert sdm.tsne_lut == {}
        # store results
        tsne1, tsne2, tsne3 = sdm.par_tsne(param_list, nprocs=3)
        np.testing.assert_allclose(ref_tsne, tsne1)
        np.testing.assert_allclose(ref_tsne, tsne3)
        np.testing.assert_allclose(ref_tsne, sdm._last_tsne)
        assert tsne1.shape == (3, 2)
        assert len(sdm.tsne_lut) == 3

        np.testing.assert_allclose(tsne1, sdm.get_tsne_kv(1)[1])
        np.testing.assert_allclose(tsne2, sdm.get_tsne_kv(2)[1])
        np.testing.assert_allclose(tsne3, sdm.get_tsne_kv(3)[1])
        np.testing.assert_allclose(tsne3, sdm.get_tsne_kv(1)[1])

    def test_tsne_default_init(self):
        tmet = 'euclidean'
        tsne_kwargs = {'metric': tmet, 'n_iter': 250,
                       'random_state': 123}
        ref_tsne = eda.tsne(self.x_3x2.toarray(), **tsne_kwargs)
        sdm = eda.SampleDistanceMatrix(
            self.x_3x2, metric=tmet, use_pdist=False)

        init_tsne = sdm._last_tsne

        assert init_tsne.shape == (3, 2)
        assert len(sdm.tsne_lut) == 1

        tsne2 = sdm.tsne(store_res=True, **tsne_kwargs)
        np.testing.assert_allclose(ref_tsne, tsne2)
        assert len(sdm.tsne_lut) == 2

    def test_ind_x(self):
        sids = list("abcdef")
        fids = list(range(10, 20))
        sdm = eda.SampleDistanceMatrix(
            np.random.ranf(60).reshape(6, -1), 
            sids=sids, fids=fids, use_pdist=False)
        # select sf
        ss_sdm = sdm.ind_x([0, 5], list(range(9)))
        assert ss_sdm._x.shape == (2, 9)
        assert ss_sdm.sids == ['a', 'f']
        assert ss_sdm.fids == list(range(10, 19))

        with pytest.raises(Exception) as excinfo:
            ss_sdm.d

        # select with Default
        ss_sdm = sdm.ind_x()
        assert ss_sdm._x.shape == (6, 10)
        assert ss_sdm.sids == list("abcdef")
        assert ss_sdm.fids == list(range(10, 20))
        with pytest.raises(Exception) as excinfo:
            ss_sdm.d

        # select with None
        ss_sdm = sdm.ind_x(None, None)
        assert ss_sdm._x.shape == (6, 10)
        assert ss_sdm.sids == list("abcdef")
        assert ss_sdm.fids == list(range(10, 20))
        with pytest.raises(Exception) as excinfo:
            ss_sdm.d

        # select non-existent inds
        with pytest.raises(IndexError) as excinfo:
            sdm.ind_x([6])

        with pytest.raises(IndexError) as excinfo:
            sdm.ind_x(None, ['a'])

    def test_ind_x_empty(self):
        sids = list("abcdef")
        fids = list(range(10, 20))
        sdm = eda.SampleDistanceMatrix(
            np.random.ranf(60).reshape(6, -1), sids=sids, fids=fids,
            use_pdist=False)
        empty_s = sdm.ind_x([])
        assert empty_s._x.shape == (0, 10)
        with pytest.raises(Exception) as excinfo:
            empty_s._d
        assert empty_s._sids.shape == (0,)
        assert empty_s._fids.shape == (10,)

        empty_f = sdm.ind_x(None, [])
        assert empty_f._x.shape == (6, 0)
        with pytest.raises(Exception) as excinfo:
            empty_f._d
        assert empty_f._sids.shape == (6,)
        assert empty_f._fids.shape == (0,)

        empty_sf = sdm.ind_x([], [])
        assert empty_sf._x.shape == (0, 0)
        with pytest.raises(Exception) as excinfo:
            empty_sf._d
        assert empty_sf._sids.shape == (0,)
        assert empty_sf._fids.shape == (0,)

    def test_id_x(self):
        sids = list("abcdef")
        fids = list(range(10, 20))
        sdm = eda.SampleDistanceMatrix(
            np.random.ranf(60).reshape(6, -1),
            sids=sids, fids=fids, use_pdist=False)
        # select sf
        ss_sdm = sdm.id_x(['a', 'f'], list(range(10, 15)))
        assert ss_sdm._x.shape == (2, 5)
        assert ss_sdm.sids == ['a', 'f']
        assert ss_sdm.fids == list(range(10, 15))
        with pytest.raises(Exception) as excinfo:
            ss_sdm.d

        # select with Default
        ss_sdm = sdm.id_x()
        assert ss_sdm._x.shape == (6, 10)
        assert ss_sdm.sids == list("abcdef")
        assert ss_sdm.fids == list(range(10, 20))
        with pytest.raises(Exception) as excinfo:
            ss_sdm.d

        # select with None
        ss_sdm = sdm.id_x(None, None)
        assert ss_sdm._x.shape == (6, 10)
        assert ss_sdm.sids == list("abcdef")
        assert ss_sdm.fids == list(range(10, 20))
        with pytest.raises(Exception) as excinfo:
            ss_sdm.d

        # select non-existent inds
        # id lookup raises ValueError
        with pytest.raises(ValueError) as excinfo:
            sdm.id_x([6])

        with pytest.raises(ValueError) as excinfo:
            sdm.id_x(None, ['a'])

    def test_id_x_empty(self):
        sids = list("abcdef")
        fids = list(range(10, 20))
        sdm = eda.SampleDistanceMatrix(
            np.random.ranf(60).reshape(6, -1),
            sids=sids, fids=fids, use_pdist=False)
        empty_s = sdm.id_x([])
        assert empty_s._x.shape == (0, 10)
        with pytest.raises(Exception) as excinfo:
            empty_s._d
        assert empty_s._sids.shape == (0,)
        assert empty_s._fids.shape == (10,)

        empty_f = sdm.id_x(None, [])
        assert empty_f._x.shape == (6, 0)
        with pytest.raises(Exception) as excinfo:
            empty_f._d
        assert empty_f._sids.shape == (6,)
        assert empty_f._fids.shape == (0,)

        empty_sf = sdm.id_x([], [])
        assert empty_sf._x.shape == (0, 0)
        with pytest.raises(Exception) as excinfo:
            empty_sf._d
        assert empty_sf._sids.shape == (0,)
        assert empty_sf._fids.shape == (0,)

    def test_getter(self):
        tmet = 'euclidean'
        sdm = eda.SampleDistanceMatrix(
            self.x_3x2, metric=tmet, use_pdist=False)
        with pytest.raises(Exception) as excinfo:
            sdm.d
        assert sdm.metric == tmet
        assert sdm.tsne_lut == {}
        assert sdm.tsne_lut is not sdm._tsne_lut
        assert sdm.tsne_lut == sdm._tsne_lut
        sdm.tsne(n_iter=250)
        assert sdm.tsne_lut is not sdm._tsne_lut
        for k in sdm.tsne_lut:
            np.testing.assert_equal(sdm.tsne_lut[k], sdm._tsne_lut[k])

    def test_s_ith_nn_d(self):
        nn_sdm = eda.SampleDistanceMatrix([[0], [1], [5], [6], [10], [20]],
                                          metric='euclidean', use_pdist=False)
        with pytest.raises(ValueError) as excinfo:
            nn_sdm.s_ith_nn_d(0)

    def test_s_ith_nn_ind(self):
        nn_sdm = eda.SampleDistanceMatrix([[0, 0, 0], [1, 1, 1], [5, 5, 5],
                                           [6, 6, 6], [10, 10, 10],
                                           [20, 20, 20]],
                                          metric='euclidean', 
                                          use_pdist=False)
        with pytest.raises(ValueError) as excinfo:
            nn_sdm.s_ith_nn_ind(0)

    def test_knn_ind_lut(self):
        nn_sdm = eda.SampleDistanceMatrix([[0, 0, 0], [1, 1, 1], [5, 5, 5],
                                           [6, 6, 6], [10, 10, 10],
                                           [20, 20, 20]],
                                          metric='euclidean', use_pdist=False)
        with pytest.raises(ValueError) as excinfo:
            nn_sdm.s_knn_ind_lut(0)

    @pytest.mark.mpl_image_compare
    def test_sdm_tsne_feature_gradient_plot(self):
        sids = list(range(8))
        fids = [str(i) for i in range(10)]
        np.random.seed(123)
        x = np.random.ranf(80).reshape(8, -1)
        x_sorted = x[np.argsort(x[:, 5])]
        sdm = eda.SampleDistanceMatrix(
            x_sorted, sids=sids, fids=fids, use_pdist=False)
        fig = sdm.tsne_feature_gradient_plot(
            '5', figsize=(10, 10), s=50)
        np.testing.assert_equal(sdm._x, x_sorted)
        np.testing.assert_equal(sdm._sids, sids)
        np.testing.assert_equal(sdm._fids, fids)
        return fig

    @pytest.mark.mpl_image_compare
    def test_sdm_tsne_feature_gradient_plus10_plot(self):
        sids = list(range(8))
        fids = [str(i) for i in range(10)]
        np.random.seed(123)
        x = np.random.ranf(80).reshape(8, -1)
        x_sorted = x[np.argsort(x[:, 5])]
        sdm = eda.SampleDistanceMatrix(
            x_sorted, sids=sids, fids=fids, use_pdist=False)
        fig = sdm.tsne_feature_gradient_plot(
            '5', transform=lambda x: x + 10, figsize=(10, 10), s=50)
        np.testing.assert_equal(sdm._x, x_sorted)
        np.testing.assert_equal(sdm._sids, sids)
        np.testing.assert_equal(sdm._fids, fids)
        return fig

    @pytest.mark.mpl_image_compare
    def test_sdm_tsne_feature_gradient_plot_sslabs(self):
        sids = list(range(8))
        fids = [str(i) for i in range(10)]
        np.random.seed(123)
        x = np.random.ranf(80).reshape(8, -1)
        x_sorted = x[np.argsort(x[:, 5])]
        sdm = eda.SampleDistanceMatrix(
            x_sorted, sids=sids, fids=fids, use_pdist=False)
        sdm.tsne_feature_gradient_plot(
            '5', labels=list('abcdefgh'), selected_labels='a',
            transform=lambda x: np.log(x+1),
            figsize=(10, 10), s=50)
        fig = sdm.tsne_feature_gradient_plot(
            '5', labels=list('abcdefgh'), selected_labels='a',
            figsize=(10, 10), s=50)
        np.testing.assert_equal(sdm._x, x_sorted)
        np.testing.assert_equal(sdm._sids, sids)
        np.testing.assert_equal(sdm._fids, fids)
        return fig

    @pytest.mark.mpl_image_compare
    def test_sdm_tsne_feature_gradient_plot_sslabs_empty(self):
        sids = list(range(8))
        fids = [str(i) for i in range(10)]
        np.random.seed(123)
        x = np.random.ranf(80).reshape(8, -1)
        x_sorted = x[np.argsort(x[:, 5])]
        sdm = eda.SampleDistanceMatrix(
            x_sorted, sids=sids, fids=fids, use_pdist=False)
        fig = sdm.tsne_feature_gradient_plot(
            '5', labels=list('abcdefgh'), selected_labels=[],
            figsize=(10, 10), s=50)
        np.testing.assert_equal(sdm._x, x_sorted)
        np.testing.assert_equal(sdm._sids, sids)
        np.testing.assert_equal(sdm._fids, fids)
        return fig

    def test_sdm_tsne_feature_gradient_plot_sslabs_wrong_args(self):
        sids = list(range(8))
        fids = [str(i) for i in range(10)]
        np.random.seed(123)
        x = np.random.ranf(80).reshape(8, -1)
        x_sorted = x[np.argsort(x[:, 5])]
        sdm = eda.SampleDistanceMatrix(
            x_sorted, sids=sids, fids=fids, use_pdist=False)
        # Mismatch labels
        with pytest.raises(ValueError) as excinfo:
            sdm.tsne_feature_gradient_plot(
                '5', labels=list('abcdefgh'), selected_labels=[11],
                figsize=(10, 10), s=50)

        with pytest.raises(ValueError) as excinfo:
            sdm.tsne_feature_gradient_plot(
                '5', labels=list('abcdefgh'), selected_labels=['i'],
                figsize=(10, 10), s=50)
        # labels not provided
        with pytest.raises(ValueError) as excinfo:
            sdm.tsne_feature_gradient_plot(
                '5', selected_labels=[11], figsize=(10, 10), s=50)

    def test_sdm_tsne_feature_gradient_plot_wrong_args(self):
        sids = list(range(8))
        fids = [str(i) for i in range(10)]
        np.random.seed(123)
        x = np.random.ranf(80).reshape(8, -1)
        x_sorted = x[np.argsort(x[:, 5])]
        sdm = eda.SampleDistanceMatrix(
            x, sids=sids, fids=fids, use_pdist=False)
        with pytest.raises(ValueError):
            sdm.tsne_feature_gradient_plot('5', transform=2)

        # wrong labels size
        with pytest.raises(ValueError):
            sdm.tsne_feature_gradient_plot('5', figsize=(10, 10),
                                           s=50, labels=[])

        with pytest.raises(ValueError):
            sdm.tsne_feature_gradient_plot('5', figsize=(10, 10),
                                           s=50, labels=[1])

        with pytest.raises(ValueError):
            sdm.tsne_feature_gradient_plot('5', figsize=(10, 10),
                                           s=50, labels=[2])

        # wrong gradient length
        with pytest.raises(ValueError):
            sdm.tsne_feature_gradient_plot([0, 1])

        with pytest.raises(ValueError):
            sdm.tsne_feature_gradient_plot(11)

        with pytest.raises(ValueError):
            sdm.tsne_feature_gradient_plot(11)

        with pytest.raises(ValueError):
            sdm.tsne_feature_gradient_plot(-1)

        with pytest.raises(ValueError):
            sdm.tsne_feature_gradient_plot(5)

        with pytest.raises(ValueError):
            sdm.tsne_feature_gradient_plot('123')

    @pytest.mark.mpl_image_compare
    def test_sdm_tsne_plot(self):
        sids = list(range(8))
        fids = [str(i) for i in range(10)]
        np.random.seed(123)
        x = np.random.ranf(80).reshape(8, -1)
        x_sorted = x[np.argsort(x[:, 5])]
        g = x_sorted[:, 5]
        sdm = eda.SampleDistanceMatrix(
            x_sorted, sids=sids, fids=fids, use_pdist=False)
        return sdm.tsne_plot(g, figsize=(10, 10), s=50)

    @pytest.mark.mpl_image_compare
    def test_sdm_pca_feature_gradient_plot(self):
        sids = list(range(8))
        fids = [str(i) for i in range(10)]
        np.random.seed(123)
        x = np.random.ranf(80).reshape(8, -1)
        x_sorted = x[np.argsort(x[:, 5])]
        sdm = eda.SampleDistanceMatrix(
            x_sorted, sids=sids, fids=fids, use_pdist=False)
        fig = sdm.pca_feature_gradient_plot(
            '5', figsize=(10, 10), s=50)
        np.testing.assert_equal(sdm._x, x_sorted)
        np.testing.assert_equal(sdm._sids, sids)
        np.testing.assert_equal(sdm._fids, fids)
        return fig

    @pytest.mark.mpl_image_compare
    def test_sdm_pca_feature_gradient_plus10_plot(self):
        sids = list(range(8))
        fids = [str(i) for i in range(10)]
        np.random.seed(123)
        x = np.random.ranf(80).reshape(8, -1)
        x_sorted = x[np.argsort(x[:, 5])]
        sdm = eda.SampleDistanceMatrix(
            x_sorted, sids=sids, fids=fids, use_pdist=False)
        fig = sdm.pca_feature_gradient_plot(
            '5', transform=lambda x: x + 10, figsize=(10, 10), s=50)
        np.testing.assert_equal(sdm._x, x_sorted)
        np.testing.assert_equal(sdm._sids, sids)
        np.testing.assert_equal(sdm._fids, fids)
        return fig

    @pytest.mark.mpl_image_compare
    def test_sdm_pca_feature_gradient_plot_sslabs(self):
        sids = list(range(8))
        fids = [str(i) for i in range(10)]
        np.random.seed(123)
        x = np.random.ranf(80).reshape(8, -1)
        x_sorted = x[np.argsort(x[:, 5])]
        sdm = eda.SampleDistanceMatrix(
            x_sorted, sids=sids, fids=fids, use_pdist=False)
        sdm.pca_feature_gradient_plot(
            '5', labels=list('abcdefgh'), selected_labels='a',
            transform=lambda x: np.log(x+1),
            figsize=(10, 10), s=50)
        fig = sdm.pca_feature_gradient_plot(
            '5', labels=list('abcdefgh'), selected_labels='a',
            figsize=(10, 10), s=50)
        np.testing.assert_equal(sdm._x, x_sorted)
        np.testing.assert_equal(sdm._sids, sids)
        np.testing.assert_equal(sdm._fids, fids)
        return fig

    @pytest.mark.mpl_image_compare
    def test_sdm_pca_feature_gradient_plot_sslabs_empty(self):
        sids = list(range(8))
        fids = [str(i) for i in range(10)]
        np.random.seed(123)
        x = np.random.ranf(80).reshape(8, -1)
        x_sorted = x[np.argsort(x[:, 5])]
        sdm = eda.SampleDistanceMatrix(
            x_sorted, sids=sids, fids=fids, use_pdist=False)
        fig = sdm.pca_feature_gradient_plot(
            '5', labels=list('abcdefgh'), selected_labels=[],
            figsize=(10, 10), s=50)
        np.testing.assert_equal(sdm._x, x_sorted)
        np.testing.assert_equal(sdm._sids, sids)
        np.testing.assert_equal(sdm._fids, fids)
        return fig

    def test_sdm_pca_feature_gradient_plot_sslabs_wrong_args(self):
        sids = list(range(8))
        fids = [str(i) for i in range(10)]
        np.random.seed(123)
        x = np.random.ranf(80).reshape(8, -1)
        x_sorted = x[np.argsort(x[:, 5])]
        sdm = eda.SampleDistanceMatrix(
            x_sorted, sids=sids, fids=fids, use_pdist=False)
        # Mismatch labels
        with pytest.raises(ValueError) as excinfo:
            sdm.pca_feature_gradient_plot(
                '5', labels=list('abcdefgh'), selected_labels=[11],
                figsize=(10, 10), s=50)

        with pytest.raises(ValueError) as excinfo:
            sdm.pca_feature_gradient_plot(
                '5', labels=list('abcdefgh'), selected_labels=['i'],
                figsize=(10, 10), s=50)
        # labels not provided
        with pytest.raises(ValueError) as excinfo:
            sdm.pca_feature_gradient_plot(
                '5', selected_labels=[11], figsize=(10, 10), s=50)

    def test_sdm_pca_feature_gradient_plot_wrong_args(self):
        sids = list(range(8))
        fids = [str(i) for i in range(10)]
        np.random.seed(123)
        x = np.random.ranf(80).reshape(8, -1)
        x_sorted = x[np.argsort(x[:, 5])]
        sdm = eda.SampleDistanceMatrix(
            x, sids=sids, fids=fids, use_pdist=False)
        with pytest.raises(ValueError):
            sdm.pca_feature_gradient_plot('5', transform=2)

        # wrong labels size
        with pytest.raises(ValueError):
            sdm.pca_feature_gradient_plot('5', figsize=(10, 10),
                                          s=50, labels=[])

        with pytest.raises(ValueError):
            sdm.pca_feature_gradient_plot('5', figsize=(10, 10),
                                          s=50, labels=[1])

        with pytest.raises(ValueError):
            sdm.pca_feature_gradient_plot('5', figsize=(10, 10),
                                          s=50, labels=[2])

        # wrong gradient length
        with pytest.raises(ValueError):
            sdm.pca_feature_gradient_plot([0, 1])

        with pytest.raises(ValueError):
            sdm.pca_feature_gradient_plot(11)

        with pytest.raises(ValueError):
            sdm.pca_feature_gradient_plot(11)

        with pytest.raises(ValueError):
            sdm.pca_feature_gradient_plot(-1)

        with pytest.raises(ValueError):
            sdm.pca_feature_gradient_plot(5)

        with pytest.raises(ValueError):
            sdm.pca_feature_gradient_plot('123')

    @pytest.mark.mpl_image_compare
    def test_sdm_pca_plot(self):
        sids = list(range(8))
        fids = [str(i) for i in range(10)]
        np.random.seed(123)
        x = np.random.ranf(80).reshape(8, -1)
        x_sorted = x[np.argsort(x[:, 5])]
        g = x_sorted[:, 5]
        sdm = eda.SampleDistanceMatrix(
            x_sorted, sids=sids, fids=fids, use_pdist=False)
        return sdm.pca_plot(gradient=g, figsize=(10, 10), s=50)

    def test_pca_dim(self):
        np.random.seed(123)
        x5k = np.random.normal(size=5000)
        sdm = eda.SampleDistanceMatrix(
            x5k.reshape(20, -1), use_pdist=False)
        assert sdm._pca_x.shape == (20, 20)

    def test_pca_var_explained(self):
        np.random.seed(123)
        x5k = np.random.normal(size=5000)
        sdm = eda.SampleDistanceMatrix(x5k.reshape(20, -1), use_pdist=False)
        assert sdm._skd_pca.explained_variance_.shape == (20,)
        assert sdm._skd_pca.explained_variance_ratio_.shape == (20,)

    @pytest.mark.mpl_image_compare
    def test_sdm_nopdist_umap_feature_gradient_plot(self):
        sids = list(range(8))
        fids = [str(i) for i in range(10)]
        np.random.seed(123)
        x = np.random.ranf(80).reshape(8, -1)
        x_sorted = x[np.argsort(x[:, 5])]
        sdm = eda.SampleDistanceMatrix(
            x_sorted, sids=sids, fids=fids, use_pdist=False)
        fig = sdm.umap_feature_gradient_plot(
            '5', figsize=(10, 10), s=50)
        np.testing.assert_equal(sdm._x, x_sorted)
        np.testing.assert_equal(sdm._sids, sids)
        np.testing.assert_equal(sdm._fids, fids)
        return fig

    @pytest.mark.mpl_image_compare
    def test_sdm_nopdist_umap_feature_gradient_plus10_plot(self):
        sids = list(range(8))
        fids = [str(i) for i in range(10)]
        np.random.seed(123)
        x = np.random.ranf(80).reshape(8, -1)
        x_sorted = x[np.argsort(x[:, 5])]
        sdm = eda.SampleDistanceMatrix(
            x_sorted, sids=sids, fids=fids, use_pdist=False)
        fig = sdm.umap_feature_gradient_plot(
            '5', transform=lambda x: x + 10, figsize=(10, 10), s=50)
        np.testing.assert_equal(sdm._x, x_sorted)
        np.testing.assert_equal(sdm._sids, sids)
        np.testing.assert_equal(sdm._fids, fids)
        return fig

    @pytest.mark.mpl_image_compare
    def test_sdm_nopdist_umap_feature_gradient_plot_sslabs(self):
        sids = list(range(8))
        fids = [str(i) for i in range(10)]
        np.random.seed(123)
        x = np.random.ranf(80).reshape(8, -1)
        x_sorted = x[np.argsort(x[:, 5])]
        sdm = eda.SampleDistanceMatrix(
            x_sorted, sids=sids, fids=fids, use_pdist=False)
        sdm.umap_feature_gradient_plot(
            '5', labels=list('abcdefgh'), selected_labels='a',
            transform=lambda x: np.log(x+1),
            figsize=(10, 10), s=50)
        fig = sdm.umap_feature_gradient_plot(
            '5', labels=list('abcdefgh'), selected_labels='a',
            figsize=(10, 10), s=50)
        np.testing.assert_equal(sdm._x, x_sorted)
        np.testing.assert_equal(sdm._sids, sids)
        np.testing.assert_equal(sdm._fids, fids)
        return fig

    @pytest.mark.mpl_image_compare
    def test_sdm_nopdist_umap_feature_gradient_plot_sslabs_empty(self):
        sids = list(range(8))
        fids = [str(i) for i in range(10)]
        np.random.seed(123)
        x = np.random.ranf(80).reshape(8, -1)
        x_sorted = x[np.argsort(x[:, 5])]
        sdm = eda.SampleDistanceMatrix(
            x_sorted, sids=sids, fids=fids, use_pdist=False)
        fig = sdm.umap_feature_gradient_plot(
            '5', labels=list('abcdefgh'), selected_labels=[],
            figsize=(10, 10), s=50)
        np.testing.assert_equal(sdm._x, x_sorted)
        np.testing.assert_equal(sdm._sids, sids)
        np.testing.assert_equal(sdm._fids, fids)
        return fig

    def test_sdm_umap_feature_gradient_plot_sslabs_wrong_args(self):
        sids = list(range(8))
        fids = [str(i) for i in range(10)]
        np.random.seed(123)
        x = np.random.ranf(80).reshape(8, -1)
        x_sorted = x[np.argsort(x[:, 5])]
        sdm = eda.SampleDistanceMatrix(
            x_sorted, sids=sids, fids=fids, use_pdist=False)
        # Mismatch labels
        with pytest.raises(ValueError) as excinfo:
            sdm.umap_feature_gradient_plot(
                '5', labels=list('abcdefgh'), selected_labels=[11],
                figsize=(10, 10), s=50)

        with pytest.raises(ValueError) as excinfo:
            sdm.umap_feature_gradient_plot(
                '5', labels=list('abcdefgh'), selected_labels=['i'],
                figsize=(10, 10), s=50)
        # labels not provided
        with pytest.raises(ValueError) as excinfo:
            sdm.umap_feature_gradient_plot(
                '5', selected_labels=[11], figsize=(10, 10), s=50)

    def test_sdm_umap_feature_gradient_plot_wrong_args(self):
        sids = list(range(8))
        fids = [str(i) for i in range(10)]
        np.random.seed(123)
        x = np.random.ranf(80).reshape(8, -1)
        x_sorted = x[np.argsort(x[:, 5])]
        sdm = eda.SampleDistanceMatrix(
            x, sids=sids, fids=fids, use_pdist=False)
        with pytest.raises(ValueError):
            sdm.umap_feature_gradient_plot('5', transform=2)

        # wrong labels size
        with pytest.raises(ValueError):
            sdm.umap_feature_gradient_plot('5', figsize=(10, 10),
                                           s=50, labels=[])

        with pytest.raises(ValueError):
            sdm.umap_feature_gradient_plot('5', figsize=(10, 10),
                                           s=50, labels=[1])

        with pytest.raises(ValueError):
            sdm.umap_feature_gradient_plot('5', figsize=(10, 10),
                                           s=50, labels=[2])

        # wrong gradient length
        with pytest.raises(ValueError):
            sdm.umap_feature_gradient_plot([0, 1])

        with pytest.raises(ValueError):
            sdm.umap_feature_gradient_plot(11)

        with pytest.raises(ValueError):
            sdm.umap_feature_gradient_plot(11)

        with pytest.raises(ValueError):
            sdm.umap_feature_gradient_plot(-1)

        with pytest.raises(ValueError):
            sdm.umap_feature_gradient_plot(5)

        with pytest.raises(ValueError):
            sdm.umap_feature_gradient_plot('123')

    @pytest.mark.mpl_image_compare
    def test_sdm_nopdist_umap_plot(self):
        sids = list(range(8))
        fids = [str(i) for i in range(10)]
        np.random.seed(123)
        x = np.random.ranf(80).reshape(8, -1)
        x_sorted = x[np.argsort(x[:, 5])]
        g = x_sorted[:, 5]
        sdm = eda.SampleDistanceMatrix(
            x_sorted, sids=sids, fids=fids, use_pdist=False)
        return sdm.umap_plot(gradient=g, figsize=(10, 10), s=50)

    def test_umap_dim(self):
        np.random.seed(123)
        x5k = np.random.normal(size=5000)
        sdm = eda.SampleDistanceMatrix(x5k.reshape(20, -1), use_pdist=False)
        assert sdm._umap_x.shape == (20, 2)

    def test_s_knn_connectivity_matrix(self):
        nn_sdm = eda.SampleDistanceMatrix([[0], [1], [5]],
                                          metric='euclidean', use_pdist=False)
        np.testing.assert_allclose([[0, 1, 0],
                                    [1, 0, 0],
                                    [0, 4, 0]],
                                   nn_sdm.s_knn_connectivity_matrix(1))

    @pytest.mark.mpl_image_compare
    def test_s_knn_graph_grad_lab(self):
        np.random.seed(123)
        x = np.concatenate((np.random.normal(0, 1, 10),
                            np.random.normal(20, 1, 20))).reshape(30, -1)
        sdm = eda.SampleDistanceMatrix(x, metric='euclidean', use_pdist=False)
        sdm.s_knn_graph(5, figsize=(5, 5))
        assert (5, 1) in sdm._knn_ng_lut
        assert len(sdm._knn_ng_lut) == 1
        # use cache
        sdm.s_knn_graph(5, figsize=(5, 5))
        sdm.s_knn_graph(5, figsize=(5, 5), fa2_kwargs={})
        sdm.s_knn_graph(5, figsize=(5, 5), nx_draw_kwargs={})
        assert len(sdm._knn_ng_lut) == 1
        gradient = np.array([1] * 10 + [10] * 20)
        labs = gradient = np.array([1] * 10 + [2] * 20)
        return sdm.s_knn_graph(5, gradient=gradient, labels=labs,
                               figsize=(5, 5),
                               alpha=0.8, random_state=123)

    @pytest.mark.mpl_image_compare
    def test_s_knn_graph_grad_lab_same_marker(self):
        np.random.seed(123)
        x = np.concatenate((np.random.normal(0, 1, 10),
                            np.random.normal(20, 1, 20))).reshape(30, -1)
        sdm = eda.SampleDistanceMatrix(x, metric='euclidean', use_pdist=False)
        sdm.s_knn_graph(5, figsize=(5, 5))
        assert (5, 1) in sdm._knn_ng_lut
        assert len(sdm._knn_ng_lut) == 1
        gradient = np.array([1] * 10 + [10] * 20)
        labs = gradient = np.array([1] * 10 + [2] * 20)
        return sdm.s_knn_graph(5, gradient=gradient, labels=labs,
                               different_label_markers=False,
                               figsize=(5, 5),
                               alpha=0.8, random_state=123)

    @pytest.mark.mpl_image_compare
    def test_s_knn_graph_grad_nolab(self):
        np.random.seed(123)
        x = np.concatenate((np.random.normal(0, 1, 10),
                            np.random.normal(20, 1, 20))).reshape(30, -1)
        sdm = eda.SampleDistanceMatrix(x, metric='euclidean', use_pdist=False)
        sdm.s_knn_graph(5, figsize=(5, 5))
        assert (5, 1) in sdm._knn_ng_lut
        assert len(sdm._knn_ng_lut) == 1
        # use cache
        sdm.s_knn_graph(5, figsize=(5, 5))
        sdm.s_knn_graph(5, figsize=(5, 5), fa2_kwargs={})
        sdm.s_knn_graph(5, figsize=(5, 5), nx_draw_kwargs={})
        assert len(sdm._knn_ng_lut) == 1
        gradient = np.array([1] * 10 + [10] * 20)
        return sdm.s_knn_graph(5, gradient=gradient, figsize=(5, 5),
                               alpha=0.8, random_state=123)

    @pytest.mark.mpl_image_compare
    def test_s_knn_graph_nograd_nolab(self):
        np.random.seed(123)
        x = np.concatenate((np.random.normal(0, 1, 10),
                            np.random.normal(20, 1, 20))).reshape(30, -1)
        sdm = eda.SampleDistanceMatrix(x, metric='euclidean', use_pdist=False)
        sdm.s_knn_graph(5, figsize=(5, 5))
        assert (5, 1) in sdm._knn_ng_lut
        assert len(sdm._knn_ng_lut) == 1
        # use cache
        sdm.s_knn_graph(5, figsize=(5, 5))
        sdm.s_knn_graph(5, figsize=(5, 5), fa2_kwargs={})
        sdm.s_knn_graph(5, figsize=(5, 5), nx_draw_kwargs={})
        assert len(sdm._knn_ng_lut) == 1
        return sdm.s_knn_graph(5, figsize=(5, 5),
                               alpha=0.8, random_state=123)

    @pytest.mark.mpl_image_compare
    def test_s_knn_graph_nograd_lab(self):
        np.random.seed(123)
        x = np.concatenate((np.random.normal(0, 1, 10),
                            np.random.normal(20, 1, 20))).reshape(30, -1)
        sdm = eda.SampleDistanceMatrix(x, metric='euclidean', use_pdist=False)
        sdm.s_knn_graph(5, figsize=(5, 5))
        assert (5, 1) in sdm._knn_ng_lut
        assert len(sdm._knn_ng_lut) == 1
        # use cache
        sdm.s_knn_graph(5, figsize=(5, 5))
        sdm.s_knn_graph(5, figsize=(5, 5), fa2_kwargs={})
        sdm.s_knn_graph(5, figsize=(5, 5), nx_draw_kwargs={})
        assert len(sdm._knn_ng_lut) == 1
        labs = np.array([1] * 10 + [2] * 20)
        return sdm.s_knn_graph(5, labels=labs, figsize=(5, 5),
                               alpha=0.8, random_state=123)

    @pytest.mark.mpl_image_compare
    def test_s_knn_graph_nograd_lab_same_marker(self):
        np.random.seed(123)
        x = np.concatenate((np.random.normal(0, 1, 10),
                            np.random.normal(20, 1, 20))).reshape(30, -1)
        sdm = eda.SampleDistanceMatrix(x, metric='euclidean', use_pdist=False)
        sdm.s_knn_graph(5, figsize=(5, 5))
        assert (5, 1) in sdm._knn_ng_lut
        assert len(sdm._knn_ng_lut) == 1
        # use cache
        sdm.s_knn_graph(5, figsize=(5, 5))
        sdm.s_knn_graph(5, figsize=(5, 5), fa2_kwargs={})
        sdm.s_knn_graph(5, figsize=(5, 5), nx_draw_kwargs={})
        assert len(sdm._knn_ng_lut) == 1
        labs = np.array([1] * 10 + [2] * 20)
        return sdm.s_knn_graph(5, labels=labs, figsize=(5, 5),
                               different_label_markers=False,
                               alpha=0.8, random_state=123)
