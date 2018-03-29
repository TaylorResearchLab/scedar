import numpy as np
import seaborn as sns
import scedar.eda as eda
import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest


class TestSampleDistanceMatrix(object):
    """docstring for TestSampleDistanceMatrix"""
    x_3x2 = [[0, 0], [1, 1], [2, 2]]
    x_2x4_arr = np.array([[0, 1, 2, 3], [1, 2, 0, 6]])

    def test_valid_init(self):
        sdm = eda.SampleDistanceMatrix(self.x_3x2, metric='euclidean')
        dist_mat = np.array([[0, np.sqrt(2), np.sqrt(8)],
                             [np.sqrt(2), 0, np.sqrt(2)],
                             [np.sqrt(8), np.sqrt(2), 0]])
        np.testing.assert_allclose(sdm.d, dist_mat)

        sdm2 = eda.SampleDistanceMatrix(
            self.x_2x4_arr, metric='euclidean', nprocs=5)
        sdm2_d1 = np.sqrt(
            np.power(self.x_2x4_arr[0] - self.x_2x4_arr[1], 2).sum())
        np.testing.assert_allclose(sdm2.d,
                                   np.array([[0, sdm2_d1], [sdm2_d1, 0]]))

        sdm3 = eda.SampleDistanceMatrix(
            self.x_2x4_arr, metric='correlation', nprocs=5)
        sdm3_corr_d = (1 - np.dot(self.x_2x4_arr[0] - self.x_2x4_arr[0].mean(),
                                  self.x_2x4_arr[1] - self.x_2x4_arr[1].mean())
                       / (np.linalg.norm(self.x_2x4_arr[0] - self.x_2x4_arr[0].mean(), 2)
                          * np.linalg.norm(self.x_2x4_arr[1] - self.x_2x4_arr[1].mean(), 2)))
        np.testing.assert_allclose(sdm3.d,
                                   np.array([[0, 0.3618551],
                                             [0.3618551, 0]]))

        np.testing.assert_allclose(sdm3.d,
                                   np.array([[0, sdm3_corr_d],
                                             [sdm3_corr_d, 0]]))

        sdm4 = eda.SampleDistanceMatrix(self.x_3x2, dist_mat)
        sdm5 = eda.SampleDistanceMatrix(
            self.x_3x2, dist_mat, metric='euclidean')
        sdm5 = eda.SampleDistanceMatrix([[1, 2]], metric='euclidean')
        assert sdm5.tsne(n_iter=250).shape == (1, 2)

    def test_empty_init(self):
        with pytest.raises(ValueError) as excinfo:
            eda.SampleDistanceMatrix(np.empty(0), metric='euclidean')
        sdm = eda.SampleDistanceMatrix(np.empty((0, 0)), metric='euclidean')
        assert len(sdm.sids) == 0
        assert len(sdm.fids) == 0
        assert sdm._x.shape == (0, 0)
        assert sdm._d.shape == (0, 0)
        assert sdm._col_sorted_d.shape == (0, 0)
        assert sdm._col_argsorted_d.shape == (0, 0)
        assert sdm.tsne(n_iter=250).shape == (0, 0)

    def test_init_wrong_metric(self):
        # when d is None, metric cannot be precomputed
        with pytest.raises(Exception) as excinfo:
            eda.SampleDistanceMatrix(self.x_3x2, metric='precomputed')

        # lazy load d
        eda.SampleDistanceMatrix(self.x_3x2, metric='unknown')
        with pytest.raises(Exception) as excinfo:
            eda.SampleDistanceMatrix(self.x_3x2, metric='unknown').d

        eda.SampleDistanceMatrix(self.x_3x2, metric=1)
        with pytest.raises(Exception) as excinfo:
            eda.SampleDistanceMatrix(self.x_3x2, metric=1).d

        eda.SampleDistanceMatrix(self.x_3x2, metric=1.)
        with pytest.raises(Exception) as excinfo:
            eda.SampleDistanceMatrix(self.x_3x2, metric=1.).d

        eda.SampleDistanceMatrix(self.x_3x2, metric=('euclidean', ))
        with pytest.raises(Exception) as excinfo:
            eda.SampleDistanceMatrix(self.x_3x2, metric=('euclidean', )).d

        eda.SampleDistanceMatrix(self.x_3x2, metric=['euclidean'])
        with pytest.raises(Exception) as excinfo:
            eda.SampleDistanceMatrix(self.x_3x2, metric=['euclidean']).d

    def test_init_wrong_d_type(self):
        d_3x3 = np.array([[0, np.sqrt(2), np.sqrt(8)],
                          ['1a1', 0, np.sqrt(2)],
                          [np.sqrt(8), np.sqrt(2), 0]])

        with pytest.raises(Exception) as excinfo:
            eda.SampleDistanceMatrix(self.x_3x2, d_3x3)

    def test_init_wrong_d_size(self):
        d_2x2 = np.array([[0, np.sqrt(2)],
                          [np.sqrt(2), 0]])

        d_2x2 = np.array([[0, np.sqrt(2)],
                          [np.sqrt(2), 0]])

        d_1x6 = np.arange(6)

        d_3x2 = np.array([[0, np.sqrt(2)],
                          [np.sqrt(2), 0],
                          [1, 2]])

        with pytest.raises(Exception) as excinfo:
            eda.SampleDistanceMatrix(self.x_3x2, d_2x2)

        with pytest.raises(Exception) as excinfo:
            eda.SampleDistanceMatrix(self.x_3x2, d_2x3)

        with pytest.raises(Exception) as excinfo:
            eda.SampleDistanceMatrix(self.x_3x2, d_3x2)

        with pytest.raises(Exception) as excinfo:
            eda.SampleDistanceMatrix(self.x_3x2, d_1x6)

    def test_get_tsne_kv(self):
        tmet = 'euclidean'
        sdm = eda.SampleDistanceMatrix(self.x_3x2, metric=tmet)
        assert sdm.get_tsne_kv(1) is None
        assert sdm.get_tsne_kv(1) is None
        assert sdm.get_tsne_kv(0) is None
        assert sdm.get_tsne_kv(2) is None

    def test_get_tsne_kv_wrong_args(self):
        tmet = 'euclidean'
        sdm = eda.SampleDistanceMatrix(self.x_3x2, metric=tmet)
        with pytest.raises(ValueError) as excinfo:
            sdm.get_tsne_kv([1,2,3])
        with pytest.raises(ValueError) as excinfo:
            sdm.get_tsne_kv({1:2})

    def test_put_tsne_wrong_args(self):
        tmet = 'euclidean'
        sdm = eda.SampleDistanceMatrix(self.x_3x2, metric=tmet)
        with pytest.raises(ValueError) as excinfo:
            sdm.put_tsne(1, [1,2,3])
        with pytest.raises(ValueError) as excinfo:
            sdm.put_tsne({1:2}, [1,2,3])

    def test_tsne(self):
        tmet = 'euclidean'
        tsne_kwargs = {'metric': tmet, 'n_iter': 250,
                       'random_state': 123}
        ref_tsne = eda.tsne(self.x_3x2, **tsne_kwargs)
        sdm = eda.SampleDistanceMatrix(self.x_3x2, metric=tmet)

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
        ref_tsne = eda.tsne(self.x_3x2, **param_list[0])
        sdm = eda.SampleDistanceMatrix(self.x_3x2, metric=tmet)
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
        ref_tsne = eda.tsne(self.x_3x2, **param_list[0])
        sdm = eda.SampleDistanceMatrix(self.x_3x2, metric=tmet)
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
        ref_tsne = eda.tsne(self.x_3x2, **tsne_kwargs)
        sdm = eda.SampleDistanceMatrix(self.x_3x2, metric=tmet)
        
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
            np.random.ranf(60).reshape(6, -1), sids=sids, fids=fids)
        # select sf
        ss_sdm = sdm.ind_x([0, 5], list(range(9)))
        assert ss_sdm._x.shape == (2, 9)
        assert ss_sdm.sids == ['a', 'f']
        assert ss_sdm.fids == list(range(10, 19))
        np.testing.assert_equal(
            ss_sdm.d, sdm._d[np.ix_((0, 5), (0, 5))])

        # select with Default
        ss_sdm = sdm.ind_x()
        assert ss_sdm._x.shape == (6, 10)
        assert ss_sdm.sids == list("abcdef")
        assert ss_sdm.fids == list(range(10, 20))
        np.testing.assert_equal(ss_sdm.d, sdm._d)
        
        # select with None
        ss_sdm = sdm.ind_x(None, None)
        assert ss_sdm._x.shape == (6, 10)
        assert ss_sdm.sids == list("abcdef")
        assert ss_sdm.fids == list(range(10, 20))
        np.testing.assert_equal(ss_sdm.d, sdm._d)
                        
        # select non-existent inds
        with pytest.raises(IndexError) as excinfo:
            sdm.ind_x([6])

        with pytest.raises(IndexError) as excinfo:
            sdm.ind_x(None, ['a'])

    def test_ind_x_empty(self):
        sids = list("abcdef")
        fids = list(range(10, 20))
        sdm = eda.SampleDistanceMatrix(
            np.random.ranf(60).reshape(6, -1), sids=sids, fids=fids)
        empty_s = sdm.ind_x([])
        assert empty_s._x.shape == (0, 10)
        assert empty_s._d.shape == (0, 0)
        assert empty_s._sids.shape == (0,)
        assert empty_s._fids.shape == (10,)

        empty_f = sdm.ind_x(None, [])
        assert empty_f._x.shape == (6, 0)
        assert empty_f._d.shape == (6, 6)
        assert empty_f._sids.shape == (6,)
        assert empty_f._fids.shape == (0,)

        empty_sf = sdm.ind_x([], [])
        assert empty_sf._x.shape == (0, 0)
        assert empty_sf._d.shape == (0, 0)
        assert empty_sf._sids.shape == (0,)
        assert empty_sf._fids.shape == (0,)

    def test_id_x(self):
        sids = list("abcdef")
        fids = list(range(10, 20))
        sdm = eda.SampleDistanceMatrix(
            np.random.ranf(60).reshape(6, -1), sids=sids, fids=fids)
        # select sf
        ss_sdm = sdm.id_x(['a', 'f'], list(range(10, 15)))
        assert ss_sdm._x.shape == (2, 5)
        assert ss_sdm.sids == ['a', 'f']
        assert ss_sdm.fids == list(range(10, 15))
        np.testing.assert_equal(
            ss_sdm.d, sdm._d[np.ix_((0, 5), (0, 5))])

        # select with Default
        ss_sdm = sdm.id_x()
        assert ss_sdm._x.shape == (6, 10)
        assert ss_sdm.sids == list("abcdef")
        assert ss_sdm.fids == list(range(10, 20))
        np.testing.assert_equal(ss_sdm.d, sdm._d)
        
        # select with None
        ss_sdm = sdm.id_x(None, None)
        assert ss_sdm._x.shape == (6, 10)
        assert ss_sdm.sids == list("abcdef")
        assert ss_sdm.fids == list(range(10, 20))
        np.testing.assert_equal(ss_sdm.d, sdm._d)
                        
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
            np.random.ranf(60).reshape(6, -1), sids=sids, fids=fids)
        empty_s = sdm.id_x([])
        assert empty_s._x.shape == (0, 10)
        assert empty_s._d.shape == (0, 0)
        assert empty_s._sids.shape == (0,)
        assert empty_s._fids.shape == (10,)

        empty_f = sdm.id_x(None, [])
        assert empty_f._x.shape == (6, 0)
        assert empty_f._d.shape == (6, 6)
        assert empty_f._sids.shape == (6,)
        assert empty_f._fids.shape == (0,)

        empty_sf = sdm.id_x([], [])
        assert empty_sf._x.shape == (0, 0)
        assert empty_sf._d.shape == (0, 0)
        assert empty_sf._sids.shape == (0,)
        assert empty_sf._fids.shape == (0,)

    def test_getter(self):
        tmet = 'euclidean'
        sdm = eda.SampleDistanceMatrix(self.x_3x2, metric=tmet)
        dist_mat = np.array([[0, np.sqrt(2), np.sqrt(8)],
                             [np.sqrt(2), 0, np.sqrt(2)],
                             [np.sqrt(8), np.sqrt(2), 0]])
        np.testing.assert_allclose(sdm.d, dist_mat)
        assert sdm.d is not sdm._d
        assert sdm.metric == tmet
        assert sdm.tsne_lut == {}
        assert sdm.tsne_lut is not sdm._tsne_lut
        assert sdm.tsne_lut == sdm._tsne_lut
        sdm.tsne(n_iter=250)
        assert sdm.tsne_lut is not sdm._tsne_lut
        for k in sdm.tsne_lut:
            np.testing.assert_equal(sdm.tsne_lut[k], sdm._tsne_lut[k])

    def test_num_correct_dist_mat(self):
        tdmat = np.array([[0, 1, 2],
                          [0.5, 0, 1.5],
                          [1, 1.6, 0.5]])
        # upper triangle is assgned with lower triangle values
        ref_cdmat = np.array([[0, 0.5, 1],
                              [0.5, 0, 1.6],
                              [1, 1.6, 0]])
        with pytest.warns(UserWarning):
            cdmat = eda.SampleDistanceMatrix.num_correct_dist_mat(tdmat)

        np.testing.assert_equal(cdmat, ref_cdmat)

        ref_cdmat2 = np.array([[0, 0.5, 1],
                               [0.5, 0, 1],
                               [1, 1, 0]])
        # with upper bound
        cdmat2 = eda.SampleDistanceMatrix.num_correct_dist_mat(tdmat, 1)
        np.testing.assert_equal(cdmat2, ref_cdmat2)

        # wrong shape
        tdmat3 = np.array([[0, 0.5],
                           [0.5, 0],
                           [1, 1]])
        # with upper bound
        with pytest.raises(Exception) as excinfo:
            eda.SampleDistanceMatrix.num_correct_dist_mat(tdmat3, 1)

        with pytest.raises(Exception) as excinfo:
            eda.SampleDistanceMatrix.num_correct_dist_mat(tdmat3)

    def test_s_ith_nn_d(self):
        nn_sdm = eda.SampleDistanceMatrix([[0], [1], [5], [6], [10], [20]],
                                          metric='euclidean')
        np.testing.assert_allclose([0, 0, 0, 0, 0, 0],
                                   nn_sdm.s_ith_nn_d(0))
        np.testing.assert_allclose([1, 1, 1, 1, 4, 10],
                                   nn_sdm.s_ith_nn_d(1))
        np.testing.assert_allclose([5, 4, 4, 4, 5, 14],
                                   nn_sdm.s_ith_nn_d(2))

    def test_s_ith_nn_ind(self):
        nn_sdm = eda.SampleDistanceMatrix([[0, 0, 0], [1, 1, 1], [5, 5, 5], 
                                           [6, 6, 6], [10, 10, 10], 
                                           [20, 20, 20]], 
                                          metric='euclidean')
        np.testing.assert_allclose([0, 1, 2, 3, 4, 5],
                                   nn_sdm.s_ith_nn_ind(0))
        np.testing.assert_allclose([1, 0, 3, 2, 3, 4],
                                   nn_sdm.s_ith_nn_ind(1))
        np.testing.assert_allclose([2, 2, 1, 4, 2, 3],
                                   nn_sdm.s_ith_nn_ind(2))

    # Because summary dist plot calls hist_dens_plot immediately after 
    # obtaining the summary statistics vector, the correctness of summary
    # statistics vector and hist_dens_plot implies the correctness of the
    # plots.
    @pytest.mark.filterwarnings("ignore:The 'normed' kwarg is depreca")
    def test_s_ith_nn_d_dist(self):
        nn_sdm = eda.SampleDistanceMatrix([[0, 0, 0], [1, 1, 1], [5, 5, 5], 
                                           [6, 6, 6], [10, 10, 10], 
                                           [20, 20, 20]], 
                                          metric='euclidean')
        nn_sdm.s_ith_nn_d_dist(1)

    def test_knn_ind_lut(self):
        nn_sdm = eda.SampleDistanceMatrix([[0, 0, 0], [1, 1, 1], [5, 5, 5], 
                                           [6, 6, 6], [10, 10, 10], 
                                           [20, 20, 20]], 
                                          metric='euclidean')
        assert nn_sdm.s_knn_ind_lut(0) == dict(zip(range(6), [[]]*6))
        assert (nn_sdm.s_knn_ind_lut(1) == 
                dict(zip(range(6), [[1], [0], [3], [2], [3], [4]])))
        assert (nn_sdm.s_knn_ind_lut(2) == 
                dict(zip(range(6), [[1, 2], [0, 2], [3, 1], 
                                    [2, 4], [3, 2], [4, 3]])))
        assert (nn_sdm.s_knn_ind_lut(3) == 
                dict(zip(range(6), [[1, 2, 3], [0, 2, 3], [3, 1, 0], 
                                    [2, 4, 1], [3, 2, 1], [4, 3, 2]])))
        nn_sdm.s_knn_ind_lut(5)

    def test_knn_ind_lut_wrong_args(self):
        nn_sdm = eda.SampleDistanceMatrix([[0, 0, 0], [1, 1, 1], [5, 5, 5], 
                                           [6, 6, 6], [10, 10, 10], 
                                           [20, 20, 20]], 
                                          metric='euclidean')
        with pytest.raises(ValueError) as excinfo:
            nn_sdm.s_knn_ind_lut(-1)

        with pytest.raises(ValueError) as excinfo:
            nn_sdm.s_knn_ind_lut(-0.5)

        with pytest.raises(ValueError) as excinfo:
            nn_sdm.s_knn_ind_lut(6)

        with pytest.raises(ValueError) as excinfo:
            nn_sdm.s_knn_ind_lut(6.5)

        with pytest.raises(ValueError) as excinfo:
            nn_sdm.s_knn_ind_lut(7)

        with pytest.raises(ValueError) as excinfo:
            nn_sdm.s_knn_ind_lut(7)

    @pytest.mark.mpl_image_compare
    def test_sdm_tsne_feature_gradient_plot(self):
        sids = list(range(8))
        fids = [str(i) for i in range(10)]
        np.random.seed(123)
        x = np.random.ranf(80).reshape(8, -1)
        x_sorted = x[np.argsort(x[:, 5])]
        sdm = eda.SampleDistanceMatrix(
            x_sorted, sids=sids, fids=fids)
        return sdm.tsne_feature_gradient_plot(
            '5', figsize=(10, 10), s=50)

    @pytest.mark.mpl_image_compare
    def test_sdm_tsne_gradient_plot(self):
        sids = list(range(8))
        fids = [str(i) for i in range(10)]
        np.random.seed(123)
        x = np.random.ranf(80).reshape(8, -1)
        x_sorted = x[np.argsort(x[:, 5])]
        g = x_sorted[:, 5]
        sdm = eda.SampleDistanceMatrix(x_sorted, sids=sids, fids=fids)
        return sdm.tsne_gradient_plot(g, figsize=(10, 10), s=50)

    def test_sdm_tsne_feature_gradient_plot_wrong_args(self):
        sids = list(range(8))
        fids = [str(i) for i in range(10)]
        np.random.seed(123)
        x = np.random.ranf(80).reshape(8, -1)
        x_sorted = x[np.argsort(x[:, 5])]
        sdm = eda.SampleDistanceMatrix(x, sids=sids, fids=fids)
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

