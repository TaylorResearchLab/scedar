import numpy as np

import matplotlib as mpl
mpl.use("agg", warn=False)  # noqa
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn.metrics.pairwise

import scipy.cluster.hierarchy as sch

import scedar.eda as eda

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
        sdm3_corr_d = (1 - np.dot(
            self.x_2x4_arr[0] - self.x_2x4_arr[0].mean(),
            self.x_2x4_arr[1] - self.x_2x4_arr[1].mean()) /
                (np.linalg.norm(self.x_2x4_arr[0] - self.x_2x4_arr[0].mean(),
                                2) *
                 np.linalg.norm(self.x_2x4_arr[1] - self.x_2x4_arr[1].mean(),
                                2)))
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

    def test_sort_x_by_d(self):
        x1 = np.array([[0, 5, 30, 10],
                      [1, 5, 30, 10],
                      [0, 5, 33, 10],
                      [2, 5, 30, 7],
                      [2, 5, 30, 9]])
        x2 = x1.copy()
        opt_inds = eda.HClustTree.sort_x_by_d(
            x=x2.T, metric='euclidean')
        assert opt_inds == [2, 3, 1, 0]
        np.testing.assert_equal(x1, x2)

        x3 = np.array([[0, 0, 30, 10],
                      [1, 2, 30, 10],
                      [0, 3, 33, 10],
                      [2, 4, 30, 7],
                      [2, 5, 30, 9]])
        x4 = x3.copy()
        opt_inds = eda.HClustTree.sort_x_by_d(
            x=x4.T, metric='euclidean')
        assert opt_inds == [2, 3, 1, 0]
        np.testing.assert_equal(x3, x4)

    def test_sort_features(self):
        x = np.array([[0, 2, 30, 10],
                      [1, 2, 30, 10],
                      [0, 3, 33, 10],
                      [2, 5, 30, 7],
                      [2, 5, 30, 9]])
        sdm = eda.SampleDistanceMatrix(
            x, metric='euclidean')
        sdm2 = eda.SampleDistanceMatrix(
            x, metric='euclidean')
        sdm2.sort_features(fdist_metric='euclidean')
        assert sdm2.fids == [2, 3, 1, 0]

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
            sdm.get_tsne_kv([1, 2, 3])
        with pytest.raises(ValueError) as excinfo:
            sdm.get_tsne_kv({1: 2})

    def test_put_tsne_wrong_args(self):
        tmet = 'euclidean'
        sdm = eda.SampleDistanceMatrix(self.x_3x2, metric=tmet)
        with pytest.raises(ValueError) as excinfo:
            sdm.put_tsne(1, [1, 2, 3])
        with pytest.raises(ValueError) as excinfo:
            sdm.put_tsne({1: 2}, [1, 2, 3])

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
            x_sorted, sids=sids, fids=fids)
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
            x_sorted, sids=sids, fids=fids)
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
            x_sorted, sids=sids, fids=fids)
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
            x_sorted, sids=sids, fids=fids)
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
        sdm = eda.SampleDistanceMatrix(x, sids=sids, fids=fids)
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
    def test_sdm_tsne_gradient_plot(self):
        sids = list(range(8))
        fids = [str(i) for i in range(10)]
        np.random.seed(123)
        x = np.random.ranf(80).reshape(8, -1)
        x_sorted = x[np.argsort(x[:, 5])]
        g = x_sorted[:, 5]
        sdm = eda.SampleDistanceMatrix(x_sorted, sids=sids, fids=fids)
        return sdm.tsne_gradient_plot(g, figsize=(10, 10), s=50)

    @pytest.mark.mpl_image_compare
    def test_sdm_pca_feature_gradient_plot(self):
        sids = list(range(8))
        fids = [str(i) for i in range(10)]
        np.random.seed(123)
        x = np.random.ranf(80).reshape(8, -1)
        x_sorted = x[np.argsort(x[:, 5])]
        sdm = eda.SampleDistanceMatrix(
            x_sorted, sids=sids, fids=fids)
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
            x_sorted, sids=sids, fids=fids)
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
            x_sorted, sids=sids, fids=fids)
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
            x_sorted, sids=sids, fids=fids)
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
            x_sorted, sids=sids, fids=fids)
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
        sdm = eda.SampleDistanceMatrix(x, sids=sids, fids=fids)
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
    def test_sdm_pca_gradient_plot(self):
        sids = list(range(8))
        fids = [str(i) for i in range(10)]
        np.random.seed(123)
        x = np.random.ranf(80).reshape(8, -1)
        x_sorted = x[np.argsort(x[:, 5])]
        g = x_sorted[:, 5]
        sdm = eda.SampleDistanceMatrix(x_sorted, sids=sids, fids=fids)
        return sdm.pca_gradient_plot(gradient=g, figsize=(10, 10), s=50)

    def test_pca_dim(self):
        np.random.seed(123)
        x5k = np.random.normal(size=5000)
        sdm = eda.SampleDistanceMatrix(x5k.reshape(20, -1))
        assert sdm._pca_x.shape == (20, 20)
        assert sdm._pca_x.shape[1] == sdm._pca_n_components

    def test_pca_var_explained(self):
        np.random.seed(123)
        x5k = np.random.normal(size=5000)
        sdm = eda.SampleDistanceMatrix(x5k.reshape(20, -1))
        assert sdm._skd_pca.explained_variance_.shape == (20,)
        assert sdm._skd_pca.explained_variance_ratio_.shape == (20,)

    def test_s_knn_connectivity_matrix(self):
        nn_sdm = eda.SampleDistanceMatrix([[0], [1], [5]],
                                          metric='euclidean')
        np.testing.assert_allclose([[0, 1, 0],
                                    [1, 0, 0],
                                    [0, 4, 0]],
                                   nn_sdm.s_knn_connectivity_matrix(1))

    @pytest.mark.mpl_image_compare
    def test_draw_s_knn_graph(self):
        np.random.seed(123)
        x = np.concatenate((np.random.normal(0, 1, 10),
                            np.random.normal(20, 1, 20))).reshape(30, -1)
        sdm = eda.SampleDistanceMatrix(x, metric='euclidean')
        sdm.draw_s_knn_graph(5, figsize=(5, 5))
        assert (5, 1) in sdm._knn_ng_lut
        assert len(sdm._knn_ng_lut) == 1
        # use cache
        sdm.draw_s_knn_graph(5, figsize=(5, 5))
        sdm.draw_s_knn_graph(5, figsize=(5, 5), fa2_kwargs={})
        sdm.draw_s_knn_graph(5, figsize=(5, 5), nx_draw_kwargs={})
        assert len(sdm._knn_ng_lut) == 1
        gradient = np.array([1] * 10 + [10] * 20)
        return sdm.draw_s_knn_graph(5, gradient=gradient, figsize=(5, 5),
                                    alpha=0.8, random_state=123)

    def test_cosine_pdist(self):
        np.random.seed(222)
        x = np.random.ranf(10000).reshape(500, -1)
        skd = sklearn.metrics.pairwise.pairwise_distances(x, metric='cosine')
        np.testing.assert_allclose(
            eda.SampleDistanceMatrix.cosine_pdist(x), skd)

        np.testing.assert_allclose(
            eda.SampleDistanceMatrix(x, metric='cosine')._d, skd)

    def test_correlation_pdist(self):
        np.random.seed(222)
        x = np.random.ranf(10000).reshape(500, -1)
        skd = sklearn.metrics.pairwise.pairwise_distances(
            x, metric='correlation')
        np.testing.assert_allclose(
            eda.SampleDistanceMatrix.correlation_pdist(x), skd)

        np.testing.assert_allclose(
            eda.SampleDistanceMatrix(x, metric='correlation')._d, skd)


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
    hct = eda.HClustTree.hclust_tree(sdm_5x2.d, linkage="auto")

    def test_hclust_tree_args(self):
        eda.HClustTree.hclust_tree(self.sdm_5x2.d, linkage="auto",
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
        assert self.hct.left().left().left().leaf_ids() == []
        assert self.hct.left().left().left_leaf_ids() == []
        assert self.hct.left().left().right().count() == 0

    def test_hclust_tree_invalid_dmat(self):
        with pytest.raises(ValueError) as excinfo:
            eda.HClustTree.hclust_tree(np.arange(5))

        with pytest.raises(ValueError) as excinfo:
            eda.HClustTree.hclust_tree(np.arange(10).reshape(2, 5))

    def test_bi_partition_no_min(self):
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

    def test_bi_partition_2min_g_cnt(self):
        #   _______|_____
        #   |       ____|___
        # __|___    |    __|___
        # |    |    |    |    |
        # 0    2    4    1    3
        # Leaves are in optimal order.
        labs1, sids1 = self.hct.bi_partition(soft_min_subtree_size=3)

        # return subtrees True
        labs2, sids2, lst, rst = self.hct.bi_partition(
            soft_min_subtree_size=3, return_subtrees=True)

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

    def test_bi_partition_min_no_spl(self):
        # ____|____ 6
        # |    ___|____ 5
        # |    |    __|___ 4
        # |    |    |    |
        # 3    2    1    0
        z = sch.linkage([[0, 0], [1, 1], [3, 3], [6, 6]],
                        metric='euclidean', method='complete',
                        optimal_ordering=True)
        hct = eda.HClustTree(sch.to_tree(z))
        assert hct.leaf_ids() == [3, 2, 1, 0]
        labs, sids, lst, rst = hct.bi_partition(
            soft_min_subtree_size=2, return_subtrees=True)
        assert labs == [0, 0, 1, 1]
        assert sids == [3, 2, 1, 0]
        # hct should be changed accordingly
        assert hct.leaf_ids() == [3, 2, 1, 0]
        assert hct.left_leaf_ids() == [3, 2]
        assert hct.right_leaf_ids() == [1, 0]
        # subtrees
        assert lst.leaf_ids() == [3, 2]
        assert rst.leaf_ids() == [1, 0]
        # prev
        assert lst._prev is hct
        assert rst._prev is hct
        # ids
        assert lst._node.id == 5
        assert lst._node.left.id == 3
        assert lst._node.right.id == 2
        # ids
        assert rst._node.id == 4
        assert rst._node.left.id == 1
        assert rst._node.right.id == 0

    def test_bi_partition_min_no_spl_lr_rev(self):
        # left right reversed
        # ____|____ 6
        # |    ___|____ 5
        # |    |    __|___ 4
        # |    |    |    |
        # 3    2    1    0
        z = sch.linkage([[0, 0], [1, 1], [3, 3], [6, 6]],
                        metric='euclidean', method='complete',
                        optimal_ordering=True)
        root = sch.to_tree(z)
        # reverse left right subtree
        root_left = root.left
        root.left = root.right
        root.right = root_left
        hct = eda.HClustTree(root)
        assert hct.leaf_ids() == [2, 1, 0, 3]
        labs, sids, lst, rst = hct.bi_partition(
            soft_min_subtree_size=2, return_subtrees=True)
        assert labs == [0, 0, 1, 1]
        assert sids == [2, 1, 0, 3]
        # hct should be changed accordingly
        assert hct.leaf_ids() == [2, 1, 0, 3]
        assert hct.left_leaf_ids() == [2, 1]
        assert hct.right_leaf_ids() == [0, 3]
        # subtrees
        assert lst.leaf_ids() == [2, 1]
        assert rst.leaf_ids() == [0, 3]
        # prev
        assert lst._prev is hct
        assert rst._prev is hct
        assert hct._left is lst._node
        assert hct._right is rst._node
        # ids
        assert rst._node.id == 4
        assert rst._node.left.id == 0
        assert rst._node.right.id == 3
        # ids
        assert lst._node.id == 5
        assert lst._node.left.id == 2
        assert lst._node.right.id == 1

    def test_bi_partition_min_spl(self):
        # _____|_____
        # |     ____|____
        # |   __|__   __|__
        # |   |   |   |   |
        # 4   3   2   1   0
        z = sch.linkage([[0, 0], [1, 1], [3, 3], [4, 4], [10, 10]],
                        metric='euclidean', method='complete',
                        optimal_ordering=True)
        hct = eda.HClustTree(sch.to_tree(z))
        assert hct.leaf_ids() == [4, 3, 2, 1, 0]
        assert hct.left_leaf_ids() == [4]
        assert hct.right().left().leaf_ids() == [3, 2]
        assert hct.right().right().leaf_ids() == [1, 0]
        labs, sids, lst, rst = hct.bi_partition(
            soft_min_subtree_size=2, return_subtrees=True)
        assert labs == [0, 0, 0, 1, 1]
        assert sids == [4, 3, 2, 1, 0]
        # hct should be changed accordingly
        assert hct.leaf_ids() == [4, 3, 2, 1, 0]
        assert hct.left_leaf_ids() == [4, 3, 2]
        assert hct.right_leaf_ids() == [1, 0]
        # left
        assert lst._prev is hct
        assert lst._node.left.left.id == 4
        assert lst._node.left.right.id == 3
        assert lst._node.right.id == 2
        # right
        assert rst._prev is hct
        assert rst._node.left.id == 1
        assert rst._node.right.id == 0

    def test_bi_partition_min_multi_spl(self):
        # ____|____
        # |   ____|___
        # |   |   ___|____
        # |   |   |   ___|___
        # |   |   |   |   __|__
        # |   |   |   |   |   |
        # 5   4   3   2   1   0
        z = sch.linkage([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]],
                        metric='euclidean', method='single',
                        optimal_ordering=True)
        root = sch.to_tree(z)
        assert root.left.id == 5
        assert root.right.left.id == 4
        assert root.right.right.left.id == 3
        assert root.right.right.right.left.id == 2
        assert root.right.right.right.right.left.id == 1
        assert root.right.right.right.right.right.id == 0
        hct = eda.HClustTree(root)
        labs, sids, lst, rst = hct.bi_partition(
            soft_min_subtree_size=3, return_subtrees=True)
        assert labs == [0, 0, 0, 1, 1, 1]
        assert sids == [5, 4, 3, 2, 1, 0]
        # lst
        assert hct._left is lst._node
        assert lst._prev is hct
        assert lst.left_leaf_ids() == [5, 4]
        assert lst.right_leaf_ids() == [3]
        # rst
        assert hct._right is rst._node
        assert rst._prev is hct
        assert rst.left_leaf_ids() == [2]
        assert rst.right_leaf_ids() == [1, 0]

    def test_bi_partition_min_switch_spl(self):
        # _______|________
        # |         _____|_____
        # |     ____|____     |
        # |   __|__   __|__   |
        # |   |   |   |   |   |
        # 0   1   2   3   4   5
        # round 1: ( ((0, (1, 2)), (3, 4)), (5) )
        # round 2: ( (0, (1, 2), (3, (4, 5)) )
        z = sch.linkage([[0], [5], [6], [8], [9], [12]],
                        method='single', optimal_ordering=True)
        root = sch.to_tree(z)
        assert root.left.id == 0
        assert root.right.right.id == 5
        assert root.right.left.left.left.id == 1
        assert root.right.left.left.right.id == 2
        assert root.right.left.right.left.id == 3
        assert root.right.left.right.right.id == 4
        hct = eda.HClustTree(root)
        labs, sids, lst, rst = hct.bi_partition(
            soft_min_subtree_size=3, return_subtrees=True)
        assert labs == [0, 0, 0, 1, 1, 1]
        assert sids == [0, 1, 2, 3, 4, 5]
        # lst
        assert hct._left is lst._node
        assert lst._prev is hct
        assert lst.left_leaf_ids() == [0]
        assert lst.right_leaf_ids() == [1, 2]
        # rst
        assert hct._right is rst._node
        assert rst._prev is hct
        assert rst.left_leaf_ids() == [3]
        assert rst.right_leaf_ids() == [4, 5]

    def test_bi_partition_wrong_args(self):
        with pytest.raises(ValueError) as excinfo:
            self.hct.bi_partition(soft_min_subtree_size=0)

        with pytest.raises(ValueError) as excinfo:
            self.hct.bi_partition(soft_min_subtree_size=0.5)

        with pytest.raises(ValueError) as excinfo:
            self.hct.bi_partition(soft_min_subtree_size=-1)

    def test_cluster_id_to_lab_list_wrong_id_list_type(self):
        with pytest.raises(ValueError) as excinfo:
            eda.HClustTree.cluster_id_to_lab_list(
                np.array([[0, 1, 2], [3, 4]]), [0, 1, 2, 3, 4])

    def test_cluster_id_to_lab_list_mismatched_ids_sids(self):
        with pytest.raises(ValueError) as excinfo:
            eda.HClustTree.cluster_id_to_lab_list(
                [[0, 1, 2], [3, 4]], [0, 1, 2, 3, 5])

    def test_cluster_id_to_lab_list_empty_cluster(self):
        with pytest.raises(ValueError) as excinfo:
            eda.HClustTree.cluster_id_to_lab_list(
                [[], [0, 1, 2, 3, 4]], [0, 1, 2, 3, 4])
