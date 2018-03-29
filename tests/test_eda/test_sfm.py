import numpy as np
import seaborn as sns
import scedar.eda as eda
import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest


class TestSampleFeatureMatrix(object):
    """docstring for TestSampleFeatureMatrix"""
    sfm5x10_arr = np.random.ranf(50).reshape(5, 10)
    sfm3x3_arr = np.random.ranf(9).reshape(3, 3)
    sfm5x10_lst = list(map(list, np.random.ranf(50).reshape(5, 10)))
    plt_arr = np.arange(60).reshape(6, 10)
    plt_sdm = eda.SampleFeatureMatrix(plt_arr, 
                                      sids=list("abcdef"), 
                                      fids=list(map(lambda i: 'f{}'.format(i), 
                                                    range(10))))
    # array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
    #        [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    #        [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
    #        [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
    #        [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
    #        [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]])

    ref_plt_f_sum = np.arange(0, 501, 100) + np.arange(10).sum()
    ref_plt_s_sum = np.arange(0, 55, 6) + np.arange(0, 51, 10).sum()
    ref_plt_f_mean = ref_plt_f_sum / 10
    ref_plt_s_mean = ref_plt_s_sum / 6
    ref_plt_f_cv = np.arange(10).std(ddof=1) / ref_plt_f_mean
    ref_plt_s_cv = np.arange(0, 51, 10).std(ddof=1) / ref_plt_s_mean
    ref_plt_f_gc = np.apply_along_axis(eda.stats.gc1d, 1, plt_arr)
    ref_plt_s_gc = np.apply_along_axis(eda.stats.gc1d, 0, plt_arr)
    ref_plt_f_a15 = np.array([0, 5, 10, 10, 10, 10])
    ref_plt_s_a35 = np.array([2, 2, 2, 2, 2, 3, 3, 3, 3, 3])

    def test_init_x_none(self):
        with pytest.raises(Exception) as excinfo:
            eda.SampleFeatureMatrix(None)

    def test_init_x_bad_type(self):
        with pytest.raises(Exception) as excinfo:
            eda.SampleFeatureMatrix([[0, 1], ['a', 2]])

    def test_init_x_1d(self):
        with pytest.raises(Exception) as excinfo:
            eda.SampleFeatureMatrix([1, 2, 3])

    def test_init_dup_sfids(self):
        with pytest.raises(Exception) as excinfo:
            eda.SampleFeatureMatrix(self.sfm5x10_lst, [0, 0, 1, 2, 3])

        with pytest.raises(Exception) as excinfo:
            eda.SampleFeatureMatrix(
                self.sfm5x10_lst, ['0', '0', '1', '2', '3'])

        with pytest.raises(Exception) as excinfo:
            eda.SampleFeatureMatrix(self.sfm5x10_lst, None, [0, 0, 1, 2, 3])

        with pytest.raises(Exception) as excinfo:
            eda.SampleFeatureMatrix(self.sfm5x10_lst, None, 
                                    ['0', '0', '1', '2', '3'])

    def test_init_empty_x_sfids(self):
        sfm1 = eda.SampleFeatureMatrix(np.array([[], []]), None, [])
        assert sfm1._x.shape == (2, 0)
        assert sfm1._sids.shape == (2,)
        assert sfm1._fids.shape == (0,)
        np.testing.assert_equal(sfm1.s_sum(), [])
        np.testing.assert_equal(sfm1.f_sum(), [0, 0])
        np.testing.assert_equal(sfm1.s_cv(), [])
        with pytest.warns(RuntimeWarning):
            np.testing.assert_equal(np.isnan(sfm1.f_cv()), [True, True])

        sfm2 = eda.SampleFeatureMatrix(np.empty((0, 0)))
        assert sfm2._x.shape == (0, 0)
        assert sfm2._sids.shape == (0,)
        assert sfm2._fids.shape == (0,)
        np.testing.assert_equal(sfm2.s_sum(), [])
        np.testing.assert_equal(sfm2.f_sum(), [])
        with pytest.warns(RuntimeWarning):
            np.testing.assert_equal(sfm2.s_cv(), [])
        with pytest.warns(RuntimeWarning):
            np.testing.assert_equal(sfm2.f_cv(), [])

    def test_init_wrong_sid_len(self):
        # wrong sid size
        with pytest.raises(Exception) as excinfo:
            eda.SampleFeatureMatrix(
                self.sfm5x10_lst, list(range(10)), list(range(5)))

        with pytest.raises(Exception) as excinfo:
            eda.SampleFeatureMatrix(self.sfm5x10_lst, list(range(10)))

    def test_init_wrong_fid_len(self):
        # wrong fid size
        with pytest.raises(Exception) as excinfo:
            eda.SampleFeatureMatrix(
                self.sfm5x10_lst, list(range(5)), list(range(2)))

    def test_init_wrong_sfid_len(self):
        # wrong sid and fid sizes
        with pytest.raises(Exception) as excinfo:
            eda.SampleFeatureMatrix(
                self.sfm5x10_lst, list(range(10)), list(range(10)))

    def test_init_non1d_sfids(self):
        with pytest.raises(Exception) as excinfo:
            eda.SampleFeatureMatrix(self.sfm3x3_arr, np.array([[0], [1], [2]]),
                                    np.array([[0], [1], [1]]))

        with pytest.raises(Exception) as excinfo:
            eda.SampleFeatureMatrix(self.sfm3x3_arr, np.array([[0], [1], [2]]),
                                    np.array([0, 1, 2]))

        with pytest.raises(Exception) as excinfo:
            eda.SampleFeatureMatrix(self.sfm3x3_arr, np.array([0, 1, 2]),
                                    np.array([[0], [1], [2]]))

    def test_init_bad_sid_type(self):
        with pytest.raises(Exception) as excinfo:
            eda.SampleFeatureMatrix(
                self.sfm3x3_arr, [False, True, 2], [0, 1, 1])

        with pytest.raises(Exception) as excinfo:
            eda.SampleFeatureMatrix(
                self.sfm3x3_arr, [[0], [0, 1], 2], [0, 1, 1])

        with pytest.raises(Exception) as excinfo:
            eda.SampleFeatureMatrix(
                self.sfm3x3_arr, np.array([0, 1, 2]), [0, 1, 1])

        with pytest.raises(Exception) as excinfo:
            eda.SampleFeatureMatrix(
                self.sfm3x3_arr, [(0), (0, 1), 2], [0, 1, 1])

    def test_init_bad_fid_type(self):
        with pytest.raises(Exception) as excinfo:
            eda.SampleFeatureMatrix(
                self.sfm3x3_arr, [0, 1, 2], [False, True, 2])

        with pytest.raises(Exception) as excinfo:
            eda.SampleFeatureMatrix(
                self.sfm3x3_arr, [0, 1, 2], [[0], [0, 1], 2])

        with pytest.raises(Exception) as excinfo:
            eda.SampleFeatureMatrix(
                self.sfm3x3_arr, [0, 1, 2], [(0), (0, 1), 2])

        with pytest.raises(Exception) as excinfo:
            eda.SampleFeatureMatrix(
                self.sfm3x3_arr, [0, 1, 2], np.array([0, 1, 2]))

    def test_valid_init(self):
        eda.SampleFeatureMatrix(
            self.sfm5x10_arr, list(range(5)), list(range(10)))
        eda.SampleFeatureMatrix(self.sfm5x10_arr, None, list(range(10)))
        eda.SampleFeatureMatrix(self.sfm5x10_arr, list(range(5)), None)
        eda.SampleFeatureMatrix(np.arange(10).reshape(-1, 1))
        eda.SampleFeatureMatrix(np.arange(10).reshape(1, -1))

    def test_ind_x(self):
        sids = list("abcdef")
        fids = list(range(10, 20))
        sfm = eda.SampleFeatureMatrix(
            np.random.ranf(60).reshape(6, -1), sids=sids, fids=fids)
        # select sf
        ss_sfm = sfm.ind_x([0, 5], list(range(9)))
        assert ss_sfm._x.shape == (2, 9)
        assert ss_sfm.sids == ['a', 'f']
        assert ss_sfm.fids == list(range(10, 19))

        # select with Default
        ss_sfm = sfm.ind_x()
        assert ss_sfm._x.shape == (6, 10)
        assert ss_sfm.sids == list("abcdef")
        assert ss_sfm.fids == list(range(10, 20))
        
        # select with None
        ss_sfm = sfm.ind_x(None, None)
        assert ss_sfm._x.shape == (6, 10)
        assert ss_sfm.sids == list("abcdef")
        assert ss_sfm.fids == list(range(10, 20))
                        
        # select non-existent inds
        with pytest.raises(IndexError) as excinfo:
            sfm.ind_x([6])

        with pytest.raises(IndexError) as excinfo:
            sfm.ind_x(None, ['a'])

    def test_ind_x_empty(self):
        sids = list("abcdef")
        fids = list(range(10, 20))
        sfm = eda.SampleFeatureMatrix(
            np.random.ranf(60).reshape(6, -1), sids=sids, fids=fids)
        empty_s = sfm.ind_x([])
        assert empty_s._x.shape == (0, 10)
        assert empty_s._sids.shape == (0,)
        assert empty_s._fids.shape == (10,)

        empty_f = sfm.ind_x(None, [])
        assert empty_f._x.shape == (6, 0)
        assert empty_f._sids.shape == (6,)
        assert empty_f._fids.shape == (0,)

        empty_sf = sfm.ind_x([], [])
        assert empty_sf._x.shape == (0, 0)
        assert empty_sf._sids.shape == (0,)
        assert empty_sf._fids.shape == (0,)

    def test_id_x(self):
        sids = list("abcdef")
        fids = list(range(10, 20))
        sfm = eda.SampleFeatureMatrix(
            np.random.ranf(60).reshape(6, -1), sids=sids, fids=fids)
        # select sf
        ss_sfm = sfm.id_x(['a', 'f'], list(range(10, 15)))
        assert ss_sfm._x.shape == (2, 5)
        assert ss_sfm.sids == ['a', 'f']
        assert ss_sfm.fids == list(range(10, 15))

        # select with Default
        ss_sfm = sfm.id_x()
        assert ss_sfm._x.shape == (6, 10)
        assert ss_sfm.sids == list("abcdef")
        assert ss_sfm.fids == list(range(10, 20))
        
        # select with None
        ss_sfm = sfm.id_x(None, None)
        assert ss_sfm._x.shape == (6, 10)
        assert ss_sfm.sids == list("abcdef")
        assert ss_sfm.fids == list(range(10, 20))
                        
        # select non-existent inds
        # id lookup raises ValueError
        with pytest.raises(ValueError) as excinfo:
            sfm.id_x([6])

        with pytest.raises(ValueError) as excinfo:
            sfm.id_x(None, ['a'])

    def test_id_x_empty(self):
        sids = list("abcdef")
        fids = list(range(10, 20))
        sfm = eda.SampleFeatureMatrix(
            np.random.ranf(60).reshape(6, -1), sids=sids, fids=fids)
        empty_s = sfm.id_x([])
        assert empty_s._x.shape == (0, 10)
        assert empty_s._sids.shape == (0,)
        assert empty_s._fids.shape == (10,)

        empty_f = sfm.id_x(None, [])
        assert empty_f._x.shape == (6, 0)
        assert empty_f._sids.shape == (6,)
        assert empty_f._fids.shape == (0,)

        empty_sf = sfm.id_x([], [])
        assert empty_sf._x.shape == (0, 0)
        assert empty_sf._sids.shape == (0,)
        assert empty_sf._fids.shape == (0,)

    @pytest.mark.mpl_image_compare
    def test_s_ind_regression_scatter_ax(self):
        fig, axs = plt.subplots(ncols=2)
        fig = self.plt_sdm.s_ind_regression_scatter(
            0, 1, figsize=(5, 5), ax=axs[0], ci=None)
        plt.close()
        return fig

    @pytest.mark.mpl_image_compare
    def test_s_ind_regression_scatter(self):
        return self.plt_sdm.s_ind_regression_scatter(0, 1, figsize=(5, 5), ci=None)

    @pytest.mark.mpl_image_compare
    def test_s_id_regression_scatter(self):
        return self.plt_sdm.s_id_regression_scatter("a", "b", 
                                           feature_filter=[1,2,3],
                                           figsize=(5, 5), ci=None)

    @pytest.mark.mpl_image_compare
    def test_s_ind_regression_scatter_custom_labs(self):
        return self.plt_sdm.s_ind_regression_scatter(0, 1, xlab='X', ylab='Y', 
                                            figsize=(5, 5), ci=None)

    @pytest.mark.mpl_image_compare
    def test_s_ind_regression_scatter_custom_bool_ff(self):
        return self.plt_sdm.s_ind_regression_scatter(
            0, 1, feature_filter=[True]*2 + [False]*8, figsize=(5, 5), ci=None)

    @pytest.mark.mpl_image_compare
    def test_s_ind_regression_scatter_custom_int_ff(self):
        return self.plt_sdm.s_ind_regression_scatter(
            0, 1, feature_filter=[0, 1], figsize=(5, 5), ci=None)

    @pytest.mark.mpl_image_compare
    def test_s_ind_regression_scatter_custom_func_ff(self):
        return self.plt_sdm.s_ind_regression_scatter(
            0, 1, feature_filter=lambda x, y: (x in (0, 1, 2)) and (10 < y < 12), 
            figsize=(5, 5), ci=None)

    @pytest.mark.mpl_image_compare
    def test_f_ind_regression_scatter_custom_func_sf(self):
        # array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
        #        [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        #        [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
        #        [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
        #        [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
        #        [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]])
        return self.plt_sdm.f_ind_regression_scatter(
            0, 1, sample_filter=lambda x, y: (x in (0, 10, 20)) and (10 < y < 30), 
            figsize=(5, 5), ci=None)

    @pytest.mark.mpl_image_compare
    def test_f_ind_regression_scatter_no_ff(self):
        return self.plt_sdm.f_ind_regression_scatter(0, 1, figsize=(5, 5), ci=None)

    @pytest.mark.mpl_image_compare
    def test_f_ind_regression_scatter_ind_ff(self):
        return self.plt_sdm.f_ind_regression_scatter(0, 1, sample_filter=[0, 2, 5], 
                                            figsize=(5, 5), ci=None)

    @pytest.mark.mpl_image_compare
    def test_f_ind_regression_scatter_labs(self):
        return self.plt_sdm.f_ind_regression_scatter(0, 1, sample_filter=[0, 2, 5], 
                                            figsize=(5, 5), title='testregscat',
                                            xlab='x', ylab='y', ci=None)

    @pytest.mark.mpl_image_compare
    def test_f_id_regression_scatter(self):
        return self.plt_sdm.f_id_regression_scatter(
            'f5', 'f6', sample_filter=[0, 2, 5], figsize=(5, 5), ci=None)

    @pytest.mark.mpl_image_compare
    @pytest.mark.filterwarnings("ignore:The 'normed' kwarg is depreca")
    def test_s_ind_dist_ax(self):
        fig, axs = plt.subplots(ncols=2)
        fig = self.plt_sdm.s_ind_dist(0, figsize=(5, 5), ax=axs[0])
        plt.close()
        return fig

    @pytest.mark.mpl_image_compare
    @pytest.mark.filterwarnings("ignore:The 'normed' kwarg is depreca")
    def test_s_ind_dist(self):
        return self.plt_sdm.s_ind_dist(0, figsize=(5, 5))

    @pytest.mark.mpl_image_compare
    @pytest.mark.filterwarnings("ignore:The 'normed' kwarg is depreca")
    def test_s_id_dist(self):
        return self.plt_sdm.s_id_dist("a", feature_filter=[1,2,3], 
                                      figsize=(5, 5))

    @pytest.mark.mpl_image_compare
    @pytest.mark.filterwarnings("ignore:The 'normed' kwarg is depreca")
    def test_s_ind_dist_custom_labs(self):
        return self.plt_sdm.s_ind_dist(0, xlab='X', ylab='Y', figsize=(5, 5))

    @pytest.mark.mpl_image_compare
    @pytest.mark.filterwarnings("ignore:The 'normed' kwarg is depreca")
    def test_s_ind_dist_custom_bool_ff(self):
        return self.plt_sdm.s_ind_dist(
            0, feature_filter=[True]*2 + [False]*8, title='testdist',
            figsize=(5, 5))

    @pytest.mark.mpl_image_compare
    @pytest.mark.filterwarnings("ignore:The 'normed' kwarg is depreca")
    def test_s_ind_dist_custom_int_ff(self):
        return self.plt_sdm.s_ind_dist(
            0, feature_filter=[0, 1], figsize=(5, 5))

    @pytest.mark.mpl_image_compare
    @pytest.mark.filterwarnings("ignore:The 'normed' kwarg is depreca")
    def test_s_ind_dist_custom_func_ff(self):
        return self.plt_sdm.s_ind_dist(
            0, feature_filter=lambda x: x in (0, 1, 2), 
            figsize=(5, 5))

    @pytest.mark.mpl_image_compare
    @pytest.mark.filterwarnings("ignore:The 'normed' kwarg is depreca")
    def test_f_ind_dist_custom_func_sf(self):
        return self.plt_sdm.f_ind_dist(
            0, sample_filter=lambda x: x in (0, 10, 20), 
            figsize=(5, 5))

    @pytest.mark.mpl_image_compare
    @pytest.mark.filterwarnings("ignore:The 'normed' kwarg is depreca")
    def test_f_ind_dist_no_ff(self):
        return self.plt_sdm.f_ind_dist(0, figsize=(5, 5))

    @pytest.mark.mpl_image_compare
    @pytest.mark.filterwarnings("ignore:The 'normed' kwarg is depreca")
    def test_f_ind_dist_ind_ff(self):
        return self.plt_sdm.f_ind_dist(0, sample_filter=[0, 2, 5], 
                                       figsize=(5, 5))

    @pytest.mark.mpl_image_compare
    @pytest.mark.filterwarnings("ignore:The 'normed' kwarg is depreca")
    def test_f_ind_dist_labs(self):
        return self.plt_sdm.f_ind_dist(0, sample_filter=[0, 2, 5], 
                                       figsize=(5, 5), 
                                       xlab='x', ylab='y')

    @pytest.mark.mpl_image_compare
    @pytest.mark.filterwarnings("ignore:The 'normed' kwarg is depreca")
    def test_f_id_dist(self):
        return self.plt_sdm.f_id_dist('f5', sample_filter=[0, 2, 5], 
                                      figsize=(5, 5))

    def test_getters(self):
        tsfm = eda.SampleFeatureMatrix(np.arange(10).reshape(5, 2),
                                       ['a', 'b', 'c', '1', '2'],
                                       ['a', 'z'])

        np.testing.assert_equal(tsfm.x, np.array(
            np.arange(10).reshape(5, 2), dtype='float64'))
        np.testing.assert_equal(tsfm.sids, np.array(['a', 'b', 'c', '1', '2']))
        np.testing.assert_equal(tsfm.fids, np.array(['a', 'z']))

        assert tsfm.x is not tsfm._x
        assert tsfm.sids is not tsfm._sids
        assert tsfm.fids is not tsfm._fids

    # array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
    #        [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    #        [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
    #        [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
    #        [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
    #        [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]])
    def test_f_sum(self):
        x = self.plt_sdm.f_sum()
        assert x.ndim == 1
        assert x.shape[0] == 6
        np.testing.assert_allclose(x, self.ref_plt_f_sum)
        # only need to test that filter has been passed correctly
        np.testing.assert_allclose(self.plt_sdm.f_sum([0, 1, 2]), 
            self.ref_plt_f_sum[:3])

    def test_s_sum(self):
        x = self.plt_sdm.s_sum()
        assert x.ndim == 1
        assert x.shape[0] == 10
        np.testing.assert_allclose(x, self.ref_plt_s_sum)
        np.testing.assert_allclose(self.plt_sdm.s_sum([0, 1, 2]), 
            self.ref_plt_s_sum[:3])

    def test_f_cv(self):
        x = self.plt_sdm.f_cv()
        assert x.ndim == 1
        assert x.shape[0] == 6
        np.testing.assert_allclose(self.plt_sdm.f_cv(), self.ref_plt_f_cv)
        np.testing.assert_allclose(self.plt_sdm.f_cv([0, 1, 2]), 
            self.ref_plt_f_cv[:3])

    def test_s_cv(self):
        x = self.plt_sdm.s_cv()
        assert x.ndim == 1
        assert x.shape[0] == 10
        np.testing.assert_allclose(x, self.ref_plt_s_cv)
        np.testing.assert_allclose(self.plt_sdm.s_cv([0, 1, 2]),
            self.ref_plt_s_cv[:3])

    def test_f_gc(self):
        x = self.plt_sdm.f_gc()
        assert x.ndim == 1
        assert x.shape[0] == 6
        np.testing.assert_allclose(x, self.ref_plt_f_gc)
        np.testing.assert_allclose(self.plt_sdm.f_gc([0, 1, 2]), 
            self.ref_plt_f_gc[:3])

    def test_s_gc(self):
        x = self.plt_sdm.s_gc()
        assert x.ndim == 1
        assert x.shape[0] == 10
        np.testing.assert_allclose(x, self.ref_plt_s_gc)
        np.testing.assert_allclose(self.plt_sdm.s_gc([0, 1, 2]),
            self.ref_plt_s_gc[:3])

    def test_f_ath(self):
        x = self.plt_sdm.f_n_above_threshold(15)
        assert x.ndim == 1
        assert x.shape[0] == 6
        np.testing.assert_allclose(x, self.ref_plt_f_a15)

    def test_s_ath(self):
        x = self.plt_sdm.s_n_above_threshold(35)
        assert x.ndim == 1
        assert x.shape[0] == 10
        np.testing.assert_allclose(x, self.ref_plt_s_a35)

    # Because summary dist plot calls hist_dens_plot immediately after 
    # obtaining the summary statistics vector, the correctness of summary
    # statistics vector and hist_dens_plot implies the correctness of the
    # plots.
    @pytest.mark.filterwarnings("ignore:The 'normed' kwarg is depreca")
    def test_summary_stat_dist(self):
        self.plt_sdm.f_sum_dist([0, 1, 2])
        self.plt_sdm.s_sum_dist([0, 1, 2])
        self.plt_sdm.f_cv_dist([0, 1, 2])
        self.plt_sdm.s_cv_dist([0, 1, 2])
        self.plt_sdm.f_gc_dist([0, 1, 2])
        self.plt_sdm.s_gc_dist([0, 1, 2])
        self.plt_sdm.f_n_above_threshold_dist(15)
        self.plt_sdm.s_n_above_threshold_dist(15)


