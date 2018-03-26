import numpy as np
import seaborn as sns
import scedar.eda as eda
import matplotlib as mpl
import pytest


class TestSampleFeatureMatrix(object):
    """docstring for TestSampleFeatureMatrix"""
    sfm5x10_arr = np.random.ranf(50).reshape(5, 10)
    sfm3x3_arr = np.random.ranf(9).reshape(3, 3)
    sfm5x10_lst = list(map(list, np.random.ranf(50).reshape(5, 10)))

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
            eda.SampleFeatureMatrix(self.sfm5x10_lst, None, [
                                    '0', '0', '1', '2', '3'])

    def test_init_empty_x_sfids(self):
        with pytest.raises(Exception) as excinfo:
            eda.SampleFeatureMatrix(np.array([[], []]), [])

        with pytest.raises(Exception) as excinfo:
            eda.SampleFeatureMatrix(np.array([[], []]), None, [])

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

    def test_is_valid_sfid(self):
        assert eda.SampleFeatureMatrix.is_valid_sfid('1')
        assert eda.SampleFeatureMatrix.is_valid_sfid(1)
        assert not eda.SampleFeatureMatrix.is_valid_sfid(np.array([1])[0])
        assert not eda.SampleFeatureMatrix.is_valid_sfid([])
        assert not eda.SampleFeatureMatrix.is_valid_sfid([1])
        assert not eda.SampleFeatureMatrix.is_valid_sfid(None)
        assert not eda.SampleFeatureMatrix.is_valid_sfid((1,))

    def test_check_is_valid_sfids(self):
        with pytest.raises(Exception) as excinfo:
            eda.SampleFeatureMatrix.check_is_valid_sfids(np.arange(5))

        with pytest.raises(Exception) as excinfo:
            eda.SampleFeatureMatrix.check_is_valid_sfids([True, False])

        with pytest.raises(Exception) as excinfo:
            eda.SampleFeatureMatrix.check_is_valid_sfids(None)

        with pytest.raises(Exception) as excinfo:
            eda.SampleFeatureMatrix.check_is_valid_sfids([])

        with pytest.raises(Exception) as excinfo:
            eda.SampleFeatureMatrix.check_is_valid_sfids([[1], [2]])

        with pytest.raises(Exception) as excinfo:
            eda.SampleFeatureMatrix.check_is_valid_sfids(['1', 2, 3])

        with pytest.raises(Exception) as excinfo:
            eda.SampleFeatureMatrix.check_is_valid_sfids(['1', '1', '3'])

        with pytest.raises(Exception) as excinfo:
            eda.SampleFeatureMatrix.check_is_valid_sfids([0, 0, 1])

        with pytest.raises(Exception) as excinfo:
            eda.SampleFeatureMatrix.check_is_valid_sfids(['1', 2, '3'])

        eda.SampleFeatureMatrix.check_is_valid_sfids([1, 2])
        eda.SampleFeatureMatrix.check_is_valid_sfids(['1', '2'])
        eda.SampleFeatureMatrix.check_is_valid_sfids([1, 2, 3])

    def test_ind_x(self):
        sids = list("abcdef")
        fids = list(range(10, 20))
        sdm = eda.SampleFeatureMatrix(
            np.random.ranf(60).reshape(6, -1), sids=sids, fids=fids)
        # select sf
        ss_sdm = sdm.ind_x([0, 5], list(range(9)))
        assert ss_sdm._x.shape == (2, 9)
        assert ss_sdm.sids == ['a', 'f']
        assert ss_sdm.fids == list(range(10, 19))

        # select with Default
        ss_sdm = sdm.ind_x()
        assert ss_sdm._x.shape == (6, 10)
        assert ss_sdm.sids == list("abcdef")
        assert ss_sdm.fids == list(range(10, 20))
        
        # select with None
        ss_sdm = sdm.ind_x(None, None)
        assert ss_sdm._x.shape == (6, 10)
        assert ss_sdm.sids == list("abcdef")
        assert ss_sdm.fids == list(range(10, 20))
                        
        # select non-existent inds
        with pytest.raises(IndexError) as excinfo:
            sdm.ind_x([6])

        with pytest.raises(IndexError) as excinfo:
            sdm.ind_x(None, ['a'])

        # select 0 ind
        # does not support empty matrix
        with pytest.raises(ValueError) as excinfo:
            sdm.ind_x([])

        with pytest.raises(ValueError) as excinfo:
            sdm.ind_x(None, [])

    def test_id_x(self):
        sids = list("abcdef")
        fids = list(range(10, 20))
        sdm = eda.SampleFeatureMatrix(
            np.random.ranf(60).reshape(6, -1), sids=sids, fids=fids)
        # select sf
        ss_sdm = sdm.id_x(['a', 'f'], list(range(10, 15)))
        assert ss_sdm._x.shape == (2, 5)
        assert ss_sdm.sids == ['a', 'f']
        assert ss_sdm.fids == list(range(10, 15))

        # select with Default
        ss_sdm = sdm.id_x()
        assert ss_sdm._x.shape == (6, 10)
        assert ss_sdm.sids == list("abcdef")
        assert ss_sdm.fids == list(range(10, 20))
        
        # select with None
        ss_sdm = sdm.id_x(None, None)
        assert ss_sdm._x.shape == (6, 10)
        assert ss_sdm.sids == list("abcdef")
        assert ss_sdm.fids == list(range(10, 20))
                        
        # select non-existent inds
        # id lookup raises ValueError
        with pytest.raises(ValueError) as excinfo:
            sdm.id_x([6])

        with pytest.raises(ValueError) as excinfo:
            sdm.id_x(None, ['a'])

        # select 0 ind
        # does not support empty matrix
        with pytest.raises(ValueError) as excinfo:
            sdm.id_x([])

        with pytest.raises(ValueError) as excinfo:
            sdm.id_x(None, [])

    @pytest.mark.mpl_image_compare
    def test_s_ind_regression_scatter(self):
        sids = list("abcdef")
        fids = list(map(lambda i: 'f{}'.format(i), range(10)))
        sfm = eda.SampleFeatureMatrix(np.arange(60).reshape(6, 10), 
                                      sids=sids, fids=fids)
        return sfm.s_ind_regression_scatter(0, 1, figsize=(5, 5))

    @pytest.mark.mpl_image_compare
    def test_s_id_regression_scatter(self):
        sids = list("abcdef")
        fids = list(map(lambda i: 'f{}'.format(i), range(10)))
        sfm = eda.SampleFeatureMatrix(np.arange(60).reshape(6, 10), 
                                      sids=sids, fids=fids)
        return sfm.s_id_regression_scatter("a", "b", figsize=(5, 5))

    @pytest.mark.mpl_image_compare
    def test_s_ind_regression_scatter_custom_labs(self):
        sids = list("abcdef")
        fids = list(map(lambda i: 'f{}'.format(i), range(10)))
        sfm = eda.SampleFeatureMatrix(np.arange(60).reshape(6, 10), 
                                      sids=sids, fids=fids)
        return sfm.s_ind_regression_scatter(0, 1, xlab='X', ylab='Y', 
                                            figsize=(5, 5))

    @pytest.mark.mpl_image_compare
    def test_s_ind_regression_scatter_custom_bool_ff(self):
        sids = list("abcdef")
        fids = list(map(lambda i: 'f{}'.format(i), range(10)))
        sfm = eda.SampleFeatureMatrix(np.arange(60).reshape(6, 10), 
                                      sids=sids, fids=fids)
        return sfm.s_ind_regression_scatter(
            0, 1, feature_filter=[True]*2 + [False]*8, figsize=(5, 5))

    @pytest.mark.mpl_image_compare
    def test_s_ind_regression_scatter_custom_int_ff(self):
        sids = list("abcdef")
        fids = list(map(lambda i: 'f{}'.format(i), range(10)))
        sfm = eda.SampleFeatureMatrix(np.arange(60).reshape(6, 10), 
                                      sids=sids, fids=fids)
        return sfm.s_ind_regression_scatter(
            0, 1, feature_filter=[0, 1], figsize=(5, 5))

    @pytest.mark.mpl_image_compare
    def test_s_ind_regression_scatter_custom_func_ff(self):
        sids = list("abcdef")
        fids = list(map(lambda i: 'f{}'.format(i), range(10)))
        sfm = eda.SampleFeatureMatrix(np.arange(60).reshape(6, 10), 
                                      sids=sids, fids=fids)
        return sfm.s_ind_regression_scatter(
            0, 1, feature_filter=lambda x, y: (x in (0, 1, 2)) and (10 < y < 12), 
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
        np.testing.assert_allclose(ref_tsne, tsne2)
        assert len(sdm.tsne_lut) == 2

        tsne_res_list = [tsne1, tsne3]
        tsne_res_lut = sdm.tsne_lut
        tsne_res_lut_sorted_keys = sorted(tsne_res_lut.keys())
        for i in range(len(tsne_res_lut)):
            iter_key = tsne_res_lut_sorted_keys[i]
            iter_key[-1] == str(i)
            np.testing.assert_allclose(tsne_res_lut[iter_key],
                                       tsne_res_list[i])
            assert tsne_res_lut[iter_key] is not tsne_res_list[i]

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

        # select 0 ind
        # does not support empty matrix
        with pytest.raises(ValueError) as excinfo:
            sdm.ind_x([])

        with pytest.raises(ValueError) as excinfo:
            sdm.ind_x(None, [])

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

        # select 0 ind
        # does not support empty matrix
        with pytest.raises(ValueError) as excinfo:
            sdm.id_x([])

        with pytest.raises(ValueError) as excinfo:
            sdm.id_x(None, [])

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


class TestSingleLabelClassifiedSamples(object):
    """docstring for TestSingleLabelClassifiedSamples"""
    np.random.seed(123)
    sfm3x3_arr = np.arange(9, dtype="float64").reshape(3, 3)
    sfm_2x0 = np.array([[], []])
    sfm5x10_arr = np.random.ranf(50).reshape(5, 10)
    sfm5x10_lst = list(map(list, np.random.ranf(50).reshape(5, 10)))

    def test_init_empty_labs(self):
        with pytest.raises(Exception) as excinfo:
            eda.SingleLabelClassifiedSamples(self.sfm_2x0, [])

    def test_init_wrong_lab_len(self):
        with pytest.raises(Exception) as excinfo:
            eda.SingleLabelClassifiedSamples(
                self.sfm3x3_arr, [0, 1], None, None)

    def test_init_non1d_labs(self):
        with pytest.raises(Exception) as excinfo:
            eda.SingleLabelClassifiedSamples(self.sfm3x3_arr, [[0], [1], [2]],
                                             [0, 1, 2], [0, 1, 2])

        with pytest.raises(Exception) as excinfo:
            eda.SingleLabelClassifiedSamples(self.sfm3x3_arr,
                                             [[0, 1], [1, 2], [2, 3]],
                                             [0, 1, 2], [0, 1, 2])

    def test_init_bad_lab_type(self):
        with pytest.raises(Exception) as excinfo:
            eda.SingleLabelClassifiedSamples(
                self.sfm3x3_arr, [False, True, 2], [0, 1, 1], None)

        with pytest.raises(Exception) as excinfo:
            eda.SingleLabelClassifiedSamples(
                self.sfm3x3_arr, [[0], [0, 1], 2], [0, 1, 1], None)

        with pytest.raises(Exception) as excinfo:
            eda.SingleLabelClassifiedSamples(
                self.sfm3x3_arr, np.array([0, 1, 2]), [0, 1, 1], None)

        with pytest.raises(Exception) as excinfo:
            eda.SingleLabelClassifiedSamples(
                self.sfm3x3_arr, [(0), (0, 1), 2], [0, 1, 1], None)

    def test_valid_init(self):
        eda.SingleLabelClassifiedSamples(
            self.sfm5x10_arr, [0, 1, 1, 2, 0], list(range(5)), list(range(10)))
        eda.SingleLabelClassifiedSamples(
            self.sfm5x10_arr, [0, 1, 1, 2, 0], None, list(range(10)))
        eda.SingleLabelClassifiedSamples(
            self.sfm5x10_arr, ['a', 'a', 'b', 'd', 'c'], list(range(5)), None)
        eda.SingleLabelClassifiedSamples(
            np.arange(100).reshape(-1, 10), list(range(10)))
        eda.SingleLabelClassifiedSamples(
            np.arange(100).reshape(10, -1), list('abcdefghij'))

    def test_is_valid_lab(self):
        assert eda.SingleLabelClassifiedSamples.is_valid_lab('1')
        assert eda.SingleLabelClassifiedSamples.is_valid_lab(1)
        assert not eda.SingleLabelClassifiedSamples.is_valid_lab(np.array([1])[
                                                                 0])
        assert not eda.SingleLabelClassifiedSamples.is_valid_lab([])
        assert not eda.SingleLabelClassifiedSamples.is_valid_lab([1])
        assert not eda.SingleLabelClassifiedSamples.is_valid_lab(None)
        assert not eda.SingleLabelClassifiedSamples.is_valid_lab((1,))

    def test_check_is_valid_labs(self):
        with pytest.raises(Exception) as excinfo:
            eda.SingleLabelClassifiedSamples.check_is_valid_labs(np.arange(5))

        with pytest.raises(Exception) as excinfo:
            eda.SingleLabelClassifiedSamples.check_is_valid_labs([True, False])

        with pytest.raises(Exception) as excinfo:
            eda.SingleLabelClassifiedSamples.check_is_valid_labs(None)

        with pytest.raises(Exception) as excinfo:
            eda.SingleLabelClassifiedSamples.check_is_valid_labs([])

        with pytest.raises(Exception) as excinfo:
            eda.SingleLabelClassifiedSamples.check_is_valid_labs([[1], [2]])

        with pytest.raises(Exception) as excinfo:
            eda.SingleLabelClassifiedSamples.check_is_valid_labs(['1', 2, 1])

        with pytest.raises(Exception) as excinfo:
            eda.SingleLabelClassifiedSamples.check_is_valid_labs(['1', 2, '3'])

        eda.SingleLabelClassifiedSamples.check_is_valid_labs([1, 2])
        eda.SingleLabelClassifiedSamples.check_is_valid_labs([1, 1, 3])
        eda.SingleLabelClassifiedSamples.check_is_valid_labs(['1', '2', '3'])

    def test_lab_sorted_sids(self):
        qsids = [0, 1, 5, 3, 2, 4]
        qlabs = [0, 0, 2, 1, 1, 1]
        rsids = [3, 4, 2, 5, 1, 0]
        slab_csamples = eda.SingleLabelClassifiedSamples(np.random.ranf(60).reshape(6, -1),
                                                         qlabs, qsids)
        rs_qsids, rs_qlabs = slab_csamples.lab_sorted_sids(rsids)
        np.testing.assert_equal(rs_qsids, np.array([3, 4, 2, 5, 1, 0]))
        np.testing.assert_equal(rs_qlabs, np.array([1, 1, 1, 2, 0, 0]))

        rs_qsids, rs_qlabs = slab_csamples.lab_sorted_sids()
        np.testing.assert_equal(rs_qsids, np.array([0, 1, 3, 2, 4, 5]))
        np.testing.assert_equal(rs_qlabs, np.array([0, 0, 1, 1, 1, 2]))

    def test_filter_min_class_n(self):
        sids = [0, 1, 2, 3, 4, 5]
        labs = [0, 0, 0, 1, 2, 2]
        slab_csamples = eda.SingleLabelClassifiedSamples(np.random.ranf(60).reshape(6, -1),
                                                         labs, sids, None)
        min_cl_n = 2
        mcnf_slab_csamples = slab_csamples.filter_min_class_n(min_cl_n)
        np.testing.assert_equal(mcnf_slab_csamples.sids,
                                np.array([0, 1, 2, 4, 5]))
        np.testing.assert_equal(mcnf_slab_csamples.labs,
                                np.array([0, 0, 0, 2, 2]))
        np.testing.assert_equal(mcnf_slab_csamples._x.shape,
                                (5, 10))
        np.testing.assert_equal(mcnf_slab_csamples.fids,
                                slab_csamples.fids)
        np.testing.assert_equal(mcnf_slab_csamples._x,
                                slab_csamples._x[np.array([0, 1, 2, 4, 5])])
        np.testing.assert_equal(mcnf_slab_csamples._d,
                                slab_csamples._d[np.array([0, 1, 2, 4, 5])][:, np.array([0, 1, 2, 4, 5])])

    def test_ind_x(self):
        sids = list("abcdef")
        fids = list(range(10, 20))
        labs = [0, 0, 0, 1, 2, 2]

        sdm = eda.SingleLabelClassifiedSamples(
            np.random.ranf(60).reshape(6, -1), labs = labs,
            sids=sids, fids=fids)
        # select sf
        ss_sdm = sdm.ind_x([0, 5], list(range(9)))
        assert ss_sdm._x.shape == (2, 9)
        assert ss_sdm.sids == ['a', 'f']
        assert ss_sdm.labs == [0, 2]
        assert ss_sdm.fids == list(range(10, 19))
        np.testing.assert_equal(
            ss_sdm.d, sdm._d[np.ix_((0, 5), (0, 5))])

        # select with Default
        ss_sdm = sdm.ind_x()
        assert ss_sdm._x.shape == (6, 10)
        assert ss_sdm.sids == list("abcdef")
        assert ss_sdm.labs == labs
        assert ss_sdm.fids == list(range(10, 20))
        np.testing.assert_equal(ss_sdm.d, sdm._d)
        
        # select with None
        ss_sdm = sdm.ind_x(None, None)
        assert ss_sdm._x.shape == (6, 10)
        assert ss_sdm.sids == list("abcdef")
        assert ss_sdm.labs == labs
        assert ss_sdm.fids == list(range(10, 20))
        np.testing.assert_equal(ss_sdm.d, sdm._d)
                        
        # select non-existent inds
        with pytest.raises(IndexError) as excinfo:
            sdm.ind_x([6])

        with pytest.raises(IndexError) as excinfo:
            sdm.ind_x(None, ['a'])

        # select 0 ind
        # does not support empty matrix
        with pytest.raises(ValueError) as excinfo:
            sdm.ind_x([])

        with pytest.raises(ValueError) as excinfo:
            sdm.ind_x(None, [])

    def test_id_x(self):
        sids = list("abcdef")
        fids = list(range(10, 20))
        labs = [0, 0, 0, 1, 2, 2]

        sdm = eda.SingleLabelClassifiedSamples(
            np.random.ranf(60).reshape(6, -1), labs = labs,
            sids=sids, fids=fids)
        # select sf
        ss_sdm = sdm.id_x(['a', 'f'], list(range(10, 15)))
        assert ss_sdm._x.shape == (2, 5)
        assert ss_sdm.sids == ['a', 'f']
        assert ss_sdm.labs == [0, 2]
        assert ss_sdm.fids == list(range(10, 15))
        np.testing.assert_equal(
            ss_sdm.d, sdm._d[np.ix_((0, 5), (0, 5))])

        # select with Default
        ss_sdm = sdm.id_x()
        assert ss_sdm._x.shape == (6, 10)
        assert ss_sdm.sids == list("abcdef")
        assert ss_sdm.labs == labs
        assert ss_sdm.fids == list(range(10, 20))
        np.testing.assert_equal(ss_sdm.d, sdm._d)
        
        # select with None
        ss_sdm = sdm.id_x(None, None)
        assert ss_sdm._x.shape == (6, 10)
        assert ss_sdm.sids == list("abcdef")
        assert ss_sdm.labs == labs
        assert ss_sdm.fids == list(range(10, 20))
        np.testing.assert_equal(ss_sdm.d, sdm._d)
                        
        # select non-existent inds
        # id lookup raises ValueError
        with pytest.raises(ValueError) as excinfo:
            sdm.id_x([6])

        with pytest.raises(ValueError) as excinfo:
            sdm.id_x(None, ['a'])

        # select 0 ind
        # does not support empty matrix
        with pytest.raises(ValueError) as excinfo:
            sdm.id_x([])

        with pytest.raises(ValueError) as excinfo:
            sdm.id_x(None, [])

    def test_cross_labs(self):
        rsids = [0, 1, 2, 3, 4]
        rlabs = [0, 0, 0, 1, 1]
        rscl_samples = eda.SingleLabelClassifiedSamples(
            self.sfm5x10_lst, rlabs, rsids)

        qsids = [0, 1, 2, 3, 4]
        qlabs = [1, 1, 0, 2, 3]
        qscl_samples = eda.SingleLabelClassifiedSamples(
            self.sfm5x10_lst, qlabs, qsids)

        cross_lab_lut = rscl_samples.cross_labs(qscl_samples)
        test_lut = {
            0: (3, ((0, 1), (1, 2))),
            1: (2, ((2, 3), (1, 1)))
        }
        assert cross_lab_lut == test_lut

        qsids2 = [0, 1, 2]
        qlabs2 = [1, 1, 0]
        qscl_samples2 = eda.SingleLabelClassifiedSamples(
            self.sfm3x3_arr, qlabs2, qsids2)

        cross_lab_lut2 = rscl_samples.cross_labs(qscl_samples2)
        test_lut2 = {
            0: (3, ((0, 1), (1, 2)))
        }
        assert cross_lab_lut2 == test_lut2

        with pytest.raises(Exception) as excinfo:
            rscl_samples.cross_labs([1, 2, 3])

        qsfm = eda.SampleFeatureMatrix(self.sfm5x10_lst)
        with pytest.raises(Exception) as excinfo:
            rscl_samples.cross_labs(qsfm)

        # Contains mismatch to rsids
        mm_qsids = [0, 1, 6]
        mm_qlabs = [1, 1, 0]
        mm_qscl_samples = eda.SingleLabelClassifiedSamples(self.sfm3x3_arr,
                                                           mm_qlabs, mm_qsids)
        with pytest.raises(Exception) as excinfo:
            rscl_samples.cross_labs(mm_qscl_samples)

    def test_labs_to_cmap(self):
        sids = [0, 1, 2, 3, 4, 5, 6, 7]
        labs = list(map(str, [3, 0, 1, 0, 0, 1, 2, 2]))
        slab_csamples = eda.SingleLabelClassifiedSamples(np.random.ranf(80).reshape(8, -1),
                                                         labs, sids)

        lab_cmap, lab_ind_arr, lab_col_lut, uniq_lab_lut = eda.SingleLabelClassifiedSamples.labs_to_cmap(
            slab_csamples.labs, return_lut=True)

        n_uniq_labs = len(set(labs))
        assert lab_cmap.N == n_uniq_labs
        assert lab_cmap.colors == sns.hls_palette(n_uniq_labs)
        np.testing.assert_equal(
            lab_ind_arr, np.array([3, 0, 1, 0, 0, 1, 2, 2]))
        assert labs == [uniq_lab_lut[x] for x in lab_ind_arr]
        assert len(uniq_lab_lut) == n_uniq_labs
        assert len(lab_col_lut) == n_uniq_labs
        assert [lab_col_lut[uniq_lab_lut[i]]
                for i in range(n_uniq_labs)] == sns.hls_palette(n_uniq_labs)

        lab_cmap2 = eda.SingleLabelClassifiedSamples.labs_to_cmap(
            slab_csamples.labs, return_lut=False)
        assert lab_cmap2.N == n_uniq_labs
        assert lab_cmap2.colors == lab_cmap.colors

    @pytest.mark.mpl_image_compare
    def test_tsne_feature_gradient_plot(self):
        sids = list(range(8))
        fids = [str(i) for i in range(10)]
        labs = list(range(8))
        np.random.seed(123)
        x = np.random.ranf(80).reshape(8, -1)
        x_sorted = x[np.argsort(x[:, 5])]
        slab_csamples = eda.SingleLabelClassifiedSamples(
            x_sorted, labs, sids=sids, fids=fids)
        return slab_csamples.tsne_feature_gradient_plot(
            '5', figsize=(10, 10), s=50)

    @pytest.mark.mpl_image_compare
    def test_tsne_gradient_plot(self):
        sids = list(range(8))
        fids = [str(i) for i in range(10)]
        labs = list(range(8))
        np.random.seed(123)
        x = np.random.ranf(80).reshape(8, -1)
        x_sorted = x[np.argsort(x[:, 5])]
        g = x_sorted[:, 5]
        slab_csamples = eda.SingleLabelClassifiedSamples(
            x_sorted, labs, sids=sids, fids=fids)
        return slab_csamples.tsne_gradient_plot(g, figsize=(10, 10), s=50)

    def test_tsne_feature_gradient_plot_wrong_args(self):
        sids = list(range(8))
        fids = [str(i) for i in range(10)]
        labs = list(range(8))
        np.random.seed(123)
        x = np.random.ranf(80).reshape(8, -1)
        x_sorted = x[np.argsort(x[:, 5])]
        slab_csamples = eda.SingleLabelClassifiedSamples(
            x, labs, sids=sids, fids=fids)
        with pytest.raises(ValueError):
            slab_csamples.tsne_feature_gradient_plot([0, 1])
        with pytest.raises(ValueError):
            slab_csamples.tsne_feature_gradient_plot(11)
        with pytest.raises(ValueError):
            slab_csamples.tsne_feature_gradient_plot(11)
        with pytest.raises(ValueError):
            slab_csamples.tsne_feature_gradient_plot(-1)
        with pytest.raises(ValueError):
            slab_csamples.tsne_feature_gradient_plot(5)
        with pytest.raises(ValueError):
            slab_csamples.tsne_feature_gradient_plot('123')

    def test_getters(self):
        tslcs = eda.SingleLabelClassifiedSamples(np.arange(10).reshape(5, 2),
                                                 [0, 0, 1, 2, 3],
                                                 ['a', 'b', 'c', '1', '2'],
                                                 ['a', 'z'])

        np.testing.assert_equal(tslcs.x, np.array(
            np.arange(10).reshape(5, 2), dtype='float64'))
        np.testing.assert_equal(
            tslcs.sids, np.array(['a', 'b', 'c', '1', '2']))
        np.testing.assert_equal(tslcs.fids, np.array(['a', 'z']))
        np.testing.assert_equal(tslcs.labs, np.array([0, 0, 1, 2, 3]))

        assert tslcs.x is not tslcs._x
        assert tslcs.sids is not tslcs._sids
        assert tslcs.fids is not tslcs._fids
        assert tslcs.labs is not tslcs._labs

    def test_lab_to_sids(self):
        tslcs = eda.SingleLabelClassifiedSamples(np.arange(10).reshape(5, 2),
                                                 [0, 0, 1, 2, 3],
                                                 ['a', 'b', 'c', '1', '2'],
                                                 ['a', 'z'])
        qsid_arr = tslcs.labs_to_sids((0, 1))
        np.testing.assert_equal(qsid_arr, (('a', 'b'), ('c',)))

    def test_sids_to_labs(self):
        tslcs = eda.SingleLabelClassifiedSamples(np.arange(10).reshape(5, 2),
                                                 [0, 0, 1, 2, 3],
                                                 ['a', 'b', 'c', '1', '2'],
                                                 ['a', 'z'])
        qlab_arr = tslcs.sids_to_labs(('a', 'b', '2'))
        np.testing.assert_equal(qlab_arr, np.array([0, 0, 3]))

        qlab_arr = tslcs.sids_to_labs(('1', 'a', 'b', '2'))
        np.testing.assert_equal(qlab_arr, np.array([2, 0, 0, 3]))


class TestRegressionScatter(object):
    """docstring for TestRegressionPlot"""
    @pytest.mark.mpl_image_compare
    def test_reg_sct_full_labels(self):
        fig = eda.regression_scatter(x=np.arange(10), y=np.arange(10, 20),
                                     xlab='x', ylab='y', title='x versus y',
                                     figsize=(10, 10))
        return fig

    @pytest.mark.mpl_image_compare
    def test_regression_no_label(self):
        fig = eda.regression_scatter(x=np.arange(10), y=np.arange(10, 20),
                                     figsize=(10, 10))
        return fig


class TestClusterScatter(object):
    """docstring for TestClusterScatter"""
    np.random.seed(123)
    x_50x2 = np.random.ranf(100).reshape(50, 2)

    def test_cluster_scatter_no_randstate(self):
        eda.cluster_scatter(self.x_50x2,
                            [0]*25 + [1]*25,
                            title='test tsne scatter',
                            xlab='tsne1', ylab='tsne2',
                            figsize=(10, 10), n_txt_per_cluster=3,
                            alpha=0.5, s=50)
        eda.cluster_scatter(self.x_50x2,
                            [0]*25 + [1]*25,
                            title='test tsne scatter',
                            xlab='tsne1', ylab='tsne2',
                            figsize=(10, 10), n_txt_per_cluster=3, alpha=0.5,
                            random_state=None, s=50)

    @pytest.mark.mpl_image_compare
    def test_cluster_scatter_no_xylab_title(self):
        fig = eda.cluster_scatter(self.x_50x2,
                                  [0]*25 + [1]*10 + [2]*15,
                                  figsize=(10, 10), n_txt_per_cluster=3, alpha=0.5,
                                  s=50, random_state=123)
        return fig

    @pytest.mark.mpl_image_compare
    def test_cluster_scatter_gradient_legends(self):
        sorted_x = self.x_50x2[np.argsort(self.x_50x2[:, 1])]
        fig = eda.cluster_scatter(sorted_x,
                                  labels=[0]*25 + [1]*25,
                                  gradient=sorted_x[:, 1],
                                  title='test tsne scatter', 
                                  xlab='tsne1', ylab='tsne2',
                                  figsize=(10, 10), n_txt_per_cluster=3, 
                                  alpha=0.5, s=50, random_state=123)
        return fig

    @pytest.mark.mpl_image_compare
    def test_cluster_scatter_gradient_legends_nolab(self):
        sorted_x = self.x_50x2[np.argsort(self.x_50x2[:, 1])]
        fig = eda.cluster_scatter(sorted_x,
                                  labels=None,
                                  gradient=sorted_x[:, 1],
                                  title='test tsne scatter', 
                                  xlab='tsne1', ylab='tsne2',
                                  figsize=(10, 10), n_txt_per_cluster=3, 
                                  alpha=0.5, s=50, random_state=123)
        return fig

    @pytest.mark.mpl_image_compare
    def test_cluster_scatter_gradient_nolegend_nolab(self):
        sorted_x = self.x_50x2[np.argsort(self.x_50x2[:, 1])]
        fig = eda.cluster_scatter(sorted_x,
                                  labels=None, add_legend=False,
                                  gradient=sorted_x[:, 1],
                                  title='test tsne scatter', 
                                  xlab='tsne1', ylab='tsne2',
                                  figsize=(10, 10), n_txt_per_cluster=3, 
                                  alpha=0.5, s=50, random_state=123)
        return fig

    @pytest.mark.mpl_image_compare
    def test_cluster_scatter_gradient_nolegend(self):
        sorted_x = self.x_50x2[np.argsort(self.x_50x2[:, 1])]
        fig = eda.cluster_scatter(sorted_x,
                                  labels=[0]*25 + [1]*25,
                                  gradient=sorted_x[:, 1],
                                  add_legend=False,
                                  title='test tsne scatter', 
                                  xlab='tsne1', ylab='tsne2',
                                  figsize=(10, 10), n_txt_per_cluster=3, 
                                  alpha=0.5, s=50, random_state=123)
        return fig

    @pytest.mark.mpl_image_compare
    def test_cluster_scatter_legends(self):
        fig = eda.cluster_scatter(self.x_50x2,
                                  [0]*25 + [1]*25,
                                  title='test tsne scatter', xlab='tsne1', ylab='tsne2',
                                  figsize=(10, 10), n_txt_per_cluster=3, alpha=0.5,
                                  s=50, random_state=123)
        return fig

    @pytest.mark.mpl_image_compare
    def test_cluster_scatter_no_legends(self):
        fig = eda.cluster_scatter(self.x_50x2,
                                  [0]*25 + [1]*25,
                                  title='test tsne scatter', xlab='tsne1', ylab='tsne2',
                                  figsize=(10, 10), add_legend=False,
                                  n_txt_per_cluster=3, alpha=0.5,
                                  s=50, random_state=123)
        return fig

    @pytest.mark.mpl_image_compare
    def test_cluster_scatter_no_labels(self):
        fig = eda.cluster_scatter(self.x_50x2,
                                  title='test tsne scatter', xlab='tsne1', ylab='tsne2',
                                  figsize=(10, 10), n_txt_per_cluster=3, alpha=0.5,
                                  s=50, random_state=123)
        return fig

    def test_cluster_scatter_wrong_tsne_shape(self):
        with pytest.raises(ValueError) as excinfo:
            eda.cluster_scatter(np.random.ranf(100).reshape(-1, 1),
                                title='test tsne scatter', xlab='tsne1', ylab='tsne2',
                                figsize=(10, 10), n_txt_per_cluster=3, alpha=0.5,
                                s=50, random_state=123)

        with pytest.raises(ValueError) as excinfo:
            eda.cluster_scatter(np.random.ranf(100).reshape(-1, 5),
                                title='test tsne scatter', xlab='tsne1', ylab='tsne2',
                                figsize=(10, 10), n_txt_per_cluster=3, alpha=0.5,
                                s=50, random_state=123)

        with pytest.raises(ValueError) as excinfo:
            eda.cluster_scatter(np.random.ranf(99).reshape(-1, 3),
                                title='test tsne scatter', xlab='tsne1', ylab='tsne2',
                                figsize=(10, 10), n_txt_per_cluster=3, alpha=0.5,
                                s=50, random_state=123)

    def test_cluster_scatter_wrong_label_shape(self):
        with pytest.raises(ValueError) as excinfo:
            eda.cluster_scatter(self.x_50x2,
                                [0] * 60,
                                title='test tsne scatter', xlab='tsne1', ylab='tsne2',
                                figsize=(10, 10), n_txt_per_cluster=3, alpha=0.5,
                                s=50, random_state=123)


class TestHeatmap(object):
    """docstring for TestHeatmap"""
    np.random.seed(123)
    x_10x5 = np.random.ranf(50).reshape(10, 5)

    @pytest.mark.mpl_image_compare
    def test_heatmap_crlabs(self):
        fig = eda.heatmap(self.x_10x5,
                          [2]*1 + [1]*3 + [5]*6,
                          [0]*3 + [1]*2,
                          title='test heatmap',
                          xlab='col label', ylab='row label',
                          figsize=(10, 10))
        return fig

    @pytest.mark.mpl_image_compare
    def test_heatmap_bilinear_interpolation(self):
        fig = eda.heatmap(self.x_10x5,
                          [2]*1 + [1]*3 + [5]*6,
                          [0]*3 + [1]*2,
                          title='test heatmap',
                          xlab='col label', ylab='row label',
                          interpolation='bilinear',
                          figsize=(10, 10))
        return fig

    @pytest.mark.mpl_image_compare
    def test_heatmap_no_xylab_title(self):
        fig = eda.heatmap(self.x_10x5,
                          [2]*1 + [1]*3 + [5]*6,
                          [0]*3 + [1]*2,
                          figsize=(10, 10))
        return fig

    @pytest.mark.mpl_image_compare
    def test_heatmap_str_crlabs(self):
        fig = eda.heatmap(self.x_10x5,
                          ['cc']*1 + ['bb']*3 + ['aa']*6,
                          ['a']*3 + ['b']*2,
                          title='test heatmap',
                          xlab='col label', ylab='row label',
                          figsize=(10, 10))
        return fig

    @pytest.mark.mpl_image_compare
    def test_heatmap_rlabs(self):
        fig = eda.heatmap(self.x_10x5,
                          row_labels=[0]*1 + [1]*3 + [2]*6,
                          title='test heatmap',
                          xlab='col label', ylab='row label',
                          figsize=(10, 10))
        return fig

    @pytest.mark.mpl_image_compare
    def test_heatmap_clabs(self):
        fig = eda.heatmap(self.x_10x5,
                          col_labels=['a']*3 + ['b']*2,
                          title='test heatmap',
                          xlab='col label', ylab='row label',
                          figsize=(10, 10))
        return fig

    @pytest.mark.mpl_image_compare
    def test_heatmap_nolabs(self):
        fig = eda.heatmap(self.x_10x5,
                          title='test heatmap',
                          xlab='col label', ylab='row label',
                          figsize=(10, 10))
        return fig

    def test_heatmap_wrong_x_shape(self):
        with pytest.raises(ValueError) as excinfo:
            eda.heatmap(np.random.ranf(1),
                        col_labels=[0],
                        title='test heatmap',
                        xlab='col label', ylab='row label',
                        figsize=(10, 10))
        with pytest.raises(ValueError) as excinfo:
            eda.heatmap(np.random.ranf(1),
                        title='test heatmap',
                        xlab='col label', ylab='row label',
                        figsize=(10, 10))

    def test_heatmap_empty_x(self):
        with pytest.raises(ValueError) as excinfo:
            eda.heatmap([[]],
                        col_labels=[],
                        title='test heatmap',
                        xlab='col label', ylab='row label',
                        figsize=(10, 10))
        with pytest.raises(ValueError) as excinfo:
            eda.heatmap([[]],
                        col_labels=[],
                        title='test heatmap',
                        xlab='col label', ylab='row label',
                        figsize=(10, 10))
        with pytest.raises(ValueError) as excinfo:
            eda.heatmap([[]],
                        row_labels=[], col_labels=[],
                        title='test heatmap',
                        xlab='col label', ylab='row label',
                        figsize=(10, 10))

    def test_heatmap_wrong_row_lab_len(self):
        with pytest.raises(ValueError) as excinfo:
            eda.heatmap(self.x_10x5,
                        ['cc']*1 + ['bb']*3 + ['aa']*5,
                        ['a']*3 + ['b']*2,
                        title='test heatmap',
                        xlab='col label', ylab='row label',
                        figsize=(10, 10))

        with pytest.raises(ValueError) as excinfo:
            eda.heatmap(self.x_10x5,
                        ['cc']*1 + ['bb']*3 + ['aa']*7,
                        ['a']*3 + ['b']*2,
                        title='test heatmap',
                        xlab='col label', ylab='row label',
                        figsize=(10, 10))

    def test_heatmap_wrong_col_lab_len(self):
        with pytest.raises(ValueError) as excinfo:
            eda.heatmap(self.x_10x5,
                        ['cc']*1 + ['bb']*3 + ['aa']*6,
                        ['a']*3 + ['b']*1,
                        title='test heatmap',
                        xlab='col label', ylab='row label',
                        figsize=(10, 10))

        with pytest.raises(ValueError) as excinfo:
            eda.heatmap(self.x_10x5,
                        ['cc']*1 + ['bb']*3 + ['aa']*6,
                        ['a']*5 + ['b']*1,
                        title='test heatmap',
                        xlab='col label', ylab='row label',
                        figsize=(10, 10))
