import numpy as np
import seaborn as sns
import scedar.eda as eda
import matplotlib as mpl
mpl.use("agg", warn=False)  # noqa
import matplotlib.pyplot as plt
import pytest


class TestSingleLabelClassifiedSamples(object):
    """docstring for TestSingleLabelClassifiedSamples"""
    np.random.seed(123)
    sfm3x3_arr = np.arange(9, dtype='float64').reshape(3, 3)
    sfm_2x0 = np.array([[], []])
    sfm5x10_arr = np.random.ranf(50).reshape(5, 10)
    sfm5x10_lst = list(map(list, np.random.ranf(50).reshape(5, 10)))

    def test_init_empty_labs(self):
        # wrong lab length. Although 2x0, there are 2 empty samples
        with pytest.raises(Exception) as excinfo:
            eda.SingleLabelClassifiedSamples(self.sfm_2x0, [])
        sfm1 = eda.SingleLabelClassifiedSamples(np.array([[], []]), [1, 0])
        assert sfm1._x.shape == (2, 0)
        assert sfm1._sids.shape == (2,)
        assert sfm1.labs == [1, 0]
        assert sfm1._fids.shape == (0,)
        # wrong x dim
        with pytest.raises(Exception) as excinfo:
            eda.SingleLabelClassifiedSamples(np.empty(0), [])
        # ok
        sfm2 = eda.SingleLabelClassifiedSamples(np.empty((0, 0)), [])
        assert sfm2._x.shape == (0, 0)
        assert sfm2._sids.shape == (0,)
        assert sfm2.labs == []
        assert sfm2._fids.shape == (0,)

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

    def test_sort_by_labels(self):
        x = np.array([[0, 0], [1, 1],
                      [100, 100], [150, 150], [125, 125],
                      [6, 6], [10, 10], [8, 8]])
        slcs = eda.SingleLabelClassifiedSamples(
            x, [0, 0, 1, 1, 1, 2, 2, 2], metric='euclidean')
        slcs_ls = slcs.sort_by_labels()
        assert slcs_ls.labs == [0, 0, 2, 2, 2, 1, 1, 1]
        assert slcs_ls.fids == list(range(2))
        assert slcs_ls.sids == [0, 1, 5, 7, 6, 2, 4, 3]

    def test_sort_by_labels_empty_mat(self):
        sfm = eda.SingleLabelClassifiedSamples(np.array([[], []]), [1, 0])

        sfm1 = sfm.sort_by_labels()
        assert sfm1._x.shape == (2, 0)
        assert sfm1._sids.shape == (2,)
        assert sfm1.labs == [1, 0]
        assert sfm1._fids.shape == (0,)

        sfm2 = eda.SingleLabelClassifiedSamples(np.empty((0, 0)),
                                                []).sort_by_labels()
        assert sfm2._x.shape == (0, 0)
        assert sfm2._sids.shape == (0,)
        assert sfm2.labs == []
        assert sfm2._fids.shape == (0,)

    def test_lab_sorted_sids(self):
        qsids = [0, 1, 5, 3, 2, 4]
        qlabs = [0, 0, 2, 1, 1, 1]
        rsids = [3, 4, 2, 5, 1, 0]
        slab_csamples = eda.SingleLabelClassifiedSamples(
            np.random.ranf(60).reshape(6, -1), qlabs, qsids)
        rs_qsids, rs_qlabs = slab_csamples.lab_sorted_sids(rsids)
        np.testing.assert_equal(rs_qsids, np.array([3, 4, 2, 5, 1, 0]))
        np.testing.assert_equal(rs_qlabs, np.array([1, 1, 1, 2, 0, 0]))

        rs_qsids, rs_qlabs = slab_csamples.lab_sorted_sids()
        np.testing.assert_equal(rs_qsids, np.array([0, 1, 3, 2, 4, 5]))
        np.testing.assert_equal(rs_qlabs, np.array([0, 0, 1, 1, 1, 2]))

    def test_filter_min_class_n(self):
        sids = [0, 1, 2, 3, 4, 5]
        labs = [0, 0, 0, 1, 2, 2]
        slab_csamples = eda.SingleLabelClassifiedSamples(
            np.random.ranf(60).reshape(6, -1), labs, sids, None)
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
        s_inds = np.array([0, 1, 2, 4, 5])
        np.testing.assert_equal(mcnf_slab_csamples._d,
                                slab_csamples._d[s_inds][:, s_inds])

    def test_ind_x(self):
        sids = list('abcdef')
        fids = list(range(10, 20))
        labs = [0, 0, 0, 1, 2, 2]

        slcs = eda.SingleLabelClassifiedSamples(
            np.random.ranf(60).reshape(6, -1), labs=labs,
            sids=sids, fids=fids)
        # select sf
        ss_slcs = slcs.ind_x([0, 5], list(range(9)))
        assert ss_slcs._x.shape == (2, 9)
        assert ss_slcs.sids == ['a', 'f']
        assert ss_slcs.labs == [0, 2]
        assert ss_slcs.fids == list(range(10, 19))
        np.testing.assert_equal(
            ss_slcs.d, slcs._d[np.ix_((0, 5), (0, 5))])

        # select with Default
        ss_slcs = slcs.ind_x()
        assert ss_slcs._x.shape == (6, 10)
        assert ss_slcs.sids == list('abcdef')
        assert ss_slcs.labs == labs
        assert ss_slcs.fids == list(range(10, 20))
        np.testing.assert_equal(ss_slcs.d, slcs._d)

        # select with None
        ss_slcs = slcs.ind_x(None, None)
        assert ss_slcs._x.shape == (6, 10)
        assert ss_slcs.sids == list('abcdef')
        assert ss_slcs.labs == labs
        assert ss_slcs.fids == list(range(10, 20))
        np.testing.assert_equal(ss_slcs.d, slcs._d)

        # select non-existent inds
        with pytest.raises(IndexError) as excinfo:
            slcs.ind_x([6])

        with pytest.raises(IndexError) as excinfo:
            slcs.ind_x(None, ['a'])

    def test_ind_x_empty(self):
        sids = list('abcdef')
        fids = list(range(10, 20))
        labs = [0, 0, 0, 1, 2, 2]

        slcs = eda.SingleLabelClassifiedSamples(
            np.random.ranf(60).reshape(6, -1), labs=labs,
            sids=sids, fids=fids)
        empty_s = slcs.ind_x([])
        assert empty_s._x.shape == (0, 10)
        assert empty_s._d.shape == (0, 0)
        assert empty_s._sids.shape == (0,)
        assert empty_s._fids.shape == (10,)
        assert empty_s._labs.shape == (0,)

        empty_f = slcs.ind_x(None, [])
        assert empty_f._x.shape == (6, 0)
        assert empty_f._d.shape == (6, 6)
        assert empty_f._sids.shape == (6,)
        assert empty_f._fids.shape == (0,)
        assert empty_f._labs.shape == (6,)

        empty_sf = slcs.ind_x([], [])
        assert empty_sf._x.shape == (0, 0)
        assert empty_sf._d.shape == (0, 0)
        assert empty_sf._sids.shape == (0,)
        assert empty_sf._fids.shape == (0,)
        assert empty_sf._labs.shape == (0,)

    def test_relabel(self):
        sids = list('abcdef')
        fids = list(range(10, 20))
        labs = [0, 0, 0, 1, 2, 2]

        slcs = eda.SingleLabelClassifiedSamples(
            np.random.ranf(60).reshape(6, -1), labs=labs,
            sids=sids, fids=fids)

        new_labs = ['a', 'b', 'c', 'd', 'e', 'f']
        slcs_rl = slcs.relabel(new_labs)
        assert slcs_rl.labs == new_labs
        assert slcs_rl._x is not slcs._x
        assert slcs_rl._d is not slcs._d
        assert slcs_rl._sids is not slcs._sids
        assert slcs_rl._fids is not slcs._fids
        np.testing.assert_equal(slcs_rl._x, slcs._x)
        np.testing.assert_equal(slcs_rl._d, slcs._d)
        np.testing.assert_equal(slcs_rl._sids, slcs._sids)
        np.testing.assert_equal(slcs_rl._fids, slcs._fids)

    def test_merge_labels(self):
        sids = list('abcdef')
        fids = list(range(10, 20))
        labs = [0, 0, 1, 1, 2, 3]

        slcs = eda.SingleLabelClassifiedSamples(
            np.random.ranf(60).reshape(6, -1), labs=labs,
            sids=sids, fids=fids)

        slcs.merge_labels([1, 2, 3], 5)
        new_labs = [0, 0, 5, 5, 5, 5]
        assert slcs.labs == new_labs
        assert slcs.sids == sids
        assert slcs.fids == fids
        assert slcs.labs_to_sids([5]) == (('c', 'd', 'e', 'f'),)
        assert slcs.sids_to_labs(sids).tolist() == new_labs
        assert slcs._uniq_labs.tolist() == [0, 5]
        assert slcs._uniq_lab_cnts.tolist() == [2, 4]

    def test_merge_labels_wrong_args(self):
        sids = list('abcdef')
        fids = list(range(10, 20))
        labs = [0, 0, 1, 1, 2, 3]

        slcs = eda.SingleLabelClassifiedSamples(
            np.random.ranf(60).reshape(6, -1), labs=labs,
            sids=sids, fids=fids)
        # wrong new lab type
        with pytest.raises(ValueError) as excinfo:
            slcs.merge_labels([1, 2, 3], [5])
        # wrong m lab type
        with pytest.raises(ValueError) as excinfo:
            slcs.merge_labels([[], [1]], 1)
        # duplicated m labs
        with pytest.raises(ValueError) as excinfo:
            slcs.merge_labels([1, 1, 2], 1)
        # m lab not in original lab
        with pytest.raises(ValueError) as excinfo:
            slcs.merge_labels([0, 1, 5], 1)

    def test_id_x(self):
        sids = list('abcdef')
        fids = list(range(10, 20))
        labs = [0, 0, 0, 1, 2, 2]

        slcs = eda.SingleLabelClassifiedSamples(
            np.random.ranf(60).reshape(6, -1), labs=labs,
            sids=sids, fids=fids)
        # select sf
        ss_slcs = slcs.id_x(['a', 'f'], list(range(10, 15)))
        assert ss_slcs._x.shape == (2, 5)
        assert ss_slcs.sids == ['a', 'f']
        assert ss_slcs.labs == [0, 2]
        assert ss_slcs.fids == list(range(10, 15))
        np.testing.assert_equal(
            ss_slcs.d, slcs._d[np.ix_((0, 5), (0, 5))])

        # select with Default
        ss_slcs = slcs.id_x()
        assert ss_slcs._x.shape == (6, 10)
        assert ss_slcs.sids == list('abcdef')
        assert ss_slcs.labs == labs
        assert ss_slcs.fids == list(range(10, 20))
        np.testing.assert_equal(ss_slcs.d, slcs._d)

        # select with None
        ss_slcs = slcs.id_x(None, None)
        assert ss_slcs._x.shape == (6, 10)
        assert ss_slcs.sids == list('abcdef')
        assert ss_slcs.labs == labs
        assert ss_slcs.fids == list(range(10, 20))
        np.testing.assert_equal(ss_slcs.d, slcs._d)

        # select non-existent inds
        # id lookup raises ValueError
        with pytest.raises(ValueError) as excinfo:
            slcs.id_x([6])

        with pytest.raises(ValueError) as excinfo:
            slcs.id_x(None, ['a'])

    def test_id_x_empty(self):
        sids = list('abcdef')
        fids = list(range(10, 20))
        labs = [0, 0, 0, 1, 2, 2]

        slcs = eda.SingleLabelClassifiedSamples(
            np.random.ranf(60).reshape(6, -1), labs=labs,
            sids=sids, fids=fids)
        empty_s = slcs.id_x([])
        assert empty_s._x.shape == (0, 10)
        assert empty_s._d.shape == (0, 0)
        assert empty_s._sids.shape == (0,)
        assert empty_s._fids.shape == (10,)
        assert empty_s._labs.shape == (0,)

        empty_f = slcs.id_x(None, [])
        assert empty_f._x.shape == (6, 0)
        assert empty_f._d.shape == (6, 6)
        assert empty_f._sids.shape == (6,)
        assert empty_f._fids.shape == (0,)
        assert empty_f._labs.shape == (6,)

        empty_sf = slcs.id_x([], [])
        assert empty_sf._x.shape == (0, 0)
        assert empty_sf._d.shape == (0, 0)
        assert empty_sf._sids.shape == (0,)
        assert empty_sf._fids.shape == (0,)
        assert empty_sf._labs.shape == (0,)

    def test_lab_x(self):
        sids = list('abcdef')
        fids = list(range(10, 20))
        labs = [0, 0, 0, 1, 2, 2]

        slcs = eda.SingleLabelClassifiedSamples(
            np.random.ranf(60).reshape(6, -1), labs=labs,
            sids=sids, fids=fids)
        # select sf
        ss_slcs = slcs.lab_x([0, 2])
        assert ss_slcs._x.shape == (5, 10)
        assert ss_slcs.sids == ['a', 'b', 'c', 'e', 'f']
        assert ss_slcs.labs == [0, 0, 0, 2, 2]
        assert ss_slcs.fids == list(range(10, 20))
        ss_s_inds = [0, 1, 2, 4, 5]
        np.testing.assert_equal(ss_slcs.d,
                                slcs._d[np.ix_(ss_s_inds, ss_s_inds)])
        # select sf
        ss_slcs = slcs.lab_x(0)
        assert ss_slcs._x.shape == (3, 10)
        assert ss_slcs.sids == ['a', 'b', 'c']
        assert ss_slcs.labs == [0, 0, 0]
        assert ss_slcs.fids == list(range(10, 20))
        ss_s_inds = [0, 1, 2]
        np.testing.assert_equal(ss_slcs.d,
                                slcs._d[np.ix_(ss_s_inds, ss_s_inds)])
        # select with None
        slcs_n = slcs.lab_x(None)
        np.testing.assert_equal(slcs_n._x, slcs._x)
        np.testing.assert_equal(slcs_n._d, slcs._d)
        np.testing.assert_equal(slcs_n._sids, slcs._sids)
        np.testing.assert_equal(slcs_n._fids, slcs._fids)
        np.testing.assert_equal(slcs_n._labs, slcs._labs)
        # select non-existent labs
        with pytest.raises(ValueError) as excinfo:
            slcs.lab_x([-1])
        with pytest.raises(ValueError) as excinfo:
            slcs.lab_x([0, 3])
        with pytest.raises(ValueError) as excinfo:
            slcs.lab_x([0, -3])

    def test_lab_x_empty(self):
        sids = list('abcdef')
        fids = list(range(10, 20))
        labs = [0, 0, 0, 1, 2, 2]

        slcs = eda.SingleLabelClassifiedSamples(
            np.random.ranf(60).reshape(6, -1), labs=labs,
            sids=sids, fids=fids)
        # select sf
        empty_s = slcs.lab_x([])
        assert empty_s._x.shape == (0, 10)
        assert empty_s._d.shape == (0, 0)
        assert empty_s._sids.shape == (0,)
        assert empty_s._labs.shape == (0,)
        assert empty_s._fids.shape == (10,)
        assert empty_s._labs.shape == (0,)

    def test_feature_importance_across_labs(self):
        # Generate simple dataset with gaussian noise
        x_centers = np.array([[0, 0,   1,  1, 5, 50, 10, 37],
                              [0, 0, 1.5,  5, 5, 50, 10, 35],
                              [0, 0,  10, 10, 5, 50, 10, 33]])
        np.random.seed(1920)
        c1x = np.array(x_centers[0]) + np.random.normal(size=(500, 8))
        c2x = np.array(x_centers[1]) + np.random.normal(size=(200, 8))
        c3x = np.array(x_centers[2]) + np.random.normal(size=(300, 8))
        x = np.vstack((c1x, c2x, c3x))
        labs = [0] * 500 + [1] * 200 + [2] * 300
        slcs = eda.SingleLabelClassifiedSamples(x, labs=labs)
        # binary logistic regression
        f_importance_list, bst = slcs.feature_importance_across_labs(
            [0, 1], silent=0)
        assert f_importance_list[0][0] == 3
        # multi class softmax
        f_importance_list2, bst2 = slcs.feature_importance_across_labs(
            [0, 1, 2], random_state=123, silent=1)
        assert f_importance_list2[0][0] == 3
        assert f_importance_list2 != f_importance_list
        # multiclass with provided parames
        xgb_params = {
            'eta': 0.3,
            'max_depth': 6,
            'silent': 0,
            'nthread': 1,
            'alpha': 1,
            'lambda': 0,
            'seed': 0,
            'objective': 'multi:softmax',
            'eval_metric': 'merror',
            'num_class': 3
        }
        f_importance_list3, bst3 = slcs.feature_importance_across_labs(
            [0, 1, 2], random_state=123, xgb_params=xgb_params)
        assert f_importance_list3 == f_importance_list2
        # shuffle features
        f_importance_list4, bst4 = slcs.feature_importance_across_labs(
            [0, 1], random_state=123, shuffle_features=True)
        assert f_importance_list2[0][0] == 3
        # bootstrapping
        f_importance_list5, bst5 = slcs.feature_importance_across_labs(
            [0, 1], random_state=123, shuffle_features=True,
            num_bootstrap_round=10)
        f_importance_list6, bst6 = slcs.feature_importance_across_labs(
            [0, 1], random_state=123, shuffle_features=True,
            num_bootstrap_round=10)
        assert f_importance_list5 == f_importance_list6
        assert f_importance_list5[0][0] == 3

    def test_feature_importance_across_labs_bootstrap(self):
        # Generate simple dataset with gaussian noise
        x_centers = np.array([[0, 0,   1,  1, 5, 50, 10, 37],
                              [0, 0, 1.5,  5, 5, 50, 10, 35],
                              [0, 0,  10, 10, 5, 50, 10, 33]])
        np.random.seed(1920)
        c1x = np.array(x_centers[0]) + np.random.normal(size=(50, 8))
        c2x = np.array(x_centers[1]) + np.random.normal(size=(20, 8))
        c3x = np.array(x_centers[2]) + np.random.normal(size=(30, 8))
        x = np.vstack((c1x, c2x, c3x))
        labs = [0] * 50 + [1] * 20 + [2] * 30
        slcs = eda.SingleLabelClassifiedSamples(x, labs=labs)
        # bootstrapping
        f_importance_list, bst = slcs.feature_importance_across_labs(
            [0, 1], random_state=123, shuffle_features=True,
            num_bootstrap_round=10)
        f_importance_list2, bst2 = slcs.feature_importance_across_labs(
            [0, 1], random_state=123, shuffle_features=True,
            num_bootstrap_round=10)
        assert f_importance_list == f_importance_list2
        assert f_importance_list2[0][0] == 3
        # no feature shuffling
        f_importance_list3, bst3 = slcs.feature_importance_across_labs(
            [0, 1], random_state=123, shuffle_features=False,
            num_bootstrap_round=10)
        # provide resampling size
        f_importance_list4, bst4 = slcs.feature_importance_across_labs(
            [0, 1], random_state=123, shuffle_features=False,
            bootstrap_size=30, num_bootstrap_round=10)
        f_importance_list4, bst4 = slcs.feature_importance_across_labs(
            [0, 1], random_state=123, shuffle_features=True,
            bootstrap_size=30, num_bootstrap_round=10)

    # resampling procedure will be repeated until all unique labels exist
    # in the resample.
    def test_feature_importance_across_labs_bootstrap_resample(self):
        x_centers = np.array([[0, 0,   1,  1, 5, 50, 10, 37],
                              [0, 0, 1.5,  5, 5, 50, 10, 35],
                              [0, 0,  10, 10, 5, 50, 10, 33]])
        np.random.seed(1920)
        c1x = np.array(x_centers[0]) + np.random.normal(size=(500, 8))
        c2x = np.array(x_centers[1]) + np.random.normal(size=(1, 8))
        c3x = np.array(x_centers[2]) + np.random.normal(size=(30, 8))
        x = np.vstack((c1x, c2x, c3x))
        labs = [0] * 500 + [1] * 1 + [2] * 30
        slcs = eda.SingleLabelClassifiedSamples(x, labs=labs)
        # bootstrapping
        f_importance_list, bst = slcs.feature_importance_across_labs(
            [0, 1], random_state=123, shuffle_features=True,
            num_bootstrap_round=10)

    def test_feature_importance_across_labs_wrong_args(self):
        # Generate simple dataset with gaussian noise
        x_centers = np.array([[0, 0,   1,  1, 5, 50, 10, 37],
                              [0, 0, 1.5,  5, 5, 50, 10, 35],
                              [0, 0,  10, 10, 5, 50, 10, 33]])
        np.random.seed(1920)
        c1x = np.array(x_centers[0]) + np.random.normal(size=(50, 8))
        c2x = np.array(x_centers[1]) + np.random.normal(size=(20, 8))
        c3x = np.array(x_centers[2]) + np.random.normal(size=(30, 8))
        x = np.vstack((c1x, c2x, c3x))
        labs = [0] * 50 + [1] * 20 + [2] * 30
        slcs = eda.SingleLabelClassifiedSamples(x, labs=labs)
        with pytest.raises(ValueError) as excinfo:
            slcs.feature_importance_across_labs([0, 3])

        with pytest.raises(ValueError) as excinfo:
            slcs.feature_importance_across_labs([0, 3], num_boost_round=0)

        with pytest.raises(ValueError) as excinfo:
            slcs.feature_importance_across_labs([0, 3], num_boost_round=0.5)

        with pytest.raises(ValueError) as excinfo:
            slcs.feature_importance_across_labs([0, 3], num_boost_round=-1)

        with pytest.raises(ValueError) as excinfo:
            slcs.feature_importance_across_labs([-1])

        with pytest.raises(ValueError) as excinfo:
            slcs.feature_importance_across_labs([3, 5])
        # meaningless to run this on empty matrix
        with pytest.raises(ValueError) as excinfo:
            slcs.feature_importance_across_labs([])

    def test_feature_importance_distintuishing_labs(self):
        # Generate simple dataset with gaussian noise
        x_centers = np.array([[0, 0,   1,  1, 5, 50, 10, 37],
                              [0, 0, 1.5,  5, 5, 50, 10, 35],
                              [0, 0,  10, 10, 5, 50, 10, 33]])
        np.random.seed(1920)
        c1x = np.array(x_centers[0]) + np.random.normal(size=(500, 8))
        c2x = np.array(x_centers[1]) + np.random.normal(size=(200, 8))
        c3x = np.array(x_centers[2]) + np.random.normal(size=(300, 8))
        x = np.vstack((c1x, c2x, c3x))
        labs = [0] * 500 + [1] * 200 + [2] * 300
        slcs = eda.SingleLabelClassifiedSamples(x, labs=labs)
        # binary logistic regression
        f_importance_list, bst = slcs.feature_importance_distintuishing_labs(
            [0, 1], silent=0)
        assert f_importance_list[0][0] == 2

    def test_feature_importance_each_lab(self):
        # Generate simple dataset with gaussian noise
        x_centers = np.array([[0, 0,   1,  1, 5, 50, 10, 37],
                              [0, 0, 1.5,  5, 5, 50, 10, 35],
                              [0, 0,  10, 10, 5, 50, 10, 33]])
        np.random.seed(1920)
        c1x = np.array(x_centers[0]) + np.random.normal(size=(500, 8))
        c2x = np.array(x_centers[1]) + np.random.normal(size=(200, 8))
        c3x = np.array(x_centers[2]) + np.random.normal(size=(300, 8))
        x = np.vstack((c1x, c2x, c3x))
        labs = [0] * 500 + [1] * 200 + [2] * 300
        slcs = eda.SingleLabelClassifiedSamples(x, labs=labs)
        # binary logistic regression
        ulab_fi_lut = slcs.feature_importance_each_lab()
        assert ulab_fi_lut[0][-1][0] == 3
        print(ulab_fi_lut)
        assert ulab_fi_lut[1][-1][0] == 2

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
    def test_tsne_feature_gradient_plot_abclabs(self):
        sids = list(range(8))
        fids = [str(i) for i in range(10)]
        labs = list(range(8))
        np.random.seed(123)
        x = np.random.ranf(80).reshape(8, -1)
        x_sorted = x[np.argsort(x[:, 5])]
        g = x_sorted[:, 5]
        slab_csamples = eda.SingleLabelClassifiedSamples(
            x_sorted, labs, sids=sids, fids=fids)
        return slab_csamples.tsne_feature_gradient_plot(
            '5', labels=list('abcdefgh'), figsize=(10, 10), s=50)

    # select specific labels to plot gradient
    @pytest.mark.mpl_image_compare
    def test_tsne_feature_gradient_plot_sslabs(self):
        sids = list(range(8))
        fids = [str(i) for i in range(10)]
        labs = list(range(8))
        np.random.seed(123)
        x = np.random.ranf(80).reshape(8, -1)
        x_sorted = x[np.argsort(x[:, 5])]
        g = x_sorted[:, 5]
        slab_csamples = eda.SingleLabelClassifiedSamples(
            x_sorted, labs, sids=sids, fids=fids)
        return slab_csamples.tsne_feature_gradient_plot(
            '5', selected_labels=[5, 6, 7], figsize=(10, 10), s=50)

    @pytest.mark.mpl_image_compare
    def test_tsne_plot(self):
        sids = list(range(8))
        fids = [str(i) for i in range(10)]
        labs = list(range(8))
        np.random.seed(123)
        x = np.random.ranf(80).reshape(8, -1)
        x_sorted = x[np.argsort(x[:, 5])]
        g = x_sorted[:, 5]
        slab_csamples = eda.SingleLabelClassifiedSamples(
            x_sorted, labs, sids=sids, fids=fids)
        return slab_csamples.tsne_plot(g, figsize=(10, 10), s=50)

    @pytest.mark.mpl_image_compare
    def test_tsne_plot_abclabs(self):
        sids = list(range(8))
        fids = [str(i) for i in range(10)]
        labs = list(range(8))
        np.random.seed(123)
        np.random.seed(123)
        x = np.random.ranf(80).reshape(8, -1)
        x_sorted = x[np.argsort(x[:, 5])]
        g = x_sorted[:, 5]
        slab_csamples = eda.SingleLabelClassifiedSamples(
            x_sorted, labs, sids=sids, fids=fids)
        return slab_csamples.tsne_plot(g, labels=list('abcdefgh'),
                                       figsize=(10, 10), s=50)

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

    @pytest.mark.mpl_image_compare
    def test_swarm_a(self):
        # array([[0, 1],
        #        [2, 3],
        #        [4, 5],
        #        [6, 7],
        #        [8, 9]])
        tslcs = eda.SingleLabelClassifiedSamples(np.arange(10).reshape(5, 2),
                                                 [0, 0, 1, 2, 3],
                                                 ['1', '2', '3', '4', '5'],
                                                 ['a', 'z'])
        return tslcs.feature_swarm_plot('a', transform=lambda x: x + 200,
                                        selected_labels=[0, 2, 3],
                                        title='test', xlab='x', ylab='y')

    @pytest.mark.mpl_image_compare
    def test_dmat_heatmap(self):
        x = [[0, 0], [1, 1], [2, 2], [10, 10], [12, 12], [11, 11], [100, 100]]
        tslcs = eda.SingleLabelClassifiedSamples(x, [0, 0, 0, 1, 1, 1, 2],
                                                 metric='euclidean')
        return tslcs.dmat_heatmap(selected_labels=[0, 1],
                                  transform=lambda x: x + 100)

    @pytest.mark.mpl_image_compare
    def test_xmat_heatmap(self):
        x = [[0, 0], [1, 1], [2, 2], [10, 10], [12, 12], [11, 11], [100, 100]]
        tslcs = eda.SingleLabelClassifiedSamples(x, [0, 0, 0, 1, 1, 1, 2],
                                                 metric='euclidean')
        return tslcs.xmat_heatmap(selected_labels=[0, 1],
                                  selected_fids=[1, 0],
                                  col_labels=['spec1', 'spec2'],
                                  transform=lambda x: x + 200)

    @pytest.mark.mpl_image_compare
    def test_swarm_minimal_z(self):
        tslcs = eda.SingleLabelClassifiedSamples(np.arange(10).reshape(5, 2),
                                                 [0, 0, 1, 2, 3],
                                                 ['1', '2', '3', '4', '5'],
                                                 ['a', 'z'])
        return tslcs.feature_swarm_plot('z')

    def test_swarm_wrong_args(self):
        tslcs = eda.SingleLabelClassifiedSamples(np.arange(10).reshape(5, 2),
                                                 [0, 0, 1, 2, 3],
                                                 ['1', '2', '3', '4', '5'],
                                                 ['a', 'z'])
        # non-callable transform
        with pytest.raises(ValueError) as excinfo:
            tslcs.feature_swarm_plot('z', transform=1)
        # wrong label size
        with pytest.raises(ValueError) as excinfo:
            tslcs.feature_swarm_plot('z', labels=[0, 2, 1])

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


class TestMDLSingleLabelClassifiedSamples(object):
    """docstring for TestMDLSingleLabelClassifiedSamples"""
    np.random.seed(5009)
    x50x5 = np.vstack((np.zeros((30, 5)), np.random.ranf((20, 5))))
    labs50 = [0]*10 + [1]*35 + [2]*5

    def test_mdl_computation(self):
        mdl_slcs = eda.MDLSingleLabelClassifiedSamples(
            self.x50x5, labs=self.labs50, metric="euclidean")
        no_lab_mdl = mdl_slcs.no_lab_mdl()
        (ulab_mdl_sum, ulab_s_ind_l, ulab_cnt_l, ulab_mdl_l,
         cluster_mdl) = mdl_slcs.lab_mdl()
        assert ulab_s_ind_l == [list(range(10)), list(range(10, 45)),
                                list(range(45, 50))]
        assert ulab_mdl_sum == np.sum(ulab_mdl_l)
        ulab_cnt_l = [10, 35, 5]

        for i in range(3):
            ci_mdl = eda.MDLSingleLabelClassifiedSamples(
                self.x50x5[ulab_s_ind_l[i], :],
                labs=[self.labs50[ii] for ii in ulab_s_ind_l[i]],
                metric="euclidean")

            np.testing.assert_allclose(
                ci_mdl.no_lab_mdl(),
                ulab_mdl_l[i] - cluster_mdl * ulab_cnt_l[i] / 50)

    def test_data_mdl_computation_mp(self):
        mdl_slcs = eda.MDLSingleLabelClassifiedSamples(
            self.x50x5, labs=self.labs50, metric="euclidean")
        no_lab_mdl = mdl_slcs.no_lab_mdl(nprocs=2)
        (ulab_mdl_sum, ulab_s_ind_l, ulab_cnt_l,
         ulab_mdl_l, cluster_mdl) = mdl_slcs.lab_mdl(nprocs=2)
        assert ulab_s_ind_l == [list(range(10)), list(range(10, 45)),
                                list(range(45, 50))]
        assert ulab_mdl_sum == np.sum(ulab_mdl_l)

        ulab_cnt_l = [10, 35, 5]

        for i in range(3):
            ci_mdl = eda.MDLSingleLabelClassifiedSamples(
                self.x50x5[ulab_s_ind_l[i], :],
                labs=[self.labs50[ii] for ii in ulab_s_ind_l[i]],
                metric="euclidean")

            np.testing.assert_allclose(
                ci_mdl.no_lab_mdl(nprocs=5),
                ulab_mdl_l[i] - cluster_mdl * ulab_cnt_l[i] / 50)

    def test_distance_mdl_computation_mp(self):
        mdl_slcs = eda.MDLSingleLabelClassifiedSamples(
            self.x50x5, labs=self.labs50, encode_type="distance",
            mdl_method=eda.mdl.GKdeMdl, metric="euclidean")
        no_lab_mdl = mdl_slcs.no_lab_mdl(nprocs=2)
        (ulab_mdl_sum, ulab_s_ind_l, ulab_cnt_l,
         ulab_mdl_l, cluster_mdl) = mdl_slcs.lab_mdl(nprocs=2)
        assert ulab_s_ind_l == [list(range(10)), list(range(10, 45)),
                                list(range(45, 50))]
        assert ulab_mdl_sum == np.sum(ulab_mdl_l)

        ulab_cnt_l = [10, 35, 5]

        for i in range(3):
            ci_mdl = eda.MDLSingleLabelClassifiedSamples(
                mdl_slcs._x[ulab_s_ind_l[i]],
                labs=[self.labs50[ii] for ii in ulab_s_ind_l[i]],
                metric="euclidean", encode_type=mdl_slcs._encode_type,
                mdl_method=mdl_slcs._mdl_method)

            np.testing.assert_allclose(
                ci_mdl.no_lab_mdl(nprocs=5),
                ulab_mdl_l[i] - cluster_mdl * ulab_cnt_l[i] / 50)

    def test_mdl_method(self):
        zigk_mdl_slcs = eda.MDLSingleLabelClassifiedSamples(
            self.x50x5, labs=self.labs50, mdl_method=eda.mdl.ZeroIGKdeMdl,
            metric="euclidean")
        gk_mdl_slcs = eda.MDLSingleLabelClassifiedSamples(
            self.x50x5, labs=self.labs50, mdl_method=eda.mdl.GKdeMdl,
            metric="euclidean")
        assert gk_mdl_slcs._mdl_method is eda.mdl.GKdeMdl
        assert zigk_mdl_slcs._mdl_method is eda.mdl.ZeroIGKdeMdl
        assert zigk_mdl_slcs.no_lab_mdl() != gk_mdl_slcs.no_lab_mdl()

        for mdl_method in eda.MDL_METHODS:
            mdl_slcs = eda.MDLSingleLabelClassifiedSamples(
                self.x50x5, labs=self.labs50, mdl_method=mdl_method,
                metric="euclidean")

            (ulab_mdl_sum, ulab_s_ind_l, ulab_cnt_l,
             ulab_mdl_l, cluster_mdl) = mdl_slcs.lab_mdl()

            assert ulab_s_ind_l == [list(range(10)), list(range(10, 45)),
                                    list(range(45, 50))]
            assert ulab_mdl_sum == np.sum(ulab_mdl_l)

            ulab_cnt_l = [10, 35, 5]

            for i in range(3):
                ci_mdl = eda.MDLSingleLabelClassifiedSamples(
                    self.x50x5[ulab_s_ind_l[i], :],
                    labs=[self.labs50[ii] for ii in ulab_s_ind_l[i]],
                    mdl_method=mdl_method,
                    metric="euclidean")

                np.testing.assert_allclose(
                    ci_mdl.no_lab_mdl(),
                    ulab_mdl_l[i] - cluster_mdl * ulab_cnt_l[i] / 50)

    def test_wrong_mdl_method(self):
        with pytest.raises(ValueError) as excinfo:
            eda.MDLSingleLabelClassifiedSamples(
                self.x50x5, labs=self.labs50, mdl_method="123",
                metric="euclidean").no_lab_mdl()

        with pytest.raises(ValueError) as excinfo:
            eda.MDLSingleLabelClassifiedSamples(
                self.x50x5, labs=self.labs50, mdl_method=int,
                metric="euclidean").no_lab_mdl()

        with pytest.raises(ValueError) as excinfo:
            eda.MDLSingleLabelClassifiedSamples(
                self.x50x5, labs=self.labs50, mdl_method="ZeroIMdl",
                metric="euclidean").no_lab_mdl()

        with pytest.raises(ValueError) as excinfo:
            eda.MDLSingleLabelClassifiedSamples(
                self.x50x5, labs=self.labs50, mdl_method=2,
                metric="euclidean").no_lab_mdl()

    def test_wrong_encode_type(self):
        with pytest.raises(ValueError) as excinfo:
            eda.MDLSingleLabelClassifiedSamples(
                self.x50x5, labs=self.labs50, encode_type="123",
                metric="euclidean").no_lab_mdl()

        with pytest.raises(ValueError) as excinfo:
            eda.MDLSingleLabelClassifiedSamples(
                self.x50x5, labs=self.labs50, encode_type=1,
                metric="euclidean").no_lab_mdl()

        with pytest.raises(ValueError) as excinfo:
            eda.MDLSingleLabelClassifiedSamples(
                self.x50x5, labs=self.labs50, encode_type=None,
                metric="euclidean").no_lab_mdl()

    def test_auto_param(self):
        eda.MDLSingleLabelClassifiedSamples(
            self.x50x5, labs=self.labs50, encode_type="auto",
            mdl_method=None, metric="euclidean")
        eda.MDLSingleLabelClassifiedSamples(
            np.zeros((100, 101)), labs=[0]*100, encode_type="auto",
            mdl_method=None, metric="euclidean")
        eda.MDLSingleLabelClassifiedSamples(
            np.ones((100, 100)), labs=[0]*100, encode_type="auto",
            mdl_method=None, metric="euclidean")
        eda.MDLSingleLabelClassifiedSamples(
            [[], []], labs=[0]*2, encode_type="auto",
            mdl_method=None, metric="euclidean")

    def test_lab_mdl_ret_internal(self):
        mdl_slcs = eda.MDLSingleLabelClassifiedSamples(
            self.x50x5, labs=self.labs50, metric="euclidean")

        ((ulab_mdl_sum, ulab_s_ind_l, ulab_cnt_l, ulab_mdl_l,
         cluster_mdl), mdl_l) = mdl_slcs.lab_mdl(ret_internal=True)
        np.testing.assert_allclose(sum(mdl_l) + cluster_mdl,
                                   sum(ulab_mdl_l))

        lab_mdl_res = mdl_slcs.lab_mdl()
        ulab_mdl_sum2 = lab_mdl_res.ulab_mdl_sum
        assert ulab_mdl_sum2 == ulab_mdl_sum
        assert ulab_mdl_sum2 == np.sum(lab_mdl_res.ulab_mdls)

    def test_per_col_encoders_wrong_xshape(self):
        with pytest.raises(ValueError) as excinfo:
            eda.MDLSingleLabelClassifiedSamples.per_col_encoders(
                np.zeros(10), "data")

        with pytest.raises(ValueError) as excinfo:
            eda.MDLSingleLabelClassifiedSamples.per_col_encoders(
                np.zeros((10, 10, 10)), "data")

    def test_encode_mdl(self):
        mdl_slcs = eda.MDLSingleLabelClassifiedSamples(
            self.x50x5, labs=self.labs50, metric="euclidean")
        # wrong dimensions
        with pytest.raises(ValueError) as excinfo:
            mdl_slcs.encode(np.zeros((10, 3)))

        with pytest.raises(ValueError) as excinfo:
            mdl_slcs.encode(np.zeros(20))

        with pytest.raises(ValueError) as excinfo:
            mdl_slcs.encode(np.zeros(20), col_summary_func=1)

        with pytest.raises(ValueError) as excinfo:
            mdl_slcs.encode(np.zeros(20), col_summary_func=None)

        emdl = mdl_slcs.encode(np.arange(100).reshape(-1, 5))
        emdl2 = mdl_slcs.encode(np.arange(100).reshape(-1, 5), nprocs=2)

        assert emdl == emdl2

        emdl3 = eda.MDLSingleLabelClassifiedSamples(
            self.x50x5, mdl_method=eda.mdl.GKdeMdl, labs=self.labs50,
            metric="euclidean").encode(np.arange(100).reshape(-1, 5))
        assert emdl != emdl3

        emdl4 = eda.MDLSingleLabelClassifiedSamples(
            np.zeros((50, 5)), mdl_method=eda.mdl.GKdeMdl, labs=self.labs50,
            metric="euclidean").encode(np.arange(100).reshape(-1, 5))
        assert emdl != emdl4

        emdl5 = eda.MDLSingleLabelClassifiedSamples(
            np.zeros((50, 5)), mdl_method=eda.mdl.GKdeMdl, labs=self.labs50,
            metric="euclidean").encode(np.arange(100).reshape(-1, 5),
                                       non_zero_only=True)
        assert emdl5 != emdl3

        emdl6 = eda.MDLSingleLabelClassifiedSamples(
            self.x50x5, encode_type="distance", mdl_method=eda.mdl.GKdeMdl,
            labs=self.labs50, metric="euclidean").encode(
                np.arange(100).reshape(-1, 50), non_zero_only=True)
        assert emdl5 != emdl3
