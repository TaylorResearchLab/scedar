import numpy as np
import seaborn as sns
import scedar.eda as eda
import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest


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
            np.random.ranf(60).reshape(6, -1), labs=labs,
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
            np.random.ranf(60).reshape(6, -1), labs=labs,
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

        lab_cmap, lab_ind_arr, lab_col_lut, uniq_lab_lut = eda.plot.labs_to_cmap(
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

        lab_cmap2 = eda.plot.labs_to_cmap(
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
