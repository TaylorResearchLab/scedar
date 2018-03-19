import numpy as np
import seaborn as sns
import scxplit.eda as eda
import pytest


class TestSingleLabelClassifiedSamples(object):
    """docstring for TestSingleLabelClassifiedSamples"""
    def test_init_dup_sids(self):
        with pytest.raises(Exception) as excinfo:
            eda.SingleLabelClassifiedSamples([0, 0, 1], [0, 0, 0])
        
        with pytest.raises(Exception) as excinfo:
            eda.SingleLabelClassifiedSamples(['0', '0', '1'], [0, 0, 0])

    def test_init_empty_sids_labs(self):
        with pytest.raises(Exception) as excinfo:
            eda.SingleLabelClassifiedSamples([], [])

    def test_init_diff_size_sids_labs(self):
        with pytest.raises(Exception) as excinfo:
            eda.SingleLabelClassifiedSamples([0, 1, 2], [0, 1])

    def test_init_non1d_sids_labs(self):
        with pytest.raises(Exception) as excinfo:
            eda.SingleLabelClassifiedSamples(np.array([[0], [1], [2]]), 
                                             np.array([[0], [1], [1]]))

        with pytest.raises(Exception) as excinfo:
            eda.SingleLabelClassifiedSamples(np.array([[0], [1], [2]]), 
                                             np.array([0, 1, 2]))

        with pytest.raises(Exception) as excinfo:
            eda.SingleLabelClassifiedSamples(np.array([0, 1, 2]),
                                             np.array([[0], [1], [2]]))

    def test_init_bad_sid_type(self):
        with pytest.raises(Exception) as excinfo:
            eda.SingleLabelClassifiedSamples([False, True, 2], [0, 1, 1])

        with pytest.raises(Exception) as excinfo:
            eda.SingleLabelClassifiedSamples([[0], [0, 1], 2], [0, 1, 1])

        with pytest.raises(Exception) as excinfo:
            eda.SingleLabelClassifiedSamples(np.array([0, 1, 2]), [0, 1, 1])

        with pytest.raises(Exception) as excinfo:
            eda.SingleLabelClassifiedSamples([(0), (0, 1), 2], [0, 1, 1])

    def test_init_bad_lab_type(self):
        with pytest.raises(Exception) as excinfo:
            eda.SingleLabelClassifiedSamples([0, 1, 2], [False, True, 2])

        with pytest.raises(Exception) as excinfo:
            eda.SingleLabelClassifiedSamples([0, 1, 2], [[0], [0, 1], 2])

        with pytest.raises(Exception) as excinfo:
            eda.SingleLabelClassifiedSamples([0, 1, 2], [(0), (0, 1), 2])

        with pytest.raises(Exception) as excinfo:
            eda.SingleLabelClassifiedSamples([0, 1, 2], np.array([0, 1, 2]))

    def test_is_valid_sid(self):
        assert eda.SingleLabelClassifiedSamples.is_valid_sid('1')
        assert eda.SingleLabelClassifiedSamples.is_valid_sid(1)
        assert not eda.SingleLabelClassifiedSamples.is_valid_sid(np.array([1])[0])
        assert not eda.SingleLabelClassifiedSamples.is_valid_sid([])
        assert not eda.SingleLabelClassifiedSamples.is_valid_sid([1])
        assert not eda.SingleLabelClassifiedSamples.is_valid_sid(None)
        assert not eda.SingleLabelClassifiedSamples.is_valid_sid((1,))

    def test_is_valid_lab(self):
        assert eda.SingleLabelClassifiedSamples.is_valid_lab('1')
        assert eda.SingleLabelClassifiedSamples.is_valid_lab(1)
        assert not eda.SingleLabelClassifiedSamples.is_valid_lab(np.array([1])[0])
        assert not eda.SingleLabelClassifiedSamples.is_valid_lab([])
        assert not eda.SingleLabelClassifiedSamples.is_valid_lab([1])
        assert not eda.SingleLabelClassifiedSamples.is_valid_lab(None)
        assert not eda.SingleLabelClassifiedSamples.is_valid_lab((1,))

    def test_assert_is_valid_sids(self):
        with pytest.raises(Exception) as excinfo:
            eda.SingleLabelClassifiedSamples.assert_is_valid_sids(np.arange(5))

        with pytest.raises(Exception) as excinfo:
            eda.SingleLabelClassifiedSamples.assert_is_valid_sids([True, False])

        with pytest.raises(Exception) as excinfo:
            eda.SingleLabelClassifiedSamples.assert_is_valid_sids(None)

        with pytest.raises(Exception) as excinfo:
            eda.SingleLabelClassifiedSamples.assert_is_valid_sids([])

        with pytest.raises(Exception) as excinfo:
            eda.SingleLabelClassifiedSamples.assert_is_valid_sids([[1], [2]])

        with pytest.raises(Exception) as excinfo:
            eda.SingleLabelClassifiedSamples.assert_is_valid_sids(['1', 2, 3])

        with pytest.raises(Exception) as excinfo:
            eda.SingleLabelClassifiedSamples.assert_is_valid_sids(['1', '1', '3'])

        with pytest.raises(Exception) as excinfo:
            eda.SingleLabelClassifiedSamples.assert_is_valid_sids([0, 0, 1])

        with pytest.raises(Exception) as excinfo:
            eda.SingleLabelClassifiedSamples.assert_is_valid_sids(['1', 2, '3'])

        eda.SingleLabelClassifiedSamples.assert_is_valid_sids([1, 2])
        eda.SingleLabelClassifiedSamples.assert_is_valid_sids(['1', '2'])
        eda.SingleLabelClassifiedSamples.assert_is_valid_sids([1, 2, 3])

    def test_assert_is_valid_labs(self):
        with pytest.raises(Exception) as excinfo:
            eda.SingleLabelClassifiedSamples.assert_is_valid_labs(np.arange(5))

        with pytest.raises(Exception) as excinfo:
            eda.SingleLabelClassifiedSamples.assert_is_valid_labs([True, False])

        with pytest.raises(Exception) as excinfo:
            eda.SingleLabelClassifiedSamples.assert_is_valid_labs(None)

        with pytest.raises(Exception) as excinfo:
            eda.SingleLabelClassifiedSamples.assert_is_valid_labs([])

        with pytest.raises(Exception) as excinfo:
            eda.SingleLabelClassifiedSamples.assert_is_valid_labs([[1], [2]])

        with pytest.raises(Exception) as excinfo:
            eda.SingleLabelClassifiedSamples.assert_is_valid_labs(['1', 2, 1])

        with pytest.raises(Exception) as excinfo:
            eda.SingleLabelClassifiedSamples.assert_is_valid_labs(['1', 2, '3'])

        eda.SingleLabelClassifiedSamples.assert_is_valid_labs([1, 2])
        eda.SingleLabelClassifiedSamples.assert_is_valid_labs([1, 1, 3])
        eda.SingleLabelClassifiedSamples.assert_is_valid_labs(['1', '2', '3'])
            
    def test_lab_sorted_sids(self):
        qsids = [0, 1, 5, 3, 2, 4]
        qlabs = [0, 0, 2, 1, 1, 1]
        rsids = [3, 4, 2, 5, 1, 0]
        slab_csamples = eda.SingleLabelClassifiedSamples(qsids, qlabs)
        rs_qsids, rs_qlabs = slab_csamples.lab_sorted_sids(rsids)
        assert np.all(rs_qsids == np.array([3, 4, 2, 5, 1, 0]))
        assert np.all(rs_qlabs == np.array([1, 1, 1, 2, 0, 0]))

        rs_qsids, rs_qlabs = slab_csamples.lab_sorted_sids()
        assert np.all(rs_qsids == np.array([0, 1, 3, 2, 4, 5]))
        assert np.all(rs_qlabs == np.array([0, 0, 1, 1, 1, 2]))

    def test_filter_min_class_n(self):
        sids = [0, 1, 2, 3, 4, 5]
        labs = [0, 0, 0, 1, 2, 2]
        slab_csamples = eda.SingleLabelClassifiedSamples(sids, labs)
        min_cl_n = 2
        mcnf_sids, mcnf_labs = slab_csamples.filter_min_class_n(min_cl_n)
        assert np.all(mcnf_sids == np.array([0, 1, 2, 4, 5]))
        assert np.all(mcnf_labs == np.array([0, 0, 0, 2, 2]))

    def test_cross_labs(self):
        rsids = [0, 1, 2, 3, 4]
        rlabs = [0, 0, 0, 1, 1]
        rscl_samples = eda.SingleLabelClassifiedSamples(rsids, rlabs)
        
        qsids = [0, 1, 2, 3, 4]
        qlabs = [1, 1, 0, 2, 3]
        qscl_samples = eda.SingleLabelClassifiedSamples(qsids, qlabs)

        cross_lab_lut = rscl_samples.cross_labs(qscl_samples)
        test_lut = {
            0 : (3, ((0, 1), (1, 2))),
            1 : (2, ((2, 3), (1, 1)))
        }
        assert cross_lab_lut == test_lut

        cross_lab_lut = rscl_samples.cross_labs(qsids=qsids, qlabs=qlabs)
        test_lut = {
            0 : (3, ((0, 1), (1, 2))),
            1 : (2, ((2, 3), (1, 1)))
        }
        assert cross_lab_lut == test_lut

    def test_labs_to_cmap(self):
        sids = [0, 1, 2, 3, 4, 5, 6, 7]
        labs = list(map(str, [3, 0, 1, 0, 0, 1, 2, 2]))
        slab_csamples = eda.SingleLabelClassifiedSamples(sids, labs)

        lab_cmap, lab_ind_arr, lab_col_lut, uniq_lab_lut = slab_csamples.labs_to_cmap(
            return_lut=True)

        n_uniq_labs = len(set(labs))
        assert lab_cmap.N == n_uniq_labs
        assert lab_cmap.colors == sns.hls_palette(n_uniq_labs)
        assert np.all(lab_ind_arr == np.array([3, 0, 1, 0, 0, 1, 2, 2]))
        assert labs == [uniq_lab_lut[x] for x in lab_ind_arr]
        assert len(uniq_lab_lut) == n_uniq_labs
        assert len(lab_col_lut) == n_uniq_labs
        assert [lab_col_lut[uniq_lab_lut[i]] 
                for i in range(n_uniq_labs)] == sns.hls_palette(n_uniq_labs)

    