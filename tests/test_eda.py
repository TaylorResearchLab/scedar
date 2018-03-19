import numpy as np
import seaborn as sns
import scxplit.eda as eda
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
            eda.SampleFeatureMatrix(self.sfm5x10_lst, ['0', '0', '1', '2', '3'])

        with pytest.raises(Exception) as excinfo:
            eda.SampleFeatureMatrix(self.sfm5x10_lst, None, [0, 0, 1, 2, 3])
        
        with pytest.raises(Exception) as excinfo:
            eda.SampleFeatureMatrix(self.sfm5x10_lst, None, ['0', '0', '1', '2', '3'])

    def test_init_empty_x_sfids(self):
        with pytest.raises(Exception) as excinfo:
            eda.SampleFeatureMatrix(np.array([[], []]), [])

        with pytest.raises(Exception) as excinfo:
            eda.SampleFeatureMatrix(np.array([[], []]), None, [])

    def test_init_wrong_sid_len(self):
        # wrong sid size
        with pytest.raises(Exception) as excinfo:
            eda.SampleFeatureMatrix(self.sfm5x10_lst, list(range(10)), list(range(5)))

        with pytest.raises(Exception) as excinfo:
            eda.SampleFeatureMatrix(self.sfm5x10_lst, list(range(10)))

    def test_init_wrong_fid_len(self):
        # wrong fid size
        with pytest.raises(Exception) as excinfo:
            eda.SampleFeatureMatrix(self.sfm5x10_lst, list(range(5)), list(range(2)))

    def test_init_wrong_sfid_len(self):
        # wrong sid and fid sizes
        with pytest.raises(Exception) as excinfo:
            eda.SampleFeatureMatrix(self.sfm5x10_lst, list(range(10)), list(range(10)))

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
            eda.SampleFeatureMatrix(self.sfm3x3_arr, [False, True, 2], [0, 1, 1])

        with pytest.raises(Exception) as excinfo:
            eda.SampleFeatureMatrix(self.sfm3x3_arr, [[0], [0, 1], 2], [0, 1, 1])

        with pytest.raises(Exception) as excinfo:
            eda.SampleFeatureMatrix(self.sfm3x3_arr, np.array([0, 1, 2]), [0, 1, 1])

        with pytest.raises(Exception) as excinfo:
            eda.SampleFeatureMatrix(self.sfm3x3_arr, [(0), (0, 1), 2], [0, 1, 1])

    def test_init_bad_fid_type(self):
        with pytest.raises(Exception) as excinfo:
            eda.SampleFeatureMatrix(self.sfm3x3_arr, [0, 1, 2], [False, True, 2])

        with pytest.raises(Exception) as excinfo:
            eda.SampleFeatureMatrix(self.sfm3x3_arr, [0, 1, 2], [[0], [0, 1], 2])

        with pytest.raises(Exception) as excinfo:
            eda.SampleFeatureMatrix(self.sfm3x3_arr, [0, 1, 2], [(0), (0, 1), 2])

        with pytest.raises(Exception) as excinfo:
            eda.SampleFeatureMatrix(self.sfm3x3_arr, [0, 1, 2], np.array([0, 1, 2]))

    def test_valid_init(self):
        eda.SampleFeatureMatrix(self.sfm5x10_arr, list(range(5)), list(range(10)))
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

    def test_getters(self):
        tsfm = eda.SampleFeatureMatrix(np.arange(10).reshape(5, 2), 
                                       ['a', 'b', 'c', '1', '2'],
                                       ['a', 'z'])

        assert np.all(tsfm.get_x() == np.array(np.arange(10).reshape(5, 2), dtype='float64'))
        assert np.all(tsfm.get_sids() == np.array(['a', 'b', 'c', '1', '2']))
        assert np.all(tsfm.get_fids() == np.array(['a', 'z']))

        assert tsfm.get_x() is not tsfm._x
        assert tsfm.get_sids() is not tsfm._sids
        assert tsfm.get_fids() is not tsfm._fids


class TestSampleDistanceMatrix(object):
    """docstring for TestSampleDistanceMatrix"""
    def test_valid_init(self):
        eda.SampleDistanceMatrix(np.arange(10).reshape(5, 2))
        
        
class TestSingleLabelClassifiedSamples(object):
    """docstring for TestSingleLabelClassifiedSamples"""
    sfm3x3_arr = np.arange(9, dtype = "float64").reshape(3, 3)
    sfm_2x0 = np.array([[], []])
    sfm5x10_arr = np.random.ranf(50).reshape(5, 10)
    sfm5x10_lst = list(map(list, np.random.ranf(50).reshape(5, 10)))

    def test_init_empty_labs(self):
        with pytest.raises(Exception) as excinfo:
            eda.SingleLabelClassifiedSamples(self.sfm_2x0, [])

    def test_init_wrong_lab_len(self):
        with pytest.raises(Exception) as excinfo:
            eda.SingleLabelClassifiedSamples(self.sfm3x3_arr, [0, 1], None, None)

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
            eda.SingleLabelClassifiedSamples(self.sfm3x3_arr, [False, True, 2], [0, 1, 1], None)

        with pytest.raises(Exception) as excinfo:
            eda.SingleLabelClassifiedSamples(self.sfm3x3_arr, [[0], [0, 1], 2], [0, 1, 1], None)

        with pytest.raises(Exception) as excinfo:
            eda.SingleLabelClassifiedSamples(self.sfm3x3_arr, np.array([0, 1, 2]), [0, 1, 1], None)

        with pytest.raises(Exception) as excinfo:
            eda.SingleLabelClassifiedSamples(self.sfm3x3_arr, [(0), (0, 1), 2], [0, 1, 1], None)

    def test_valid_init(self):
        eda.SingleLabelClassifiedSamples(self.sfm5x10_arr, [0, 1, 1, 2, 0], list(range(5)), list(range(10)))
        eda.SingleLabelClassifiedSamples(self.sfm5x10_arr, [0, 1, 1, 2, 0], None, list(range(10)))
        eda.SingleLabelClassifiedSamples(self.sfm5x10_arr, ['a', 'a', 'b', 'd', 'c'], list(range(5)), None)
        eda.SingleLabelClassifiedSamples(np.arange(10).reshape(-1, 1), list(range(10)))
        eda.SingleLabelClassifiedSamples(np.arange(10).reshape(1, -1), ['a'])

    def test_is_valid_lab(self):
        assert eda.SingleLabelClassifiedSamples.is_valid_lab('1')
        assert eda.SingleLabelClassifiedSamples.is_valid_lab(1)
        assert not eda.SingleLabelClassifiedSamples.is_valid_lab(np.array([1])[0])
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
        assert np.all(rs_qsids == np.array([3, 4, 2, 5, 1, 0]))
        assert np.all(rs_qlabs == np.array([1, 1, 1, 2, 0, 0]))

        rs_qsids, rs_qlabs = slab_csamples.lab_sorted_sids()
        assert np.all(rs_qsids == np.array([0, 1, 3, 2, 4, 5]))
        assert np.all(rs_qlabs == np.array([0, 0, 1, 1, 1, 2]))

    def test_filter_min_class_n(self):
        sids = [0, 1, 2, 3, 4, 5]
        labs = [0, 0, 0, 1, 2, 2]
        slab_csamples = eda.SingleLabelClassifiedSamples(np.random.ranf(60).reshape(6, -1), 
                                                         labs, sids, None)
        min_cl_n = 2
        mcnf_sids, mcnf_labs = slab_csamples.filter_min_class_n(min_cl_n)
        assert np.all(mcnf_sids == np.array([0, 1, 2, 4, 5]))
        assert np.all(mcnf_labs == np.array([0, 0, 0, 2, 2]))

    def test_cross_labs(self):
        rsids = [0, 1, 2, 3, 4]
        rlabs = [0, 0, 0, 1, 1]
        rscl_samples = eda.SingleLabelClassifiedSamples(self.sfm5x10_lst, rlabs, rsids)
        
        qsids = [0, 1, 2, 3, 4]
        qlabs = [1, 1, 0, 2, 3]
        qscl_samples = eda.SingleLabelClassifiedSamples(self.sfm5x10_lst, qlabs, qsids)

        cross_lab_lut = rscl_samples.cross_labs(qscl_samples)
        test_lut = {
            0 : (3, ((0, 1), (1, 2))),
            1 : (2, ((2, 3), (1, 1)))
        }
        assert cross_lab_lut == test_lut

        qsids2 = [0, 1, 2]
        qlabs2 = [1, 1, 0]
        qscl_samples2 = eda.SingleLabelClassifiedSamples(self.sfm3x3_arr, qlabs2, qsids2)

        cross_lab_lut2 = rscl_samples.cross_labs(qscl_samples2)
        test_lut2 = {
            0 : (3, ((0, 1), (1, 2)))
        }
        assert cross_lab_lut2 == test_lut2

        with pytest.raises(Exception) as excinfo:
            rscl_samples.cross_labs([1,2,3])

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
        slab_csamples = eda.SingleLabelClassifiedSamples(np.random.ranf(8).reshape(8, -1), 
                                                         labs, sids)

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

        lab_cmap2 = slab_csamples.labs_to_cmap(return_lut=False)
        assert lab_cmap2.N == n_uniq_labs
        assert lab_cmap2.colors == lab_cmap.colors

    def test_getters(self):
        tslcs = eda.SingleLabelClassifiedSamples(np.arange(10).reshape(5, 2), 
                                                 [0, 0, 1, 2, 3],
                                                 ['a', 'b', 'c', '1', '2'],
                                                 ['a', 'z'])

        assert np.all(tslcs.get_x() == np.array(np.arange(10).reshape(5, 2), dtype='float64'))
        assert np.all(tslcs.get_sids() == np.array(['a', 'b', 'c', '1', '2']))
        assert np.all(tslcs.get_fids() == np.array(['a', 'z']))
        assert np.all(tslcs.get_labs() == np.array([0, 0, 1, 2, 3]))

        assert tslcs.get_x() is not tslcs._x
        assert tslcs.get_sids() is not tslcs._sids
        assert tslcs.get_fids() is not tslcs._fids
        assert tslcs.get_labs() is not tslcs._labs

    def test_lab_to_sids(self):
        tslcs = eda.SingleLabelClassifiedSamples(np.arange(10).reshape(5, 2), 
                                                 [0, 0, 1, 2, 3],
                                                 ['a', 'b', 'c', '1', '2'],
                                                 ['a', 'z'])
        qsid_arr = tslcs.labs_to_sids((0, 1))
        assert np.all(qsid_arr == (('a', 'b'), ('c',)))

    def test_sids_to_labs(self):
        tslcs = eda.SingleLabelClassifiedSamples(np.arange(10).reshape(5, 2), 
                                                 [0, 0, 1, 2, 3],
                                                 ['a', 'b', 'c', '1', '2'],
                                                 ['a', 'z'])
        qlab_arr = tslcs.sids_to_labs(('a', 'b', '2'))
        assert np.all(qlab_arr == np.array([0, 0, 3]))

        qlab_arr = tslcs.sids_to_labs(('1', 'a', 'b', '2'))
        assert np.all(qlab_arr == np.array([2, 0, 0, 3]))

