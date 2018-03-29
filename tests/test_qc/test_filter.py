import scedar.qc as qc
import scedar.eda as eda
import numpy as np
import pytest


class TestSampleKNNFilter(object):
    """docstring for TestSampleKNNFilter"""
    tsdm = eda.SampleDistanceMatrix(
        [[0,0], [1, 1], [200, 200], [200, 200], [200, 200],
         [100, 100], [101, 101], [99, 99], [100, 100], [102, 102]],
        metric="euclidean")

    def test_knn_filter_samples(self):
        skf = qc.SampleKNNFilter(self.tsdm)
        d = skf._sdm._d.copy()
        res_sdml = skf.knn_filter_samples([3, 4, 5], [10]*3, [5, 6, 7])
        resl = [x.sids for x in res_sdml]
        assert resl == [list(range(5, 10)), list(range(5, 10)), []]
        assert len(skf._res_lut) == 3
        assert skf._res_lut[(3, 10, 5)][1][-1] == resl[0]
        assert skf._res_lut[(4, 10, 6)][1][-1] == resl[1]
        assert skf._res_lut[(5, 10, 7)][1][-1] == resl[2]
        # d should not be changed
        np.testing.assert_equal(skf._sdm._d, d)
        
    def test_knn_filter_samples_par(self):
        skf = qc.SampleKNNFilter(self.tsdm)
        d = skf._sdm._d.copy()
        res_sdml = skf.knn_filter_samples([3, 4, 5], [10]*3, [5, 6, 7], 3)
        resl = [x.sids for x in res_sdml]
        assert resl == [list(range(5, 10)), list(range(5, 10)), []]
        assert len(skf._res_lut) == 3
        assert skf._res_lut[(3, 10, 5)][1][-1] == resl[0]
        assert skf._res_lut[(4, 10, 6)][1][-1] == resl[1]
        assert skf._res_lut[(5, 10, 7)][1][-1] == resl[2]
        # d should not be changed
        np.testing.assert_equal(skf._sdm._d, d)

    def test_knn_filter_samples_single_run(self):
        tsdm = eda.SampleDistanceMatrix(
            [[0,0], [1, 1], [200, 200], [101, 101],
             [99, 99], [100, 100], [102, 102]],
            metric="euclidean")
        skf = qc.SampleKNNFilter(tsdm)
        resl = skf.knn_filter_samples(1, 0.1, 1)
        assert resl[0]._x.shape == (0, 2)
        assert resl[0].sids == []
        resl2 = skf.knn_filter_samples(1, 0.1, 5)
        assert resl2[0]._x.shape == (0, 2)
        assert resl2[0].sids == []

    def test_knn_filter_samples_empty_subset(self):
        skf = qc.SampleKNNFilter(self.tsdm)
        resl = skf.knn_filter_samples(1, 10, 5)
        resl2 = skf.knn_filter_samples([1], [10], 5)
        # scalar and list params should have the same results
        assert [x.sids for x in resl] == [x.sids for x in resl2]
        # result lut should be the same length
        assert len(skf._res_lut) == 1
        assert skf._res_lut[(1, 10, 5)][1][-1] == resl[0].sids

    def test_knn_filter_samples_wrong_args(self):
        skf = qc.SampleKNNFilter(self.tsdm)
        # Invalid parameters
        with pytest.raises(ValueError) as excinfo:
            skf.knn_filter_samples(0, 1, 1)
        with pytest.raises(ValueError) as excinfo:
            skf.knn_filter_samples(1, 0, 1)
        with pytest.raises(ValueError) as excinfo:
            skf.knn_filter_samples(1, 1, 0)
        with pytest.raises(ValueError) as excinfo:
            skf.knn_filter_samples(1, 1, 0.5)
        with pytest.raises(ValueError) as excinfo:
            skf.knn_filter_samples(0.5, 1, 1)
        with pytest.raises(ValueError) as excinfo:
            skf.knn_filter_samples(1, -0.1, 1)
        with pytest.raises(ValueError) as excinfo:
            skf.knn_filter_samples(1, 1, 1, 0.5)
        with pytest.raises(ValueError) as excinfo:
            skf.knn_filter_samples(self.tsdm._x.shape[0], 1, 1, 1)
        # Parameters of different length
        with pytest.raises(ValueError) as excinfo:
            skf.knn_filter_samples([1, 2], 1, 1)
        with pytest.raises(ValueError) as excinfo:
            skf.knn_filter_samples(1, [1, 2], 1)
        with pytest.raises(ValueError) as excinfo:
            skf.knn_filter_samples(1, 1, [1, 2])

def test_remove_constant_features():
    tsfm = eda.SampleFeatureMatrix([[0, 1, 2, 5],
                                    [0, 1, 0, 2],
                                    [0, 1, 0, 5],
                                    [0, 1, 0, 5],
                                    [0, 1, 0, 5]])
    f_tsfm = qc.remove_constant_features(tsfm)
    assert f_tsfm.x == [[2, 5],
                        [0, 2],
                        [0, 5],
                        [0, 5],
                        [0, 5]]
    tsfm2 = eda.SampleFeatureMatrix([[0, 1, 2, 5]])
    f_tsfm2 = qc.remove_constant_features(tsfm2)
    assert f_tsfm2._x.shape == (1, 0)
    assert f_tsfm2._sids.shape == (1,)
    assert f_tsfm2._fids.shape == (0,)
    