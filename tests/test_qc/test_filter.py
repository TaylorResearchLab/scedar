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
        d = skf._sdm._d
        resl = skf.knn_filter_samples([3, 4, 5], [10]*3, [5, 6, 7])
        assert resl == [list(range(5, 10)), list(range(5, 10)), []]
        assert len(skf._res_lut) == 3
        assert skf._res_lut[(3, 10, 5)][-1][1] == resl[0]
        assert skf._res_lut[(4, 10, 6)][-1][1] == resl[1]
        assert skf._res_lut[(5, 10, 7)][-1][1] == resl[2]
        # 0 is also stored in res lut
        assert [t[0] for t in skf._res_lut[(3, 10, 5)]] == list(range(5+1))
        assert [t[0] for t in skf._res_lut[(4, 10, 6)]] == list(range(6+1))
        assert [t[0] for t in skf._res_lut[(5, 10, 7)]] == list(range(7+1))
        np.testing.assert_equal(skf._sdm._d, d)
        
    def test_knn_filter_samples_par(self):
        skf = qc.SampleKNNFilter(self.tsdm)
        d = skf._sdm._d
        resl = skf.knn_filter_samples([3, 4, 5], [10]*3, [5, 6, 7], 3)
        assert resl == [list(range(5, 10)), list(range(5, 10)), []]
        assert len(skf._res_lut) == 3
        assert skf._res_lut[(3, 10, 5)][-1][1] == resl[0]
        assert skf._res_lut[(4, 10, 6)][-1][1] == resl[1]
        assert skf._res_lut[(5, 10, 7)][-1][1] == resl[2]
        # 0 is also stored in res lut
        assert [t[0] for t in skf._res_lut[(3, 10, 5)]] == list(range(5+1))
        assert [t[0] for t in skf._res_lut[(4, 10, 6)]] == list(range(6+1))
        assert [t[0] for t in skf._res_lut[(5, 10, 7)]] == list(range(7+1))
        np.testing.assert_equal(skf._sdm._d, d)

    def test_knn_filter_samples_single_run(self):
        skf = qc.SampleKNNFilter(self.tsdm)
        resl = skf.knn_filter_samples(1, 10, 5)
        resl2 = skf.knn_filter_samples([1], [10], 5)
        # scalar and list params should have the same results
        assert resl == resl2
        # result lut should be the same length
        assert len(skf._res_lut) == 1
        assert skf._res_lut[1, 10, 5][-1][1] == resl[0]

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
    with pytest.raises(ValueError) as excinfo:
        f_tsfm2 = qc.remove_constant_features(tsfm2)
    