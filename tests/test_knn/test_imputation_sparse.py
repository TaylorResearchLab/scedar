import pytest

import scedar.knn as knn
import scedar.eda as eda

import numpy as np

import scipy.sparse as spsp


class TestFeatureImputationSparse(object):
    """docstring for TestFeatureImputation"""
    tx = [[0, 0], [1, 1], [1, 2], [2, 3], [2, 5], [0, 100], [0, 100],
          [0, 101], [10, 100]]
    tsdm = eda.SampleDistanceMatrix(
        spsp.csr_matrix(tx), metric="euclidean")

    def run_test_impute_features(self, nprocs):
        fmp = knn.FeatureImputation(self.tsdm)
        d = fmp._sdm._d.copy()
        assert d.shape == (9, 9)
        res_sdml = fmp.impute_features([1, 3, 5], [1, 1, 3],
                                       [0.5, 1.5, 0.5], [1, 1, 5],
                                       nprocs=nprocs)
        assert len(fmp._res_lut) == 3
        assert res_sdml[0]._x.A.shape == (9, 2)
        assert res_sdml[1]._x.A.shape == (9, 2)
        assert res_sdml[2]._x.A.shape == (9, 2)
        assert not np.array_equal(res_sdml[0]._d, d)
        assert not np.array_equal(res_sdml[1]._d, d)
        assert not np.array_equal(res_sdml[2]._d, d)
        # value
        ref_res0 = np.array([[1, 1], [1, 1], [1, 2], [2, 3], [2, 5],
                             [0, 100], [0, 100], [0, 101], [10, 100]])
        np.testing.assert_equal(res_sdml[0]._x.A, ref_res0)

        ref_res1 = np.array([[np.median([1, 1, 2]), np.median([1, 2, 3])],
                             [np.median([1, 1, 2]), np.median([1, 2, 3])],
                             [np.median([1, 1, 2]), 2],
                             [2, 3],
                             [2, 5],
                             [np.median([0, 0, 10]), 100],
                             [np.median([0, 0, 10]), 100],
                             [np.median([0, 0, 10]), 101],
                             [10, 100]])
        np.testing.assert_equal(res_sdml[1]._x.A, ref_res1)

        assert res_sdml[2]._x.A[0, 0] > 0
        assert res_sdml[2]._x.A[0, 1] > 0
        assert res_sdml[2]._x.A[5, 0] > 0
        assert res_sdml[2]._x.A[6, 0] > 0
        assert res_sdml[2]._x.A[7, 0] > 0

        # lookup
        np.testing.assert_equal(fmp._res_lut[(1, 1, 0.5, 1, np.median)][0].A,
                                res_sdml[0]._x.A)
        np.testing.assert_equal(fmp._res_lut[(3, 1, 1.5, 1, np.median)][0].A,
                                res_sdml[1]._x.A)
        np.testing.assert_equal(fmp._res_lut[(5, 3, 0.5, 5, np.median)][0].A,
                                res_sdml[2]._x.A)
        # d should not be changed
        np.testing.assert_equal(fmp._sdm._d, d)
        fmp.impute_features([1, 3, 5], [1, 1, 3],
                            [0.5, 1.5, 0.5], [1, 1, 5],
                            nprocs=nprocs)
        assert len(fmp._res_lut) == 3
        np.testing.assert_equal(fmp._sdm._d, d)
        np.testing.assert_equal(fmp._sdm.x.A.tolist(), self.tx)
        # run results should be placed with the correct order
        fmp2 = knn.FeatureImputation(self.tsdm)
        res1 = fmp2.impute_features([1, 3, 5], [1, 1, 3],
                                    [0.5, 1.5, 0.5], [1, 1, 5],
                                    nprocs=nprocs)
        fmp3 = knn.FeatureImputation(self.tsdm)
        res2 = fmp3.impute_features([2, 3, 4], [1, 1, 3],
                                    [0.5, 1.5, 0.5], [1, 1, 5],
                                    nprocs=nprocs)
        res3 = fmp2.impute_features([2, 3, 4], [1, 1, 3],
                                    [0.5, 1.5, 0.5], [1, 1, 5],
                                    nprocs=nprocs)
        np.testing.assert_equal(res1[1]._x.A, res3[1]._x.A)
        np.testing.assert_equal(res2[0]._x.A, res3[0]._x.A)
        np.testing.assert_equal(res2[2]._x.A, res3[2]._x.A)

    def test_impute_features(self):
        self.run_test_impute_features(1)
        self.run_test_impute_features(3)

    def test_impute_features_single_run(self):
        fmp = knn.FeatureImputation(self.tsdm)
        d = fmp._sdm._d.copy()
        assert d.shape == (9, 9)
        res_sdml = fmp.impute_features([8], [3], [0.5], [10])

        fmp2 = knn.FeatureImputation(self.tsdm)
        res_sdml2 = fmp2.impute_features(8, 3, 0.5, 10)
        np.testing.assert_equal(res_sdml2[0]._x.A, res_sdml[0]._x.A)

    def test_impute_features_stat_fun(self):
        fmp = knn.FeatureImputation(self.tsdm)
        d = fmp._sdm._d.copy()
        assert d.shape == (9, 9)
        res_sdml = fmp.impute_features([8], [3], [0.5], [10])

        res_sdml2 = fmp.impute_features(8, 3, 0.5, 10, 1, np.median)
        np.testing.assert_equal(res_sdml2[0]._x.A, res_sdml[0]._x.A)
        assert len(fmp._res_lut) == 1

        res_sdml3 = fmp.impute_features(
            8, 3, 0.5, 10, 1, lambda x, axis=0: np.median(x, axis=axis))
        np.testing.assert_equal(res_sdml3[0]._x.A, res_sdml[0]._x.A)
        assert len(fmp._res_lut) == 2

        res_sdml4 = fmp.impute_features(
            8, 3, 0.5, 10, 1, lambda x, axis=0: np.median(x, axis=axis))
        np.testing.assert_equal(res_sdml4[0]._x.A, res_sdml[0]._x.A)
        assert len(fmp._res_lut) == 3

        res_sdml5 = fmp.impute_features(8, 3, 0.5, 10, 1, np.min)
        assert not np.array_equal(res_sdml5[0]._x.A, res_sdml[0]._x.A)
        assert len(fmp._res_lut) == 4

        res_sdml6 = fmp.impute_features(
            8, 3, 0.5, 10, 1, lambda x, axis=0: np.min(x, axis=axis))
        np.testing.assert_equal(res_sdml5[0]._x.A, res_sdml6[0]._x.A)
        assert len(fmp._res_lut) == 5

    def test_impute_features_wrong_args(self):
        fmp = knn.FeatureImputation(self.tsdm)
        # Invalid parameters
        with pytest.raises(ValueError) as excinfo:
            fmp.impute_features(0, 1, 10, 1)
        with pytest.raises(ValueError) as excinfo:
            fmp.impute_features(1, 0, 10, 1)
        with pytest.raises(ValueError) as excinfo:
            fmp.impute_features(1, 1, 10, 0)
        with pytest.raises(ValueError) as excinfo:
            fmp.impute_features(1, 1, 10, 0.5)
        with pytest.raises(ValueError) as excinfo:
            fmp.impute_features(0.5, 1, 10, 1)
        with pytest.raises(ValueError) as excinfo:
            fmp.impute_features(1, -0.1, 10, 1)
        with pytest.raises(ValueError) as excinfo:
            fmp.impute_features(1, 1, 1, 10, 0.5)
        with pytest.raises(ValueError) as excinfo:
            fmp.impute_features(9, 1, 1, 10, 1)
        # invalid stats funcs
        with pytest.raises(ValueError) as excinfo:
            fmp.impute_features([1, 3, 5], [1, 1, 3],
                                [0.5, 1.5, 0.5], [1, 1, 5],
                                1, 1)
        with pytest.raises(ValueError) as excinfo:
            fmp.impute_features([1, 3, 5], [1, 1, 3],
                                [0.5, 1.5, 0.5], [1, 1, 5],
                                1, np.array)
        with pytest.raises(ValueError) as excinfo:
            fmp.impute_features([1, 3, 5], [1, 1, 3],
                                [0.5, 1.5, 0.5], [1, 1, 5],
                                1, lambda x, y: x + y)
        with pytest.raises(ValueError) as excinfo:
            fmp.impute_features([1, 3, 5], [1, 1, 3],
                                [0.5, 1.5, 0.5], [1, 1, 5],
                                1, lambda x, y: x + y)
        # Parameters of different length
        with pytest.raises(ValueError) as excinfo:
            fmp.impute_features([1, 2], 1, 10, 1)
        with pytest.raises(ValueError) as excinfo:
            fmp.impute_features(1, [1, 2], 10, 1)
        with pytest.raises(ValueError) as excinfo:
            fmp.impute_features(1, 1, 10, [1, 2])
        with pytest.raises(ValueError) as excinfo:
            fmp.impute_features(1, 1, [10, 20], 1)
