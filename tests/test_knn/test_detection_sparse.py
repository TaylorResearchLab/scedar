import scedar.knn as knn
import scedar.eda as eda
import numpy as np
import scipy.sparse as spsp
import pytest


class TestRareSampleDetectionSparse(object):
    """docstring for TestRareSampleDetection"""

    def test_detect_rare_samples(self):
        tsdm = eda.SampleDistanceMatrix(
            spsp.csr_matrix(
                [[0.00001, 0.00001], [1, 1], [200, 200], [200, 200],
                 [200, 200], [100, 100], [101, 101], [99, 99], [100, 100],
                 [102, 102]]),
            metric="euclidean", use_pdist=False)
        rsd = knn.RareSampleDetection(tsdm)
        resl = rsd.detect_rare_samples([3, 4, 5], [10]*3, [5, 6, 7])
        assert resl == [list(range(5, 10)), list(range(5, 10)), []]
        assert len(rsd._res_lut) == 3
        assert rsd._res_lut[(3, 10, 5)][1][-1] == resl[0]
        assert rsd._res_lut[(4, 10, 6)][1][-1] == resl[1]
        assert rsd._res_lut[(5, 10, 7)][1][-1] == resl[2]

    def test_detect_rare_samples_par(self):
        tsdm = eda.SampleDistanceMatrix(
            spsp.csr_matrix(
                [[0.00001, 0.00001], [1, 1], [200, 200], [200, 200],
                 [200, 200], [100, 100], [101, 101], [99, 99], [100, 100],
                 [102, 102]]),
            metric="euclidean", use_pdist=False)
        rsd = knn.RareSampleDetection(tsdm)
        resl = rsd.detect_rare_samples([3, 4, 5], [10]*3, [5, 6, 7], 3)
        assert resl == [list(range(5, 10)), list(range(5, 10)), []]
        assert len(rsd._res_lut) == 3
        assert rsd._res_lut[(3, 10, 5)][1][-1] == resl[0]
        assert rsd._res_lut[(4, 10, 6)][1][-1] == resl[1]
        assert rsd._res_lut[(5, 10, 7)][1][-1] == resl[2]

    def test_detect_rare_samples_single_run(self):
        tsdm = eda.SampleDistanceMatrix(
            spsp.csr_matrix(
                [[0.00001, 0.00001], [1, 1], [200, 200], [101, 101],
                 [99, 99], [100, 100], [102, 102]]),
            metric="euclidean", use_pdist=False)
        rsd = knn.RareSampleDetection(tsdm)
        res = rsd._no_pdist_rare_s_detect(1, 0.1, 1)
        resl = rsd.detect_rare_samples(1, 0.1, 1)
        assert resl[0] == []
        resl2 = rsd.detect_rare_samples(1, 0.1, 5)
        assert resl2[0] == []

    def test_detect_rare_samples_empty_subset(self):
        tsdm = eda.SampleDistanceMatrix(
            spsp.csr_matrix(
                [[0.00001, 0.00001], [1, 1], [200, 200], [200, 200],
                 [200, 200], [100, 100], [101, 101], [99, 99], [100, 100],
                 [102, 102]]),
            metric="euclidean", use_pdist=False)

        rsd = knn.RareSampleDetection(tsdm)
        resl = rsd.detect_rare_samples(1, 10, 5)
        resl2 = rsd.detect_rare_samples([1], [10], 5)
        # scalar and list params should have the same results
        assert resl == resl2
        # result lut should be the same length
        assert len(rsd._res_lut) == 1
        assert rsd._res_lut[(1, 10, 5)][1][-1] == resl[0]

    def test_detect_rare_samples_wrong_args(self):
        tsdm = eda.SampleDistanceMatrix(
            spsp.csr_matrix(
                [[0.00001, 0.00001], [1, 1], [200, 200], [200, 200],
                 [200, 200], [100, 100], [101, 101], [99, 99], [100, 100],
                 [102, 102]]),
            metric="euclidean", use_pdist=False)
        rsd = knn.RareSampleDetection(tsdm)
        # Invalid parameters
        with pytest.raises(ValueError) as excinfo:
            rsd.detect_rare_samples(0, 1, 1)
        with pytest.raises(ValueError) as excinfo:
            rsd.detect_rare_samples(1, 0, 1)
        with pytest.raises(ValueError) as excinfo:
            rsd.detect_rare_samples(1, 1, 0)
        with pytest.raises(ValueError) as excinfo:
            rsd.detect_rare_samples(1, 1, 0.5)
        with pytest.raises(ValueError) as excinfo:
            rsd.detect_rare_samples(0.5, 1, 1)
        with pytest.raises(ValueError) as excinfo:
            rsd.detect_rare_samples(1, -0.1, 1)
        # Parameters of different length
        with pytest.raises(ValueError) as excinfo:
            rsd.detect_rare_samples([1, 2], 1, 1)
        with pytest.raises(ValueError) as excinfo:
            rsd.detect_rare_samples(1, [1, 2], 1)
        with pytest.raises(ValueError) as excinfo:
            rsd.detect_rare_samples(1, 1, [1, 2])

    def test_hnsw_detect_rare_samples(self):
        tsdm = eda.SampleDistanceMatrix(
            spsp.csr_matrix(
                [[0.00001, 0.00001], [1, 1], [200, 200], [200, 200],
                 [200, 200], [100, 100], [101, 101], [99, 99], [100, 100],
                 [102, 102]]),
            metric="euclidean", use_pdist=False)
        rsd = knn.RareSampleDetection(tsdm)
        resl = rsd.detect_rare_samples([3, 4, 5], [10]*3, [5, 6, 7],
                                       use_pca=True, use_hnsw=True)
        assert resl == [list(range(5, 10)), list(range(5, 10)), []]
        assert len(rsd._res_lut) == 3
        assert rsd._res_lut[(3, 10, 5)][1][-1] == resl[0]
        assert rsd._res_lut[(4, 10, 6)][1][-1] == resl[1]
        assert rsd._res_lut[(5, 10, 7)][1][-1] == resl[2]

    def test_hnsw_detect_rare_samples_par(self):
        tsdm = eda.SampleDistanceMatrix(
            spsp.csr_matrix(
                [[0.00001, 0.00001], [1, 1], [200, 200], [200, 200],
                 [200, 200], [100, 100], [101, 101], [99, 99], [100, 100],
                 [102, 102]]),
            metric="euclidean", use_pdist=False)
        rsd = knn.RareSampleDetection(tsdm)
        resl = rsd.detect_rare_samples([3, 4, 5], [10]*3, [5, 6, 7], 3,
                                       use_pca=True, use_hnsw=True)
        assert resl == [list(range(5, 10)), list(range(5, 10)), []]
        assert len(rsd._res_lut) == 3
        assert rsd._res_lut[(3, 10, 5)][1][-1] == resl[0]
        assert rsd._res_lut[(4, 10, 6)][1][-1] == resl[1]
        assert rsd._res_lut[(5, 10, 7)][1][-1] == resl[2]

    def test_hnsw_detect_rare_samples_single_run(self):
        tsdm = eda.SampleDistanceMatrix(
            spsp.csr_matrix(
                [[0.00001, 0.00001], [1, 1], [200, 200], [101, 101],
                 [99, 99], [100, 100], [102, 102]]),
            metric="euclidean", use_pdist=False)
        rsd = knn.RareSampleDetection(tsdm)
        res = rsd._no_pdist_rare_s_detect(1, 0.1, 1)
        resl = rsd.detect_rare_samples(1, 0.1, 1, use_pca=True, use_hnsw=True)
        assert resl[0] == []
        resl2 = rsd.detect_rare_samples(1, 0.1, 5, use_pca=True, use_hnsw=True)
        assert resl2[0] == []

    def test_hnsw_detect_rare_samples_empty_subset(self):
        tsdm = eda.SampleDistanceMatrix(
            spsp.csr_matrix(
                [[0.00001, 0.00001], [1, 1], [200, 200], [200, 200],
                 [200, 200], [100, 100], [101, 101], [99, 99], [100, 100],
                 [102, 102]]),
            metric="euclidean", use_pdist=False)

        rsd = knn.RareSampleDetection(tsdm)
        resl = rsd.detect_rare_samples(1, 10, 5, use_pca=True, use_hnsw=True)
        resl2 = rsd.detect_rare_samples([1], [10], 5, use_pca=True,
                                        use_hnsw=True)
        # scalar and list params should have the same results
        assert resl == resl2
        # result lut should be the same length
        assert len(rsd._res_lut) == 1
        assert rsd._res_lut[(1, 10, 5)][1][-1] == resl[0]

    def test_hnsw_detect_rare_samples_wrong_args(self):
        tsdm = eda.SampleDistanceMatrix(
            spsp.csr_matrix(
                [[0.00001, 0.00001], [1, 1], [200, 200], [200, 200],
                 [200, 200], [100, 100], [101, 101], [99, 99], [100, 100],
                 [102, 102]]),
            metric="euclidean", use_pdist=False)
        rsd = knn.RareSampleDetection(tsdm)
        # Invalid parameters
        with pytest.raises(ValueError) as excinfo:
            rsd.detect_rare_samples(0, 1, 1, use_pca=True, use_hnsw=True)
        with pytest.raises(ValueError) as excinfo:
            rsd.detect_rare_samples(1, 0, 1, use_pca=True, use_hnsw=True)
        with pytest.raises(ValueError) as excinfo:
            rsd.detect_rare_samples(1, 1, 0, use_pca=True, use_hnsw=True)
        with pytest.raises(ValueError) as excinfo:
            rsd.detect_rare_samples(1, 1, 0.5, use_pca=True, use_hnsw=True)
        with pytest.raises(ValueError) as excinfo:
            rsd.detect_rare_samples(0.5, 1, 1, use_pca=True, use_hnsw=True)
        with pytest.raises(ValueError) as excinfo:
            rsd.detect_rare_samples(1, -0.1, 1, use_pca=True, use_hnsw=True)
        # Parameters of different length
        with pytest.raises(ValueError) as excinfo:
            rsd.detect_rare_samples([1, 2], 1, 1, use_pca=True, use_hnsw=True)
        with pytest.raises(ValueError) as excinfo:
            rsd.detect_rare_samples(1, [1, 2], 1, use_pca=True, use_hnsw=True)
        with pytest.raises(ValueError) as excinfo:
            rsd.detect_rare_samples(1, 1, [1, 2], use_pca=True, use_hnsw=True)

    def test_pdist_detect_rare_samples(self):
        tsdm = eda.SampleDistanceMatrix(
            spsp.csr_matrix(
                [[0.00001, 0.00001], [1, 1], [200, 200], [200, 200],
                 [200, 200], [100, 100], [101, 101], [99, 99], [100, 100],
                 [102, 102]]),
            metric="euclidean", use_pdist=True)
        rsd = knn.RareSampleDetection(tsdm)
        resl = rsd.detect_rare_samples([3, 4, 5], [10]*3, [5, 6, 7])
        assert resl == [list(range(5, 10)), list(range(5, 10)), []]
        assert len(rsd._res_lut) == 3
        assert rsd._res_lut[(3, 10, 5)][1][-1] == resl[0]
        assert rsd._res_lut[(4, 10, 6)][1][-1] == resl[1]
        assert rsd._res_lut[(5, 10, 7)][1][-1] == resl[2]

    def test_pdist_detect_rare_samples_par(self):
        tsdm = eda.SampleDistanceMatrix(
            spsp.csr_matrix(
                [[0.00001, 0.00001], [1, 1], [200, 200], [200, 200],
                 [200, 200], [100, 100], [101, 101], [99, 99], [100, 100],
                 [102, 102]]),
            metric="euclidean", use_pdist=True)
        rsd = knn.RareSampleDetection(tsdm)
        resl = rsd.detect_rare_samples([3, 4, 5], [10]*3, [5, 6, 7], 3)
        assert resl == [list(range(5, 10)), list(range(5, 10)), []]
        assert len(rsd._res_lut) == 3
        assert rsd._res_lut[(3, 10, 5)][1][-1] == resl[0]
        assert rsd._res_lut[(4, 10, 6)][1][-1] == resl[1]
        assert rsd._res_lut[(5, 10, 7)][1][-1] == resl[2]

    def test_pdist_detect_rare_samples_single_run(self):
        tsdm = eda.SampleDistanceMatrix(
            spsp.csr_matrix(
                [[0.00001, 0.00001], [1, 1], [200, 200], [101, 101],
                 [99, 99], [100, 100], [102, 102]]),
            metric="euclidean", use_pdist=True)
        rsd = knn.RareSampleDetection(tsdm)
        res = rsd._no_pdist_rare_s_detect(1, 0.1, 1)
        resl = rsd.detect_rare_samples(1, 0.1, 1)
        assert resl[0] == []
        resl2 = rsd.detect_rare_samples(1, 0.1, 5)
        assert resl2[0] == []

    def test_pdist_detect_rare_samples_empty_subset(self):
        tsdm = eda.SampleDistanceMatrix(
            spsp.csr_matrix(
                [[0.00001, 0.00001], [1, 1], [200, 200], [200, 200],
                 [200, 200], [100, 100], [101, 101], [99, 99], [100, 100],
                 [102, 102]]),
            metric="euclidean", use_pdist=True)

        rsd = knn.RareSampleDetection(tsdm)
        resl = rsd.detect_rare_samples(1, 10, 5)
        resl2 = rsd.detect_rare_samples([1], [10], 5)
        # scalar and list params should have the same results
        assert resl == resl2
        # result lut should be the same length
        assert len(rsd._res_lut) == 1
        assert rsd._res_lut[(1, 10, 5)][1][-1] == resl[0]

    def test_pdist_detect_rare_samples_wrong_args(self):
        tsdm = eda.SampleDistanceMatrix(
            spsp.csr_matrix(
                [[0.00001, 0.00001], [1, 1], [200, 200], [200, 200],
                 [200, 200], [100, 100], [101, 101], [99, 99], [100, 100],
                 [102, 102]]),
            metric="euclidean", use_pdist=True)
        rsd = knn.RareSampleDetection(tsdm)
        # Invalid parameters
        with pytest.raises(ValueError) as excinfo:
            rsd.detect_rare_samples(0, 1, 1)
        with pytest.raises(ValueError) as excinfo:
            rsd.detect_rare_samples(1, 0, 1)
        with pytest.raises(ValueError) as excinfo:
            rsd.detect_rare_samples(1, 1, 0)
        with pytest.raises(ValueError) as excinfo:
            rsd.detect_rare_samples(1, 1, 0.5)
        with pytest.raises(ValueError) as excinfo:
            rsd.detect_rare_samples(0.5, 1, 1)
        with pytest.raises(ValueError) as excinfo:
            rsd.detect_rare_samples(1, -0.1, 1)
        # Parameters of different length
        with pytest.raises(ValueError) as excinfo:
            rsd.detect_rare_samples([1, 2], 1, 1)
        with pytest.raises(ValueError) as excinfo:
            rsd.detect_rare_samples(1, [1, 2], 1)
        with pytest.raises(ValueError) as excinfo:
            rsd.detect_rare_samples(1, 1, [1, 2])
