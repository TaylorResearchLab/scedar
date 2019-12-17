import pytest

import numpy as np

import sklearn.datasets as skdset
from sklearn import metrics

import scedar.cluster as cluster
import scedar.eda as eda


class TestCommunity(object):
    '''docstring for TestMIRAC'''
    np.random.seed(123)
    x_20x5 = np.random.uniform(size=(20, 5))

    def test_simple_run(self):
        cluster.Community(self.x_20x5).labs

    def test_wrong_args(self):
        with pytest.raises(ValueError):
            cluster.Community(self.x_20x5, aff_scale=-0.1).labs
        with pytest.raises(ValueError):
            cluster.Community(self.x_20x5, metric='123').labs
        with pytest.raises(ValueError):
            cluster.Community(self.x_20x5, metric='correlation').labs
        with pytest.raises(ValueError):
            cluster.Community(
                self.x_20x5, partition_method='NotImplementedMethod').labs
    
    def test_different_partition_methods(self):
        cluster.Community(
            self.x_20x5, 
            partition_method="RBConfigurationVertexPartition").labs
        cluster.Community(
            self.x_20x5, partition_method="RBERVertexPartition").labs
        cluster.Community(
            self.x_20x5, partition_method="CPMVertexPartition").labs
        cluster.Community(
            self.x_20x5, partition_method="SignificanceVertexPartition").labs
        cluster.Community(
            self.x_20x5, partition_method="SurpriseVertexPartition").labs

    def test_provide_graph(self):
        sdm = eda.SampleDistanceMatrix(self.x_20x5)
        knn_conn_mat = sdm.s_knn_connectivity_matrix(5)
        knn_aff_graph = eda.SampleDistanceMatrix.knn_conn_mat_to_aff_graph(
            knn_conn_mat, 2)
        cluster.Community(
            self.x_20x5, graph=knn_aff_graph,
            partition_method="RBConfigurationVertexPartition").labs
