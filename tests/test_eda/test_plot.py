import numpy as np
import seaborn as sns
import scedar.eda as eda
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import pytest


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
    def test_cluster_scatter_shuffle_labcol(self):
        fig = eda.cluster_scatter(self.x_50x2,
                                  [0]*25 + [1]*10 + [2]*15,
                                  shuffle_label_colors=True,
                                  figsize=(10, 10), n_txt_per_cluster=3,
                                  alpha=0.5, s=50, random_state=2)
        return fig

    @pytest.mark.mpl_image_compare
    def test_cluster_scatter_gradient_legends(self):
        sorted_x = self.x_50x2[np.argsort(self.x_50x2[:, 1])]
        # should not have error even if gradient is provided
        eda.cluster_scatter(sorted_x,
                            labels=[0]*25 + [1]*25,
                            shuffle_label_colors=True,
                            gradient=sorted_x[:, 1],
                            title='test tsne scatter',
                            xlab='tsne1', ylab='tsne2',
                            figsize=(10, 10), n_txt_per_cluster=3,
                            alpha=0.5, s=50, random_state=123)

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

@pytest.mark.mpl_image_compare
def test_networkx_graph():
    ng = nx.Graph()
    ng.add_edge(1, 2, weight=1)
    ng.add_edge(1, 3, weight=10)
    return eda.networkx_graph(ng, nx.kamada_kawai_layout(ng),
                              figsize=(5, 5), alpha=1, with_labels=True)
