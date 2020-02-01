import numpy as np
import seaborn as sns
import scedar.eda as eda
import matplotlib as mpl
mpl.use("agg", warn=False)  # noqa
import matplotlib.pyplot as plt
import networkx as nx
import pytest


def test_labs_to_cmap():
    sids = [0, 1, 2, 3, 4, 5, 6, 7]
    labs = list(map(str, [3, 0, 1, 0, 0, 1, 2, 2]))
    slab_csamples = eda.SingleLabelClassifiedSamples(
        np.random.ranf(80).reshape(8, -1), labs, sids)

    (lab_cmap, lab_norm, lab_ind_arr, lab_col_lut,
     uniq_lab_lut) = eda.plot.labs_to_cmap(slab_csamples.labs, return_lut=True)

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

    lab_cmap2, lab_norm2 = eda.plot.labs_to_cmap(
        slab_csamples.labs, return_lut=False)
    assert lab_cmap2.N == n_uniq_labs
    assert lab_cmap2.colors == lab_cmap.colors
    np.testing.assert_equal(lab_norm2.boundaries, lab_norm.boundaries)


class TestRegressionScatter(object):
    """docstring for TestRegressionPlot"""
    @pytest.mark.mpl_image_compare
    def test_reg_sct_full_labels(self):
        fig, ax = plt.subplots()
        fig2 = eda.regression_scatter(x=np.arange(10), y=np.arange(10, 20),
                                      xlab='x', ylab='y', title='x versus y',
                                      figsize=(10, 10), ax=ax)
        assert fig is fig2
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
                                  figsize=(10, 10), n_txt_per_cluster=3,
                                  alpha=0.5,
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
    def test_cluster_scatter_gradient_legends_slab0(self):
        sorted_x = self.x_50x2[np.argsort(self.x_50x2[:, 1])]
        fig = eda.cluster_scatter(sorted_x,
                                  labels=[0]*25 + [1]*25,
                                  selected_labels=[0],
                                  gradient=sorted_x[:, 1],
                                  title='test tsne scatter',
                                  xlab='tsne1', ylab='tsne2',
                                  figsize=(10, 10), n_txt_per_cluster=3,
                                  alpha=0.5, s=50, random_state=123)
        return fig

    @pytest.mark.mpl_image_compare
    def test_cluster_scatter_nogradient_legends_slab0(self):
        sorted_x = self.x_50x2[np.argsort(self.x_50x2[:, 1])]
        fig = eda.cluster_scatter(sorted_x,
                                  labels=[0]*25 + [1]*25,
                                  selected_labels=[0],
                                  title='test tsne scatter',
                                  xlab='tsne1', ylab='tsne2',
                                  figsize=(10, 10), n_txt_per_cluster=3,
                                  alpha=0.5, s=50, random_state=123)
        return fig

    @pytest.mark.mpl_image_compare
    def test_cluster_scatter_gradient_legends_slabempty(self):
        sorted_x = self.x_50x2[np.argsort(self.x_50x2[:, 1])]
        fig = eda.cluster_scatter(sorted_x,
                                  labels=[0]*25 + [1]*25,
                                  selected_labels=[],
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
                                  title='test tsne scatter',
                                  xlab='tsne1', ylab='tsne2',
                                  figsize=(10, 10), n_txt_per_cluster=3,
                                  alpha=0.5,
                                  s=50, random_state=123)
        return fig

    @pytest.mark.mpl_image_compare
    def test_cluster_scatter_no_legends(self):
        fig = eda.cluster_scatter(self.x_50x2,
                                  [0]*25 + [1]*25,
                                  title='test tsne scatter',
                                  xlab='tsne1', ylab='tsne2',
                                  figsize=(10, 10), add_legend=False,
                                  n_txt_per_cluster=3, alpha=0.5,
                                  s=50, random_state=123)
        return fig

    @pytest.mark.mpl_image_compare
    def test_cluster_scatter_no_labels(self):
        fig = eda.cluster_scatter(self.x_50x2,
                                  title='test tsne scatter',
                                  xlab='tsne1', ylab='tsne2',
                                  figsize=(10, 10), n_txt_per_cluster=3,
                                  alpha=0.5,
                                  s=50, random_state=123)
        return fig

    def test_cluster_scatter_wrong_args(self):
        # wrong projection2d dimention
        with pytest.raises(ValueError) as excinfo:
            eda.cluster_scatter(np.random.ranf(100).reshape(-1, 1),
                                title='test tsne scatter',
                                xlab='tsne1', ylab='tsne2',
                                figsize=(10, 10), n_txt_per_cluster=3,
                                alpha=0.5,
                                s=50, random_state=123)

        with pytest.raises(ValueError) as excinfo:
            eda.cluster_scatter(np.random.ranf(100).reshape(-1, 5),
                                title='test tsne scatter',
                                xlab='tsne1', ylab='tsne2',
                                figsize=(10, 10), n_txt_per_cluster=3,
                                alpha=0.5,
                                s=50, random_state=123)

        with pytest.raises(ValueError) as excinfo:
            eda.cluster_scatter(np.random.ranf(99).reshape(-1, 3),
                                title='test tsne scatter',
                                xlab='tsne1', ylab='tsne2',
                                figsize=(10, 10), n_txt_per_cluster=3,
                                alpha=0.5,
                                s=50, random_state=123)

        # wrong label shape
        with pytest.raises(ValueError) as excinfo:
            eda.cluster_scatter(self.x_50x2,
                                [0] * 60,
                                title='test tsne scatter',
                                xlab='tsne1', ylab='tsne2',
                                figsize=(10, 10), n_txt_per_cluster=3,
                                alpha=0.5,
                                s=50, random_state=123)
        with pytest.raises(ValueError) as excinfo:
            eda.cluster_scatter(self.x_50x2,
                                [[0]] * 50,
                                title='test tsne scatter',
                                xlab='tsne1', ylab='tsne2',
                                figsize=(10, 10), n_txt_per_cluster=3,
                                alpha=0.5,
                                s=50, random_state=123)
        # wrong gradient shape
        with pytest.raises(ValueError) as excinfo:
            eda.cluster_scatter(self.x_50x2,
                                [0] * 50, gradient=list(range(60)),
                                title='test tsne scatter',
                                xlab='tsne1', ylab='tsne2',
                                figsize=(10, 10), n_txt_per_cluster=3,
                                alpha=0.5,
                                s=50, random_state=123)
        with pytest.raises(ValueError) as excinfo:
            eda.cluster_scatter(self.x_50x2,
                                [[0]] * 50,
                                gradient=np.arange(50).reshape(-1, 1),
                                title='test tsne scatter',
                                xlab='tsne1', ylab='tsne2',
                                figsize=(10, 10), n_txt_per_cluster=3,
                                alpha=0.5,
                                s=50, random_state=123)
        # select labels without providing labels
        with pytest.raises(ValueError) as excinfo:
            eda.cluster_scatter(self.x_50x2,
                                selected_labels=[0], gradient=list(range(50)),
                                title='test tsne scatter',
                                xlab='tsne1', ylab='tsne2',
                                figsize=(10, 10), n_txt_per_cluster=3,
                                alpha=0.5,
                                s=50, random_state=123)
        # select absent labels
        with pytest.raises(ValueError) as excinfo:
            eda.cluster_scatter(self.x_50x2, [0] * 50,
                                selected_labels=[1], gradient=list(range(50)),
                                title='test tsne scatter',
                                xlab='tsne1', ylab='tsne2',
                                figsize=(10, 10), n_txt_per_cluster=3,
                                alpha=0.5,
                                s=50, random_state=123)
        with pytest.raises(ValueError) as excinfo:
            eda.cluster_scatter(self.x_50x2, [0] * 50,
                                selected_labels=[1, 0],
                                gradient=list(range(50)),
                                title='test tsne scatter',
                                xlab='tsne1', ylab='tsne2',
                                figsize=(10, 10), n_txt_per_cluster=3,
                                alpha=0.5,
                                s=50, random_state=123)

    # Test markers
    @pytest.mark.mpl_image_compare
    def test_cluster_scatter_grad_lab_diff_markers(self):
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
                                  plot_different_markers=True,
                                  title='test tsne scatter',
                                  xlab='tsne1', ylab='tsne2',
                                  figsize=(10, 10), n_txt_per_cluster=3,
                                  alpha=0.5, s=50, random_state=123)
        return fig

    @pytest.mark.mpl_image_compare
    def test_cluster_scatter_grad_lab_diff_custom_markers(self):
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
                                  plot_different_markers=True,
                                  label_markers=['o']*10 + ['*']*25 + ['^']*15,
                                  title='test tsne scatter',
                                  xlab='tsne1', ylab='tsne2',
                                  figsize=(10, 10), n_txt_per_cluster=3,
                                  alpha=0.5, s=50, random_state=123)
        return fig

    @pytest.mark.mpl_image_compare
    def test_cluster_scatter_diff_lab_markers(self):
        fig = eda.cluster_scatter(self.x_50x2,
                                  [0]*25 + [1]*10 + [2]*15,
                                  figsize=(10, 10), n_txt_per_cluster=3,
                                  alpha=0.5, plot_different_markers=True,
                                  s=50, random_state=123)
        return fig

    @pytest.mark.mpl_image_compare
    def test_cluster_scatter_diff_lab_custom_markers(self):
        fig = eda.cluster_scatter(self.x_50x2,
                                  [0]*25 + [1]*10 + [2]*15,
                                  figsize=(10, 10), n_txt_per_cluster=3,
                                  alpha=0.5, plot_different_markers=True,
                                  label_markers=['o']*10 + ['*']*25 + ['^']*15,
                                  s=50, random_state=123)
        return fig

    def test_cluster_scatter_diff_lab_markers_wrong_args(self):
        # labs not provided
        with pytest.raises(ValueError) as excinfo:
            eda.cluster_scatter(self.x_50x2,
                                figsize=(10, 10), n_txt_per_cluster=3,
                                alpha=0.5, plot_different_markers=True,
                                label_markers=['o']*10 + ['*']*25 + ['^']*15,
                                s=50, random_state=123)
        # labs and markers have different lengths
        # wrong marker length
        with pytest.raises(ValueError) as excinfo:
            eda.cluster_scatter(self.x_50x2,
                                labels=[0]*25 + [1]*10 + [2]*15,
                                figsize=(10, 10), n_txt_per_cluster=3,
                                alpha=0.5, plot_different_markers=True,
                                label_markers=['o']*10 + ['*']*25 + ['^']*10,
                                s=50, random_state=123)
        # wrong label length
        with pytest.raises(ValueError) as excinfo:
            eda.cluster_scatter(self.x_50x2,
                                labels=[0]*25 + [1]*10 + [2]*10,
                                figsize=(10, 10), n_txt_per_cluster=3,
                                alpha=0.5, plot_different_markers=True,
                                label_markers=['o']*10 + ['*']*25 + ['^']*15,
                                s=50, random_state=123)

    @pytest.mark.mpl_image_compare
    def test_cluster_scatter_xylim(self):
        fig = eda.cluster_scatter(np.array([list(range(50)),
                                            list(range(100, 150))]).T,
                                  [0]*25 + [1]*25,
                                  title='xlim=(25, 50), ylim=(120, 150)',
                                  figsize=(10, 10), add_legend=False,
                                  xlim=(25, 50), ylim=(120, 150),
                                  n_txt_per_cluster=3, alpha=0.5,
                                  s=50, random_state=123)
        return fig


@pytest.mark.filterwarnings("ignore:The 'normed' kwarg is depreca")
@pytest.mark.mpl_image_compare
def test_hist_dens_plot():
    fig, ax = plt.subplots()
    eda.hist_dens_plot(np.arange(100), xlab='x', ylab='y', title='title',
                       ax=ax)
    eda.hist_dens_plot(np.arange(100), xlab='x', ylab='y', title='title')
    return eda.hist_dens_plot(np.arange(100))


class TestSwarm(object):
    """docstring for TestSwarm"""
    @pytest.mark.mpl_image_compare
    def test_swarm(self):
        x = np.arange(20)
        labels = [0]*10 + [1]*10
        return eda.swarm(x, labels, title='test swarm', xlab='x', ylab='y')

    @pytest.mark.mpl_image_compare
    def test_swarm_nolab(self):
        x = np.arange(20)
        labels = [0]*10 + [1]*10
        return eda.swarm(x)

    @pytest.mark.mpl_image_compare
    def test_swarm_s1(self):
        x = np.arange(20)
        labels = [0]*10 + [1]*10
        return eda.swarm(x, labels, 1)

    def test_swarm_ax(self):
        _, ax = plt.subplots()
        x = np.arange(20)
        labels = [0]*10 + [1]*10
        fig = eda.swarm(x, labels, ax=ax)
        axs = fig.get_axes()
        assert len(axs) == 1
        assert axs[0] is ax

    def test_swarm_wrong_args(self):
        x = np.arange(20)
        labels = [0]*10 + [1]*10
        # Wrong x shape
        with pytest.raises(ValueError) as excinfo:
            eda.swarm(x.reshape(-1, 1), labels + [0])
        # Wrong label shape
        with pytest.raises(ValueError) as excinfo:
            eda.swarm(x, labels + [0])
        with pytest.raises(ValueError) as excinfo:
            eda.swarm(x, [[9]]*20)
        # select absent labels
        with pytest.raises(ValueError) as excinfo:
            eda.swarm(x, labels=labels, selected_labels=[-1])
        # select labels without providing labels
        with pytest.raises(ValueError) as excinfo:
            eda.swarm(x, labels=None, selected_labels=[-1])
        # nothing get selected
        with pytest.raises(ValueError) as excinfo:
            eda.swarm(x, labels, [])
        # empty x
        with pytest.raises(ValueError) as excinfo:
            eda.swarm([])


class TestHeatmap(object):
    """docstring for TestHeatmap"""
    np.random.seed(123)
    x_10x5 = np.random.ranf(50).reshape(10, 5)

    @pytest.mark.mpl_image_compare
    def test_heatmap_crlabs(self):
        fig = eda.heatmap(self.x_10x5,
                          [2]*1 + [1]*3 + [5]*6,
                          [0] + [1]*2 + [2] + [3],
                          title='test heatmap',
                          xlab='col label', ylab='row label',
                          figsize=(10, 10))
        return fig

    @pytest.mark.mpl_image_compare
    def test_heatmap_crlabs_shuffle_rowc(self):
        fig = eda.heatmap(self.x_10x5,
                          [2]*1 + [1]*3 + [5]*6,
                          [0] + [1]*2 + [2] + [3],
                          title='test heatmap',
                          xlab='col label', ylab='row label',
                          figsize=(10, 10),  shuffle_row_colors=True,
                          random_state=17)
        return fig

    @pytest.mark.mpl_image_compare
    def test_heatmap_crlabs_shuffle_colc(self):
        fig = eda.heatmap(self.x_10x5,
                          [2]*1 + [1]*3 + [5]*6,
                          [0] + [1]*2 + [2] + [3],
                          title='test heatmap',
                          xlab='col label', ylab='row label',
                          figsize=(10, 10),  shuffle_col_colors=True,
                          random_state=17)
        return fig

    @pytest.mark.mpl_image_compare
    def test_heatmap_transform(self):
        # not callable transform
        with pytest.raises(ValueError) as excinfo:
            eda.heatmap(self.x_10x5,
                        [2]*1 + [1]*3 + [5]*6,
                        [0]*3 + [1]*2,
                        title='test heatmap',
                        xlab='col label', ylab='row label',
                        transform=1,
                        figsize=(10, 10))

        with pytest.raises(ValueError) as excinfo:
            eda.heatmap(self.x_10x5,
                        [2]*1 + [1]*3 + [5]*6,
                        [0]*3 + [1]*2,
                        title='test heatmap',
                        xlab='col label', ylab='row label',
                        transform=[],
                        figsize=(10, 10))

        with pytest.raises(ValueError) as excinfo:
            eda.heatmap(self.x_10x5,
                        [2]*1 + [1]*3 + [5]*6,
                        [0]*3 + [1]*2,
                        title='test heatmap',
                        xlab='col label', ylab='row label',
                        transform=self.x_10x5,
                        figsize=(10, 10))

        fig = eda.heatmap(np.arange(50).reshape(10, 5),
                          [2]*1 + [1]*3 + [5]*6,
                          [0]*3 + [1]*2,
                          title='test heatmap',
                          xlab='col label', ylab='row label',
                          transform=lambda x: x + 100,
                          figsize=(10, 10))
        return fig

    @pytest.mark.mpl_image_compare
    def test_heatmap_cmap(self):
        fig = eda.heatmap(self.x_10x5,
                          [2]*1 + [1]*3 + [5]*6,
                          [0]*3 + [1]*2,
                          title='test heatmap',
                          xlab='col label', ylab='row label',
                          cmap='viridis',
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
                              figsize=(5, 5), alpha=1, node_with_labels=True)
