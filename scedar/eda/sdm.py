import numpy as np

import scipy.cluster.hierarchy as sch
import scipy.spatial as spspatial
import scipy.sparse as spsparse

import sklearn as skl
import sklearn.metrics
import sklearn.manifold
from sklearn.neighbors import NearestNeighbors
import sklearn.preprocessing
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

import warnings

import random

from collections import defaultdict

import networkx as nx

from fa2 import ForceAtlas2

import igraph as ig

from umap import UMAP

import nmslib

import scedar
from scedar.eda.plot import cluster_scatter
from scedar.eda.plot import heatmap
from scedar.eda.plot import hist_dens_plot
from scedar.eda.plot import networkx_graph
from scedar.eda.sfm import SampleFeatureMatrix
from scedar.eda.mdl import MultinomialMdl
from scedar import utils
from scedar.eda import mtype


class SampleDistanceMatrix(SampleFeatureMatrix):
    """
    SampleDistanceMatrix: data with pairwise distance matrix

    Parameters
    ----------
    x : ndarray or list
        data matrix (n_samples, n_features)
    d : ndarray or list or None
        distance matrix (n_samples, n_samples)
        If is None, d will be computed with x, metric, and nprocs.
    metric : string
        distance metric
    use_pdist : boolean
        to use the pairwise distance matrix or not. The pairwise distance
        matrix may be too large to save for datasets with a large number of
        cells.
    sids : homogenous list of int or string
        sample ids. Should not contain duplicated elements.
    fids : homogenous list of int or string
        feature ids. Should not contain duplicated elements.
    nprocs : int
        the number of processes for computing pairwise distance matrix

    Attributes
    ----------
    _x : ndarray
        data matrix (n_samples, n_features)
    _d : ndarray
        distance matrix (n_samples, n_samples)
    _metric : string
        distance metric
    _sids : ndarray
        sample ids.
    _fids : ndarray
        sample ids.
    _tsne_lut: dict
        lookup table for previous tsne calculations. Each run has an
        indexed entry, {(param_str, index) : tsne_res}
    _last_tsne: float array
        The last *stored* tsne results. In no tsne performed before, a run
        with default parameters will be performed.
    _hnsw_index_lut: {string_index_parameters: hnsw_index}
    _last_k: int
        The last *k* used for s_knns computation.
    _last_knns: (knn_indices, knn_distances)
        The last computed s_knns.
    _knn_ng_lut: dict
        {(k, aff_scale): knn_graph}
    """

    def __init__(self, x, d=None, metric="cosine", use_pdist=True,
                 sids=None, fids=None, nprocs=None):
        super(SampleDistanceMatrix, self).__init__(x=x, sids=sids, fids=fids)

        if d is not None:
            if not use_pdist:
                raise ValueError("pdist cannot be provided when "
                                 "use pdist = False")
            try:
                d = np.array(d, copy=False, dtype="float64")
            except ValueError as e:
                raise ValueError("d must be float. {}".format(e))

            if (d.ndim != 2 or
                    d.shape[0] != d.shape[1] or
                    d.shape[0] != self._x.shape[0]):
                # check provided distance matrix shape
                raise ValueError("d should have shape (n_samples, n_samples)")

            d = self.num_correct_dist_mat(d)
        else:
            if metric == "precomputed":
                raise ValueError("metric cannot be precomputed when "
                                 "d is None.")

        if nprocs is None:
            self._nprocs = 1
        else:
            self._nprocs = max(int(nprocs), 1)

        self._lazy_load_d = d
        self._tsne_lut = {}
        self._lazy_load_last_tsne = None
        self._metric = metric
        self._use_pdist = use_pdist
        # Sorted distance matrix. Each column ascending from top to bottom.
        self._lazy_load_col_sorted_d = None
        self._lazy_load_col_argsorted_d = None
        self._knn_ng_lut = {}
        # sklearn.decomposition.PCA instance
        # use truncated svd for sparse
        if self._is_sparse:
            self._pca_n_components = min(100, self._x.shape[1] - 1)
        else:
            self._pca_n_components = min(
                100, self._x.shape[0], self._x.shape[1])
        self._lazy_load_skd_pca = None
        self._lazy_load_pca_x = None
        # umap
        self._lazy_load_umap_x = None
        # knn conn mat related
        self._last_k = None
        self._last_knns = None
        self._hnsw_index_lut = {}

    def to_classified(self, labels):
        """Convert to SingleLabelClassifiedSamples

        Args:
            labels (list of labels): sample labels.

        Returns:
            SingleLabelClassifiedSamples
        """
        slcs = scedar.eda.SingleLabelClassifiedSamples(
            self._x, labels, sids=self.sids, fids=self.fids, d=None,
            metric=self._metric, use_pdist=self._use_pdist,
            nprocs=self._nprocs)
        # pairwise distance matrix
        slcs._lazy_load_d = self._lazy_load_d
        # tsne
        slcs._tsne_lut = self._tsne_lut.copy()
        slcs._lazy_load_last_tsne = self._lazy_load_last_tsne
        # sorted distance matrix
        slcs._lazy_load_col_sorted_d = self._lazy_load_col_sorted_d
        slcs._lazy_load_col_argsorted_d = self._lazy_load_col_argsorted_d
        # knn graph
        slcs._knn_ng_lut = self._knn_ng_lut.copy()
        # pca
        slcs._pca_n_components = self._pca_n_components
        slcs._lazy_load_skd_pca = self._lazy_load_skd_pca
        slcs._lazy_load_pca_x = self._lazy_load_pca_x
        slcs._hnsw_index_lut = self._hnsw_index_lut.copy()
        return slcs

    def sort_features(self, fdist_metric="cosine", optimal_ordering=False):
        sorted_f_inds = HClustTree.sort_x_by_d(
            self._x.T, metric=fdist_metric, optimal_ordering=optimal_ordering,
            nprocs=self._nprocs)
        self._x = self._x[:, sorted_f_inds]
        self._fids = self._fids[sorted_f_inds]
        return

    # numerically correct dmat
    @staticmethod
    def num_correct_dist_mat(dmat, upper_bound=None):
        if (not isinstance(dmat, np.ndarray) or
                dmat.ndim != 2 or
                dmat.shape[0] != dmat.shape[1]):
            # check distance matrix shape
            raise ValueError("dmat must be a 2D (n_samples, n_samples)"
                             " np array")

        try:
            # Distance matrix diag vals should be close to 0.
            np.testing.assert_allclose(dmat[np.diag_indices(dmat.shape[0])], 0,
                                       atol=1e-5)
        except AssertionError as e:
            warnings.warn("distance matrix might not be numerically "
                          "correct. diag vals "
                          "should be close to 0. {}".format(e))

        try:
            # distance matrix should be approximately symmetric
            np.testing.assert_allclose(dmat[np.triu_indices_from(dmat)],
                                       dmat.T[np.triu_indices_from(dmat)],
                                       rtol=0.001)
        except AssertionError as e:
            warnings.warn("distance matrix might not be numerically "
                          "correct. should be approximately "
                          "symmetric. {}".format(e))

        dmat[dmat < 0] = 0
        dmat[np.diag_indices(dmat.shape[0])] = 0
        if upper_bound is not None:
            upper_bound = float(upper_bound)
            dmat[dmat > upper_bound] = upper_bound

        dmat[np.triu_indices_from(dmat)] = dmat.T[np.triu_indices_from(dmat)]
        return dmat

    def put_tsne(self, str_params, res):
        """
        Put t-SNE results into the lookup table.
        """
        if type(str_params) != str:
            raise ValueError("Unknown key type: {}".format(str_params))
        curr_store_ind = len(self._tsne_lut) + 1
        tsne_params_key = (str_params, curr_store_ind)
        res_copy = res.copy()
        self._tsne_lut[tsne_params_key] = res_copy
        self._lazy_load_last_tsne = res_copy

    def get_tsne_kv(self, key):
        """
        Get t-SNE results from the lookup table. Return None if non-existent.

        Returns
        -------
        res_tuple: tuple
            (key, val) pair of tsne result.
        """
        if type(key) == int:
            # tuple key (param_str, ind)
            for tk in self._tsne_lut:
                if key == tk[1]:
                    return (tk, self._tsne_lut[tk])
        elif type(key) == str:
            for tk in self._tsne_lut:
                if key == tk[0]:
                    return (tk, self._tsne_lut[tk])
        else:
            raise ValueError("Unknown key type: {}".format(key))
        # key cannot be found
        return None

    def tsne(self, store_res=True, **kwargs):
        """
        Run t-SNE on distance matrix.

        Parameters
        ----------
        store_res: bool
            Store the results in lookup table or not.
        **kwargs
            Keyword arguments passed to tsne computation.

        Returns
        -------
        tsne_res: float array
            t-SNE projections, (n_samples, m dimensions).
        """
        # TODO: make parameter keys consistent such that same set of
        # parameters but different order will sill be the same.
        # check input args
        if self._use_pdist:
            if ("metric" in kwargs and
                    kwargs["metric"] not in ("precomputed", self._metric)):
                raise ValueError("If you want to calculate t-SNE of a "
                                 "different "
                                 "metric than the instance metric, create "
                                 "another "
                                 "instance of the desired metric.")
            else:
                kwargs["metric"] = "precomputed"
        else:
            if "metric" in kwargs:
                if kwargs["metric"] == "precomputed":
                    raise ValueError("Metric cannot be precomputed when "
                                     "use_pdist is False")
                elif kwargs["metric"] != self._metric:
                    raise ValueError("If you want to calculate t-SNE of a "
                                    "different "
                                    "metric than the instance metric, create "
                                    "another "
                                    "instance of the desired metric.")
            else:
                kwargs["metric"] = self._metric

        str_params = str(sorted(kwargs.items()))
        tsne_kv = self.get_tsne_kv(str_params)
        if tsne_kv is None:
            if self._x.shape[0] == 0:
                tsne_res = np.empty((0, 0))
            elif self._x.shape[0] == 1:
                tsne_res = np.zeros((1, 2))
            else:
                if self._use_pdist:
                    tsne_res = tsne(self._d, **kwargs)
                else:
                    if self._is_sparse:
                        tsne_res = tsne(self._x.toarray(), **kwargs)
                    else:
                        tsne_res = tsne(self._x, **kwargs)

        else:
            tsne_res = tsne_kv[1]
        if store_res:
            self.put_tsne(str_params, tsne_res)

        return tsne_res

    def par_tsne(self, param_list, store_res=True, nprocs=1):
        """
        Run t-SNE with multiple sets of parameters parallely.

        Parameters
        ----------
        param_list: list of dict
            List of parameters being passed to t-SNE.
        nprocs: int
            Number of processes.

        Returns
        -------
        tsne_res_list: list of float arrays
            List of t-SNE results of corresponding parameter set.

        Notes
        -----
        Parallel running results cannot be stored during the run, because
        racing conditions may happen.
        """
        nprocs = min(int(nprocs), len(param_list))
        # single run tsne

        def srun_tsne(param_dict):
            return self.tsne(store_res=False, **param_dict)

        resl = utils.parmap(srun_tsne, param_list, nprocs)
        if store_res:
            for i in range(len(param_list)):
                self.put_tsne(str(param_list[i]), resl[i])
        return resl

    def tsne_plot(self, gradient=None, labels=None,
                  selected_labels=None,
                  plot_different_markers=False,
                  label_markers=None,
                  shuffle_label_colors=False,
                  xlim=None, ylim=None,
                  title=None, xlab=None, ylab=None,
                  figsize=(20, 20), add_legend=True,
                  n_txt_per_cluster=3, alpha=1, s=0.5,
                  random_state=None, **kwargs):
        """
        Plot the last t-SNE projection with the provided gradient as color.
        Gradient is None by default.
        """
        # labels are checked in cluster_scatter
        return cluster_scatter(self._last_tsne, labels=labels,
                               selected_labels=selected_labels,
                               plot_different_markers=plot_different_markers,
                               label_markers=label_markers,
                               shuffle_label_colors=shuffle_label_colors,
                               gradient=gradient,
                               xlim=xlim, ylim=ylim,
                               title=title, xlab=xlab, ylab=ylab,
                               figsize=figsize, add_legend=add_legend,
                               n_txt_per_cluster=n_txt_per_cluster,
                               alpha=alpha, s=s, random_state=random_state,
                               **kwargs)

    def tsne_feature_gradient_plot(self, fid, transform=None, labels=None,
                                   selected_labels=None,
                                   plot_different_markers=False,
                                   label_markers=None,
                                   shuffle_label_colors=False,
                                   xlim=None, ylim=None,
                                   title=None, xlab=None, ylab=None,
                                   figsize=(20, 20), add_legend=True,
                                   n_txt_per_cluster=3, alpha=1, s=0.5,
                                   random_state=None, **kwargs):
        """
        Plot the last t-SNE projection with the provided gradient as color.

        Parameters
        ----------
        fid: feature id scalar
            ID of the feature to be used for gradient plot.
        transform: callable
            Map transform on feature before plotting.
        labels: label array
            Labels assigned to each point, (n_samples,).
        selected_labels: label array
            Show gradient only for selected labels. Do not show non-selected.
        """
        if mtype.is_valid_sfid(fid):
            fid = [fid]
            f_ind = self.f_id_to_ind(fid)[0]
        else:
            raise ValueError("Invalid fid {}."
                             "Currently only support 1 "
                             "feature gradient plot.".format(fid))

        fx = self.f_ind_x_vec(f_ind, transform=transform)

        if labels is not None and len(labels) != fx.shape[0]:
            raise ValueError("labels ({}) must have same length as "
                             "n_samples.".format(labels))

        return cluster_scatter(self._last_tsne, labels=labels,
                               selected_labels=selected_labels,
                               plot_different_markers=plot_different_markers,
                               label_markers=label_markers,
                               shuffle_label_colors=shuffle_label_colors,
                               gradient=fx, xlim=xlim, ylim=ylim,
                               title=title, xlab=xlab, ylab=ylab,
                               figsize=figsize, add_legend=add_legend,
                               n_txt_per_cluster=n_txt_per_cluster,
                               alpha=alpha, s=s, random_state=random_state,
                               **kwargs)

    def pca_plot(self, component_ind_pair=(0, 1), gradient=None,
                 labels=None, selected_labels=None,
                 plot_different_markers=False,
                 label_markers=None,
                 shuffle_label_colors=False,
                 xlim=None, ylim=None,
                 title=None, xlab=None, ylab=None,
                 figsize=(20, 20), add_legend=True,
                 n_txt_per_cluster=3, alpha=1, s=0.5,
                 random_state=None, **kwargs):
        """
        Plot the PCA projection with the provided gradient as color.
        Gradient is None by default.
        """
        # labels are checked in cluster_scatter
        return cluster_scatter(self._pca_x[:, component_ind_pair],
                               labels=labels, selected_labels=selected_labels,
                               plot_different_markers=plot_different_markers,
                               label_markers=label_markers,
                               shuffle_label_colors=shuffle_label_colors,
                               gradient=gradient, xlim=xlim, ylim=ylim,
                               title=title, xlab=xlab, ylab=ylab,
                               figsize=figsize, add_legend=add_legend,
                               n_txt_per_cluster=n_txt_per_cluster,
                               alpha=alpha, s=s, random_state=random_state,
                               **kwargs)

    def pca_feature_gradient_plot(self, fid, component_ind_pair=(0, 1),
                                  transform=None, labels=None,
                                  selected_labels=None,
                                  plot_different_markers=False,
                                  label_markers=None,
                                  shuffle_label_colors=False,
                                  xlim=None, ylim=None,
                                  title=None, xlab=None, ylab=None,
                                  figsize=(20, 20), add_legend=True,
                                  n_txt_per_cluster=3, alpha=1, s=0.5,
                                  random_state=None, **kwargs):
        """
        Plot the last PCA projection with the provided gradient as color.

        Parameters
        ----------
        component_ind_pair: tuple of two ints
            Indices of the components to plot.
        fid: feature id scalar
            ID of the feature to be used for gradient plot.
        transform: callable
            Map transform on feature before plotting.
        labels: label array
            Labels assigned to each point, (n_samples,).
        selected_labels: label array
            Show gradient only for selected labels. Do not show non-selected.
        """
        if mtype.is_valid_sfid(fid):
            fid = [fid]
            f_ind = self.f_id_to_ind(fid)[0]
        else:
            raise ValueError("Invalid fid {}."
                             "Currently only support 1 "
                             "feature gradient plot.".format(fid))

        fx = self.f_ind_x_vec(f_ind, transform=transform)

        if labels is not None and len(labels) != fx.shape[0]:
            raise ValueError("labels ({}) must have same length as "
                             "n_samples.".format(labels))

        return cluster_scatter(self._pca_x[:, component_ind_pair],
                               labels=labels, selected_labels=selected_labels,
                               plot_different_markers=plot_different_markers,
                               label_markers=label_markers,
                               shuffle_label_colors=shuffle_label_colors,
                               gradient=fx, xlim=xlim, ylim=ylim,
                               title=title, xlab=xlab, ylab=ylab,
                               figsize=figsize, add_legend=add_legend,
                               n_txt_per_cluster=n_txt_per_cluster,
                               alpha=alpha, s=s, random_state=random_state,
                               **kwargs)

    def umap(self, use_pca=True, n_neighbors=5, n_components=2, n_epochs=None,
             learning_rate=1.0, init='spectral', min_dist=0.1, spread=1.0,
             set_op_mix_ratio=1.0, local_connectivity=1.0,
             repulsion_strength=1.0, negative_sample_rate=5,
             transform_queue_size=4.0, a=None, b=None, random_state=None,
             metric_kwds=None, angular_rp_forest=False, target_n_neighbors=-1,
             target_metric='categorical', target_metric_kwds=None,
             target_weight=0.5, transform_seed=42, verbose=False):
        if use_pca:
            umap_x = UMAP(
                n_neighbors=n_neighbors, n_components=n_components,
                metric=self._metric, n_epochs=n_epochs,
                learning_rate=learning_rate, init=init, min_dist=min_dist,
                spread=spread, set_op_mix_ratio=set_op_mix_ratio,
                local_connectivity=local_connectivity,
                repulsion_strength=repulsion_strength,
                negative_sample_rate=negative_sample_rate,
                transform_queue_size=transform_queue_size, a=a, b=b,
                random_state=random_state, metric_kwds=metric_kwds,
                angular_rp_forest=angular_rp_forest,
                target_n_neighbors=target_n_neighbors,
                target_metric=target_metric,
                target_metric_kwds=target_metric_kwds,
                target_weight=target_weight, transform_seed=transform_seed,
                verbose=verbose).fit_transform(self._pca_x)
        else:
            if self._use_pdist:
                umap_x = UMAP(
                    n_neighbors=n_neighbors, n_components=n_components,
                    metric='precomputed', n_epochs=n_epochs,
                    learning_rate=learning_rate, init=init, min_dist=min_dist,
                    spread=spread, set_op_mix_ratio=set_op_mix_ratio,
                    local_connectivity=local_connectivity,
                    repulsion_strength=repulsion_strength,
                    negative_sample_rate=negative_sample_rate,
                    transform_queue_size=transform_queue_size, a=a, b=b,
                    random_state=random_state, metric_kwds=metric_kwds,
                    angular_rp_forest=angular_rp_forest,
                    target_n_neighbors=target_n_neighbors,
                    target_metric=target_metric,
                    target_metric_kwds=target_metric_kwds,
                    target_weight=target_weight, transform_seed=transform_seed,
                    verbose=verbose).fit_transform(self._d)
            else:
                umap_x = UMAP(
                    n_neighbors=n_neighbors, n_components=n_components,
                    metric=self._metric, n_epochs=n_epochs,
                    learning_rate=learning_rate, init=init, min_dist=min_dist,
                    spread=spread, set_op_mix_ratio=set_op_mix_ratio,
                    local_connectivity=local_connectivity,
                    repulsion_strength=repulsion_strength,
                    negative_sample_rate=negative_sample_rate,
                    transform_queue_size=transform_queue_size, a=a, b=b,
                    random_state=random_state, metric_kwds=metric_kwds,
                    angular_rp_forest=angular_rp_forest,
                    target_n_neighbors=target_n_neighbors,
                    target_metric=target_metric,
                    target_metric_kwds=target_metric_kwds,
                    target_weight=target_weight, transform_seed=transform_seed,
                    verbose=verbose).fit_transform(self._x)

        self._lazy_load_umap_x = umap_x
        return umap_x

    def umap_plot(self, component_ind_pair=(0, 1), gradient=None,
                  labels=None, selected_labels=None,
                  plot_different_markers=False,
                  label_markers=None,
                  shuffle_label_colors=False,
                  xlim=None, ylim=None,
                  title=None, xlab=None, ylab=None,
                  figsize=(20, 20), add_legend=True,
                  n_txt_per_cluster=3, alpha=1, s=0.5,
                  random_state=None, **kwargs):
        """
        Plot the UMAP projection with the provided gradient as color.
        Gradient is None by default.

        TODO: refactor plotting interface. Merge multiple plotting methods into
        one.
        """
        # labels are checked in cluster_scatter
        return cluster_scatter(self._umap_x[:, component_ind_pair],
                               labels=labels, selected_labels=selected_labels,
                               plot_different_markers=plot_different_markers,
                               label_markers=label_markers,
                               shuffle_label_colors=shuffle_label_colors,
                               gradient=gradient, xlim=xlim, ylim=ylim,
                               title=title, xlab=xlab, ylab=ylab,
                               figsize=figsize, add_legend=add_legend,
                               n_txt_per_cluster=n_txt_per_cluster,
                               alpha=alpha, s=s, random_state=random_state,
                               **kwargs)

    def umap_feature_gradient_plot(self, fid, component_ind_pair=(0, 1),
                                   transform=None, labels=None,
                                   selected_labels=None,
                                   plot_different_markers=False,
                                   label_markers=None,
                                   shuffle_label_colors=False,
                                   xlim=None, ylim=None,
                                   title=None, xlab=None, ylab=None,
                                   figsize=(20, 20), add_legend=True,
                                   n_txt_per_cluster=3, alpha=1, s=0.5,
                                   random_state=None, **kwargs):
        """
        Plot the last UMAP projection with the provided gradient as color.

        Parameters
        ----------
        component_ind_pair: tuple of two ints
            Indices of the components to plot.
        fid: feature id scalar
            ID of the feature to be used for gradient plot.
        transform: callable
            Map transform on feature before plotting.
        labels: label array
            Labels assigned to each point, (n_samples,).
        selected_labels: label array
            Show gradient only for selected labels. Do not show non-selected.
        """
        if mtype.is_valid_sfid(fid):
            fid = [fid]
            f_ind = self.f_id_to_ind(fid)[0]
        else:
            raise ValueError("Invalid fid {}."
                             "Currently only support 1 "
                             "feature gradient plot.".format(fid))

        fx = self.f_ind_x_vec(f_ind, transform=transform)

        if labels is not None and len(labels) != fx.shape[0]:
            raise ValueError("labels ({}) must have same length as "
                             "n_samples.".format(labels))

        return cluster_scatter(self._umap_x[:, component_ind_pair],
                               labels=labels, selected_labels=selected_labels,
                               plot_different_markers=plot_different_markers,
                               label_markers=label_markers,
                               shuffle_label_colors=shuffle_label_colors,
                               gradient=fx, xlim=xlim, ylim=ylim,
                               title=title, xlab=xlab, ylab=ylab,
                               figsize=figsize, add_legend=add_legend,
                               n_txt_per_cluster=n_txt_per_cluster,
                               alpha=alpha, s=s, random_state=random_state,
                               **kwargs)

    def ind_x(self, selected_s_inds=None, selected_f_inds=None):
        """
        Subset samples by (sample IDs, feature IDs).

        Parameters
        ----------
        selected_s_inds: int array
            Index array of selected samples. If is None, select all.
        selected_f_inds: int array
            Index array of selected features. If is None, select all.

        Returns
        -------
        subset: SampleDistanceMatrix
        """
        if selected_s_inds is None:
            selected_s_inds = slice(None, None)

        if selected_f_inds is None:
            selected_f_inds = slice(None, None)

        if self._use_pdist:
            return SampleDistanceMatrix(
                x=self._x[selected_s_inds, :][:, selected_f_inds].copy(),
                d=self._d[selected_s_inds, :][:, selected_s_inds].copy(),
                metric=self._metric,
                use_pdist=self._use_pdist,
                sids=self._sids[selected_s_inds].tolist(),
                fids=self._fids[selected_f_inds].tolist(),
                nprocs=self._nprocs)
        else:
            return SampleDistanceMatrix(
                x=self._x[selected_s_inds, :][:, selected_f_inds].copy(),
                d=None,
                metric=self._metric,
                use_pdist=self._use_pdist,
                sids=self._sids[selected_s_inds].tolist(),
                fids=self._fids[selected_f_inds].tolist(),
                nprocs=self._nprocs)

    def id_x(self, selected_sids=None, selected_fids=None):
        """
        Subset samples by (sample IDs, feature IDs).

        Parameters
        ----------
        selected_s_inds: int array
            Index array of selected samples. If is None, select all.
        selected_f_inds: int array
            Index array of selected features. If is None, select all.

        Returns
        -------
        subset: SampleDistanceMatrix
        """
        if selected_sids is None:
            selected_s_inds = None
        else:
            selected_s_inds = self.s_id_to_ind(selected_sids)

        if selected_fids is None:
            selected_f_inds = None
        else:
            selected_f_inds = self.f_id_to_ind(selected_fids)
        return self.ind_x(selected_s_inds, selected_f_inds)

    def s_ith_nn_d(self, i):
        """
        Computes the distances of the i-th nearest neighbor of all samples.
        """
        return self._col_sorted_d[i, :]

    def s_ith_nn_ind(self, i):
        """
        Computes the sample indices of the i-th nearest neighbor of all
        samples.
        """
        return self._col_argsorted_d[i, :]

    def s_ith_nn_d_dist(self, i, xlab=None, ylab=None, title=None,
                        figsize=(5, 5), ax=None, **kwargs):
        """
        Plot the distances of the i-th nearest neighbor of all samples.
        """
        x = self.s_ith_nn_d(i)
        return hist_dens_plot(x, title=title, xlab=xlab, ylab=ylab,
                              figsize=figsize, ax=ax, **kwargs)

    def s_knn_ind_lut(self, k, metric=None, use_pca=False,
                      use_hnsw=False,
                      index_params=None, query_params=None,
                      verbose=False):
        """
        Computes the lookup table for sample i and its KNN indices, i.e.
        `{i : [1st_NN_ind, 2nd_NN_ind, ..., nth_NN_ind], ...}`
        """
        if k < 0 or k > self._x.shape[0] - 1:
            raise ValueError("k ({}) should >= 0 and <= n_samples-1".format(k))
        k = int(k)
        if self._use_pdist:
            # each column is its KNN index from 1 to k
            knn_ind_arr = self._col_argsorted_d[1:k+1, :].copy()
            knn_order_ind_lut = dict(zip(range(knn_ind_arr.shape[1]),
                                            knn_ind_arr.T.tolist()))
            return knn_order_ind_lut
        else:
            if self._last_k is None or self._last_k < k:
                if k < 10:
                    compute_k = min(10, max(self._x.shape[0] - 1, 0))
                else:
                    compute_k = k
                targets, distances = self.s_knns(
                    compute_k, metric=metric, use_pca=use_pca,
                    use_hnsw=use_hnsw,
                    index_params=index_params, query_params=query_params,
                    verbose=verbose)
            else:
                # if two samples are identical, their distance is 0
                targets, distances = self._last_knns

            knn_order_ind_lut = defaultdict(list)
            for i in range(len(targets)):
                # create a key val pair
                knn_order_ind_lut[i]
                for j in range(k):
                    knn_order_ind_lut[i].append(targets[i][j])

            return knn_order_ind_lut

    def s_knns(self, k, metric=None, use_pca=False,
               use_hnsw=False,
               index_params=None, query_params=None,
               verbose=False):
        """
        Computes the k-nearest neighbors (KNNs) of samples.

        Parameters
        ----------
        k: int
            The number of nearest neighbors.
        metric: {'cosine', 'euclidean', None}
            If none, self._metric is used.
        use_pca: bool
            Use PCA for nearest neighbors or not.
        use_hnsw: bool
            Use Hierarchical Navigable Small World graph to compute
            approximate nearest neighbor.
        index_params: dict
            Parameters used by HNSW in indexing.

            efConstruction: int
                Default 100. Higher value improves the quality of a constructed
                graph and leads to higher accuracy of search. However this also
                leads to longer indexing times. The reasonable range of values
                is 100-2000.
            M: int
                Default 5. Higher value leads to better recall and shorter
                retrieval times, at the expense of longer indexing time. The
                reasonable range of values is 5-100.
            delaunay_type: {0, 1, 2, 3}
                Default 2. Pruning heuristic, which affects the trade-off
                between retrieval performance and indexing time. The default
                is usually quite good.
            post: {0, 1, 2}
                Default 0. The amount and type of postprocessing applied to the
                constructed graph. 0 means no processing. 2 means more
                processing.
            indexThreadQty: int
                Default self._nprocs. The number of threads used.
        query_params: dict
            Parameters used by HNSW in querying.

            efSearch: int
                Default 100. Higher value improves recall at the expense of
                longer retrieval time. The reasonable range of values is
                100-2000.
        verbose: bool

        Returns
        -------
        knn_indices: list of numpy arrays
            The i-th array is the KNN indices of the i-th sample.
        knn_distances: list of numpy arrays
            The i-th array is the KNN distances of the i-th sample.
        """
        if k < 1:
            raise ValueError("k should >= 1")

        if use_hnsw:
            targets, distances = self._s_knns_hnsw(
                k=k, metric=metric, use_pca=use_pca,
                index_params=index_params,
                query_params=query_params,
                verbose=verbose)
        else:
            if index_params is not None:
                raise ValueError("index_params are not used with "
                                 "use_hnsw=False.")
            if query_params is not None:
                raise ValueError("query_params are not used with "
                                 "use_hnsw=False.")

            targets, distances = self._s_knns_skl(
                k, metric=metric, use_pca=use_pca, verbose=verbose)

        self._last_k = k
        self._last_knns = (targets, distances)
        return targets, distances

    def s_knn_connectivity_matrix(self, k, metric=None, use_pca=False,
                                  use_hnsw=False,
                                  index_params=None, query_params=None,
                                  verbose=False):
        """
        Computes the connectivity matrix of KNN of samples. If an entry
        `(i, j)` has value 0, node `i` is not in node `j`'s KNN. If an entry
        `(i, j)` has value != 0, node `i` is in `j`'s KNN, and their distance
        is the entry value. If two NNs have distance euqal to 0, 0 will be
        replaced by -np.inf.

        Parameters
        ----------
        k: int
            The number of nearest neighbors.
        metric: {'cosine', 'euclidean', None}
            If none, self._metric is used.
        use_pca: bool
            Use PCA for nearest neighbors or not.
        use_hnsw: bool
            Use Hierarchical Navigable Small World graph to compute
            approximate nearest neighbor.
        index_params: dict
            Parameters used by HNSW in indexing.

            efConstruction: int
                Default 100. Higher value improves the quality of a constructed
                graph and leads to higher accuracy of search. However this also
                leads to longer indexing times. The reasonable range of values
                is 100-2000.
            M: int
                Default 5. Higher value leads to better recall and shorter
                retrieval times, at the expense of longer indexing time. The
                reasonable range of values is 5-100.
            delaunay_type: {0, 1, 2, 3}
                Default 2. Pruning heuristic, which affects the trade-off
                between retrieval performance and indexing time. The default
                is usually quite good.
            post: {0, 1, 2}
                Default 0. The amount and type of postprocessing applied to the
                constructed graph. 0 means no processing. 2 means more
                processing.
            indexThreadQty: int
                Default self._nprocs. The number of threads used.

        query_params: dict
            Parameters used by HNSW in querying.

            efSearch: int
                Default 100. Higher value improves recall at the expense of
                longer retrieval time. The reasonable range of values is
                100-2000.

        Returns
        -------
        knn_conn_mat: float array
            (n_samples, n_samles) Non-zero entries are nearest neighbors (NNs).
            The values are distances. If two NNs have distance euqal to 0, 0
            will be replaced by -np.inf.
        """
        targets, distances = self.s_knns(
            k=k, metric=metric, use_pca=use_pca, use_hnsw=use_hnsw,
            index_params=index_params, query_params=query_params,
            verbose=verbose)

        sources = [np.repeat(i, len(targets[i])) for i in range(len(targets))]

        sources_1d = np.concatenate(sources, axis=0)
        targets_1d = np.concatenate(targets, axis=0)
        distances_1d = np.concatenate(distances, axis=0)

        distances_1d[distances_1d == 0] = -np.inf
        knn_conn_mat = spsparse.coo_matrix(
            (distances_1d, (sources_1d, targets_1d)),
            shape=(self._x.shape[0], self._x.shape[0])).tocsr()
        return knn_conn_mat

    def _s_knns_hnsw(self, k, metric=None, use_pca=False,
                     index_params=None, query_params=None,
                     verbose=False):
        if k < 1:
            raise ValueError("k should >= 1.")

        if metric is None:
            metric = self._metric

        if use_pca:
            data_x = self._pca_x
            is_sparse = False
            if metric == "euclidean":
                metric = "l2"
            elif metric == "cosine":
                metric = "cosinesimil"
            else:
                raise ValueError(
                    "HNSW only supports cosine and euclidean distance")
        else:
            data_x = self._x
            is_sparse = self._is_sparse
            if metric == "euclidean":
                if is_sparse:
                    metric = "l2_sparse"
                else:
                    metric = "l2"
            elif metric == "cosine":
                if is_sparse:
                    metric = "cosinesimil_sparse_fast"
                else:
                    metric = "cosinesimil"
            else:
                raise ValueError(
                    "HNSW only supports cosine and euclidean distance")

        if is_sparse:
            data_type = nmslib.DataType.SPARSE_VECTOR
        else:
            data_type = nmslib.DataType.DENSE_VECTOR

        if index_params is None:
            index_params = {
                "efConstruction": 100,
                "M": 5,
                "delaunay_type": 2,
                "post": 0,
                "indexThreadQty": self._nprocs
            }

        if query_params is None:
            query_params = {
                "efSearch": 100
            }

        # create index
        ind_pm_key = sorted([(k, v) for k, v in index_params.items()
                              if k in ["efConstruction", "M",
                                       "delaunay_type", "post"]])
        ind_pm_key.append(("metric", metric))
        ind_pm_key.append(("use_pca", use_pca))
        str_ind_pm_key = str(ind_pm_key)
        if str_ind_pm_key in self._hnsw_index_lut:
            hnsw = self._hnsw_index_lut[str_ind_pm_key]
        else:
            hnsw = nmslib.init(method="hnsw", space=metric,
                               data_type=data_type)
            hnsw.addDataPointBatch(data_x)
            hnsw.createIndex(index_params, print_progress=verbose)
            self._hnsw_index_lut[str_ind_pm_key] = hnsw
        # query KNN
        hnsw.setQueryTimeParams(query_params)
        # k nearest neighbors
        # hnsw query may include self.
        compute_k = k + 1
        knns = hnsw.knnQueryBatch(
            data_x, k=compute_k, num_threads=self._nprocs)
        # print(knns)
        # construct knn conn mat.
        knn_targets_sep_l = []
        knn_weights_sep_l = []
        # need benchmark
        for i in range(len(knns)):
            i_targets = knns[i][0]
            # TODO: warn if query result length is different from compute_k
            # TODO: add arg. If true, raise error when length is wrong,
            #       otherwise warn.
            i_weights = knns[i][1]
            # Note that the query result mey have < compute_k neighbors.
            for j in range(len(i_weights)):
                if i_targets[j] == i:
                    i_targets = np.delete(i_targets, j)
                    i_weights = np.delete(i_weights, j)
                    break
            else:
                # there is no self in the knn list
                i_targets = np.delete(i_targets, -1)
                i_weights = np.delete(i_weights, -1)

            knn_targets_sep_l.append(i_targets)
            knn_weights_sep_l.append(i_weights)

        return knn_targets_sep_l, knn_weights_sep_l

    def _s_knns_skl(self, k, metric=None, use_pca=False, verbose=False):
        """
        Runner for exact KNN using sklearn.
        """
        if metric is None:
            metric = self._metric

        if use_pca:
            if verbose:
                print("Construct {} distance KNN graph on PCs.".format(
                    metric))
            nn_ins = NearestNeighbors(
                n_neighbors=k, metric=metric).fit(self._pca_x)
        else:
            if verbose:
                print("Construct {} distance KNN graph on raw data.".format(
                    metric))
            if self._use_pdist:
                nn_ins = NearestNeighbors(
                    n_neighbors=k, metric="precomputed").fit(self._d)
            else:
                # TODO: better error message.
                # some metrics are not supported here.
                nn_ins = NearestNeighbors(
                    n_neighbors=k, metric=metric).fit(self._x)
        distances, targets = nn_ins.kneighbors(
            None, n_neighbors=k, return_distance=True)

        targets = list(targets)
        distances = list(distances)
        return targets, distances

    @staticmethod
    def knn_conn_mat_to_aff_graph(knn_conn_mat, aff_scale=1):
        sources, targets = knn_conn_mat.nonzero()
        weights = knn_conn_mat[sources, targets].A1
        weights[weights == -np.inf] = 0
        weights = (weights.max() - weights) * aff_scale
        graph = ig.Graph(edges=list(zip(sources, targets)),
                         directed=False, edge_attrs={"weight": weights})
        return graph


    def s_knn_graph(self, k, gradient=None, labels=None,
                    different_label_markers=True, aff_scale=1,
                    iterations=2000, figsize=(20, 20), node_size=30,
                    alpha=0.05, random_state=None, init_pos=None,
                    node_with_labels=False, fa2_kwargs=None,
                    nx_draw_kwargs=None):
        """
        Draw KNN graph of SampleDistanceMatrix. Graph layout using
        forceatlas2 for its speed on large graph.

        Parameters
        ----------
        k: int
        gradient: float array
            (n_samples,) color gradient
        labels: label list
            (n_samples,) labels
        different_label_markers: bool
            whether plot different labels with different markers
        aff_scale: float
            Affinity is calculated by `(max(distance) - distance) * aff_scale`
        iterations: int
            ForceAtlas2 iterations
        figsize: (float, float)
        node_size: float
        alpha: float
        random_state: int
        init_pos: float array
            Initial position of ForceAtlas2, (n_samples, 2).
        node_with_labels: bool
        fa2_kwargs: dict
        nx_draw_kwargs: dict

        Returns
        -------
        fig: matplotlib figure
            KNN graph.
        """
        # TODO: Docstring. Feature gradient.
        # (n_samples, n_samples). Non-neighbor entries are 0.
        knn_ng_param_key = (k, aff_scale)
        if knn_ng_param_key in self._knn_ng_lut:
            ng = self._knn_ng_lut[knn_ng_param_key]
        else:
            knn_d_csr = self.s_knn_connectivity_matrix(k)
            # affinity shoud negatively correlate to distance
            knn_d_csr.data = (knn_d_csr.max() - knn_d_csr.data) * aff_scale
            # Undirected graph
            ng = nx.Graph()
            ng = nx.from_scipy_sparse_matrix(
                knn_d_csr, parallel_edges=False, create_using=ng,
                edge_attribute="weight")
            self._knn_ng_lut[knn_ng_param_key] = ng

        if fa2_kwargs is None:
            fa2_kwargs = {}
        else:
            fa2_kwargs = fa2_kwargs.copy()

        random.seed(random_state)
        forceatlas2 = ForceAtlas2(
            # Dissuade hubs
            outboundAttractionDistribution=fa2_kwargs.pop(
                "outboundAttractionDistribution", True),
            edgeWeightInfluence=fa2_kwargs.pop("edgeWeightInfluence", 1.0),
            # Performance
            # Tolerance
            jitterTolerance=fa2_kwargs.pop("jitterTolerance", 1.0),
            barnesHutOptimize=fa2_kwargs.pop("barnesHutOptimize", True),
            barnesHutTheta=fa2_kwargs.pop("barnesHutTheta", 1.2),
            # Tuning
            scalingRatio=fa2_kwargs.pop("scalingRatio", 2.0),
            strongGravityMode=fa2_kwargs.pop("strongGravityMode", True),
            gravity=fa2_kwargs.pop("gravity", 1.0),
            # Log
            verbose=fa2_kwargs.pop("verbose", False), **fa2_kwargs)

        knn_fa2pos = forceatlas2.forceatlas2_networkx_layout(
            ng, pos=init_pos, iterations=iterations)

        if nx_draw_kwargs is None:
            nx_draw_kwargs = {}
        else:
            nx_draw_kwargs = nx_draw_kwargs.copy()

        fig = networkx_graph(ng, knn_fa2pos, alpha=alpha, figsize=figsize,
                             gradient=gradient, labels=labels,
                             different_label_markers=different_label_markers,
                             node_size=node_size,
                             node_with_labels=node_with_labels,
                             nx_draw_kwargs=nx_draw_kwargs)
        return fig

    @property
    def d(self):
        return self._d.tolist()

    @property
    def _d(self):
        if self._use_pdist:
            if self._is_sparse:
                sparse_compatible_metrics = ["cityblock", "cosine",
                                             "euclidean",
                                             "l1", "l2", "manhattan"]
                if self._metric not in sparse_compatible_metrics:
                    raise ValueError("Only the following metrics are "
                                     "supportted "
                                     "for sparse matrices, {}".format(
                                        sparse_compatible_metrics))

            if self._lazy_load_d is None:
                if self._x.size == 0:
                    self._lazy_load_d = np.zeros((self._x.shape[0],
                                                  self._x.shape[0]))
                else:
                    if self._metric == "cosine":
                        pdmat = self.cosine_pdist(self._x)
                    elif self._metric == "correlation":
                        pdmat = self.correlation_pdist(self._x)
                    else:
                        pdmat = skl.metrics.pairwise.pairwise_distances(
                            self._x, metric=self._metric, n_jobs=self._nprocs)
                    self._lazy_load_d = self.num_correct_dist_mat(pdmat)
        else:
            raise ValueError("This SDM instance does not use pdist.")
        return self._lazy_load_d

    @staticmethod
    def cosine_pdist(x):
        """
        Compute pairwise cosine pdist for x (n_samples, n_features).

        Adapted from Waylon Flinn's post on
        https://stackoverflow.com/a/20687984/4638182 .

        Cosine distance is undefined if one of the vectors contain only 0s.

        Parameters
        ----------
        x: ndarray
            (n_samples, n_features)

        Returns
        -------
        d: ndarray
            Pairwise distance matrix, (n_samples, n_samples).
        """
        # pairwise dot product matrix
        if spsparse.issparse(x):
            pdot_prod = np.dot(x, x.T).toarray()
        else:
            pdot_prod = np.dot(x, x.T)
        # diagonal values are self dot product, i.e. squared and sum.
        squared_sum = np.diag(pdot_prod)
        # inverse squared sum
        # when a sample has all 0s, its squared sum is also 0, thus inverse
        # is infinite. By sklearn pairwise_distances convention,
        # Zero vectors, e.g. [0, 0, 0, 0, 0], have cosine distance 0 to
        # themselves, but 1 distance to any other vectors including another
        # zero vector. In correlation distance, zero vectors have 0 distances
        # to themselves, but nan to others.
        # if it doesn't occur, set it's inverse magnitude to zero (instead of
        # inf)
        inv_squared_sum = [1 / ss if ss != 0 else 0 for ss in squared_sum]
        # square root of squared sum is l2 norm.
        inv_l2norm = np.sqrt(inv_squared_sum)
        # cosine similarity (elementwise multiply by inverse l2norm product)
        cosine_sim = pdot_prod * inv_l2norm
        cosine_sim = cosine_sim.T * inv_l2norm
        cosine_sim[np.diag_indices_from(cosine_sim)] = 1
        return 1 - cosine_sim

    @staticmethod
    def correlation_pdist(x):
        """
        Compute pairwise correlation pdist for x (n_samples, n_features).

        Adapted from Waylon Flinn's post on
        https://stackoverflow.com/a/20687984/4638182 .

        Parameters
        ----------
        x: ndarray
            (n_samples, n_features)

        Returns
        -------
        d: ndarray
            Pairwise distance matrix, (n_samples, n_samples).
        """
        centered_x = x - x.mean(axis=1).reshape(x.shape[0], 1)
        return SampleDistanceMatrix.cosine_pdist(centered_x)

    @property
    def _col_sorted_d(self):
        # Use mergesort to be stable
        if self._lazy_load_col_sorted_d is None:
            self._lazy_load_col_sorted_d = np.sort(self._d,
                                                   kind="mergesort",
                                                   axis=0)
        return self._lazy_load_col_sorted_d

    @property
    def _col_argsorted_d(self):
        # Use mergesort to be stable
        if self._lazy_load_col_argsorted_d is None:
            self._lazy_load_col_argsorted_d = np.argsort(self._d,
                                                         kind="mergesort",
                                                         axis=0)
        return self._lazy_load_col_argsorted_d

    @property
    def _last_tsne(self):
        if self._lazy_load_last_tsne is None:
            self._lazy_load_last_tsne = self.tsne()
        return self._lazy_load_last_tsne

    @property
    def _skd_pca(self):
        if self._lazy_load_skd_pca is None:
            if self._is_sparse:
                # use TruncatedSVD for sparse matrix
                self._lazy_load_skd_pca = TruncatedSVD(
                    n_components=self._pca_n_components, random_state=17)
                self._lazy_load_skd_pca.fit(self._x)
            else:
                # use standard PCA for dense matrix
                self._lazy_load_skd_pca = PCA(
                    n_components=self._pca_n_components,
                    svd_solver="auto", random_state=17)
                self._lazy_load_skd_pca.fit(self._x)

        return self._lazy_load_skd_pca

    @property
    def _pca_x(self):
        if self._lazy_load_pca_x is None:
            self._lazy_load_pca_x = self._skd_pca.transform(self._x)
        return self._lazy_load_pca_x

    @property
    def _umap_x(self):
        if self._lazy_load_umap_x is None:
            self._lazy_load_umap_x = self.umap(random_state=17)
        return self._lazy_load_umap_x

    @property
    def metric(self):
        return self._metric

    @property
    def tsne_lut(self):
        return dict((key, val.copy()) for key, val in self._tsne_lut.items())


# x : (n_samples, n_features) or (n_samples, n_samples)
# If metric is 'precomputed', x must be a pairwise distance matrix
def tsne(x, n_components=2, perplexity=30.0, early_exaggeration=12.0,
         learning_rate=200.0, n_iter=1000, n_iter_without_progress=300,
         min_grad_norm=1e-07, metric="euclidean", init="random", verbose=0,
         random_state=None, method="barnes_hut", angle=0.5):
    x_tsne = sklearn.manifold.TSNE(
        n_components=n_components, perplexity=perplexity,
        early_exaggeration=early_exaggeration,
        learning_rate=learning_rate, n_iter=n_iter,
        n_iter_without_progress=n_iter_without_progress,
        min_grad_norm=min_grad_norm, metric=metric,
        init=init, verbose=verbose,
        random_state=random_state, method=method,
        angle=angle).fit_transform(x)
    return x_tsne


class HClustTree(object):
    """
    Hierarchical clustering tree.

    Implement simple tree operation routines. HCT is binary unbalanced tree.

    Attributes
    ----------
    node : scipy.cluster.hierarchy.ClusterNode
        current node
    prev : HClustTree
        parent of current node

    """

    def __init__(self, node, prev=None):
        super(HClustTree, self).__init__()
        self._node = node
        self._prev = prev

        if node is None:
            left = None
            right = None
        else:
            left = node.get_left()
            right = node.get_right()

        self._left = left
        self._right = right

    @property
    def prev(self):
        return self._prev

    def count(self):
        if self._node is None:
            return 0
        else:
            return self._node.get_count()

    def left(self):
        return HClustTree(self._left, self)

    def left_count(self):
        return self.left().count()

    def right(self):
        return HClustTree(self._right, self)

    def right_count(self):
        return self.right().count()

    def leaf_ids(self):
        """Returns the list of leaf IDs from left to right

        Returns:
            :obj:`list` of leaf IDs
        """
        if self._node is None:
            return []
        else:
            return self._node.pre_order(lambda xn: xn.get_id())

    def left_leaf_ids(self):
        return self.left().leaf_ids()

    def right_leaf_ids(self):
        return self.right().leaf_ids()

    def bi_partition(self, soft_min_subtree_size=1,
                     return_subtrees=False):
        """
        soft_min_subtree_size: when curr tree size < 2 * soft_min_subtree_size,
        it is impossible to have a bipartition with a minimum sub tree size
        bigger than soft_min_subtree_size. In this case, return the first
        partition.

        When soft_min_subtree_size = 1, the performance is the same as taking
        the first bipartition.

        When curr size = 1, the first bipartition gives
        (1, 0). Because curr size < 2 * soft_min_subtree_size, it goes directly
        to return.

        When curr size = 2, the first bipartition guarantees to give (1, 1),
        with the invariant that parent nodes of leaves always have 2 child
        nodes. This also goes directly to return.

        When curr size >= 3, the first bipartition guarantees to give two
        subtrees with size >= 1, with the same invariant in size = 2.
        """
        soft_min_subtree_size = int(soft_min_subtree_size)
        if soft_min_subtree_size < 1:
            raise ValueError("soft_min_subtree_size should >= 1")
        lst = self.left()
        rst = self.right()
        if (self.count() >= 2 * soft_min_subtree_size and
                lst.count() != rst.count()):
            # cut is not balanced
            if lst.count() < rst.count():
                min_st = lst
                max_st = rst
                min_side = "left"
            else:
                min_st = rst
                max_st = lst
                min_side = "right"
            # Invariants:
            # 1. min_st < max_st
            # 2. min_st size >= 1, which implies that max_st size >= 2
            # 3. parent node of a leaf has two child nodes.
            while min_st.count() < soft_min_subtree_size:
                # increase min_st size until it is larger than min_st_size
                # or the max st size
                if min_side == "left":
                    max_spl_st = max_st.left()
                    max_const_st = max_st.right()
                else:
                    # min side is right
                    max_spl_st = max_st.right()
                    max_const_st = max_st.left()
                if max_spl_st.count() <= 1:
                    if max_spl_st.count() <= 0:
                        # count < 1 or count == 0
                        # This should not happen given max_st > min_st >= 1
                        raise NotImplementedError("Unexpected branch reached")
                    # split side only has 1 node
                    # create an empty tree
                    min_merge_st = min_st
                    max_merge_st = max_spl_st
                else:
                    # split max_spl_st
                    if min_side == "left":
                        max_merge_st = max_spl_st.right()
                        min_merge_st_node = sch.ClusterNode(
                            id=max_spl_st._node.id,
                            left=min_st._node,
                            right=max_spl_st._left,
                            dist=0,
                            count=min_st.count() + max_spl_st.left_count())
                    else:
                        max_merge_st = max_spl_st.left()
                        min_merge_st_node = sch.ClusterNode(
                            id=max_spl_st._node.id,
                            left=max_spl_st._right,
                            right=min_st._node,
                            dist=0,
                            count=min_st.count() + max_spl_st.right_count())
                    min_merge_st = HClustTree(min_merge_st_node, None)
                if min_side == "left":
                    merge_st_node = sch.ClusterNode(
                        id=max_st._node.id,
                        left=min_merge_st._node,
                        right=max_merge_st._node,
                        dist=0,
                        count=min_merge_st.count() + max_merge_st.count())
                else:
                    merge_st_node = sch.ClusterNode(
                        id=max_st._node.id,
                        left=max_merge_st._node,
                        right=min_merge_st._node,
                        dist=0,
                        count=min_merge_st.count() + max_merge_st.count())
                merge_st = HClustTree(merge_st_node, self)
                min_merge_st._prev = merge_st
                max_merge_st._prev = merge_st

                max_const_st._prev = self

                if max_const_st.count() >= merge_st.count():
                    max_st = max_const_st
                    min_st = merge_st
                else:
                    # max_st > min_st
                    max_st = merge_st
                    min_st = max_const_st
                    # reverse min side
                    min_side = "left" if min_side == "right" else "right"
            # obtained a min size cut
            if min_side == "left":
                lst = min_st
                rst = max_st
            else:
                lst = max_st
                rst = min_st
            # instance is modified.
            self._left = lst._node
            self._right = rst._node
        labs, sids = self.cluster_id_to_lab_list(
            [lst.leaf_ids(), rst.leaf_ids()], self.leaf_ids())
        if return_subtrees:
            return labs, sids, lst, rst
        else:
            return labs, sids

    def n_round_bipar_cnt(self, n):
        assert n > 0
        nr_bipar_cnt_list = []
        curr_hct_list = [self]
        curr_hct_cnt_list = []
        next_hct_list = []
        for i in range(n):
            for iter_hct in curr_hct_list:
                iter_left = iter_hct.left()
                iter_right = iter_hct.right()
                next_hct_list += [iter_left, iter_right]
                curr_hct_cnt_list += [iter_left.count(), iter_right.count()]
            nr_bipar_cnt_list.append(curr_hct_cnt_list)

            curr_hct_list = next_hct_list
            next_hct_list = []
            curr_hct_cnt_list = []
        return nr_bipar_cnt_list

    @staticmethod
    def cluster_id_to_lab_list(cl_sid_list, sid_list):
        """
        Convert nested clustered ID list into cluster label list.

        For example, convert `[[0, 1, 2], [3,4]]` to `[0, 0, 0, 1, 1]`
        according to id_arr `[0, 1, 2, 3, 4]`

        Parameters

        cl_sid_list: list[list[id]]
            Nested list with each sublist as a sert of IDs from a cluster.
        sid_list: list[id]
            Flat list of lists.

        """

        # checks uniqueness
        # This guarantees that clusters are all non-empty
        mtype.check_is_valid_sfids(sid_list)

        if type(cl_sid_list) != list:
            raise ValueError(
                "cl_sid_list must be a list: {}".format(cl_sid_list))

        for x in cl_sid_list:
            mtype.check_is_valid_sfids(x)

        cl_id_mlist = np.concatenate(cl_sid_list).tolist()
        mtype.check_is_valid_sfids(cl_id_mlist)

        # np.unique returns sorted unique values
        if sorted(sid_list) != sorted(cl_id_mlist):
            raise ValueError(
                "sid_list should have the same ids as cl_sid_list.")

        cl_ind_lut = {}
        # iter_cl_ind : cluster index
        # iter_cl_sids: individual cluster list
        for iter_cl_ind, iter_cl_sids in enumerate(cl_sid_list):
            for sid in iter_cl_sids:
                cl_ind_lut[sid] = iter_cl_ind

        lab_list = [cl_ind_lut[x] for x in sid_list]
        return lab_list, sid_list

    @staticmethod
    def hclust_linkage(dmat, linkage="complete", n_eval_rounds=None,
                       is_euc_dist=False, optimal_ordering=False,
                       verbose=False):
        dmat = np.array(dmat, dtype="float")
        dmat = SampleDistanceMatrix.num_correct_dist_mat(dmat)

        n = dmat.shape[0]

        if linkage == "auto":
            try_linkages = ("single", "complete", "average", "weighted")

            if is_euc_dist:
                try_linkages += ("centroid", "median", "ward")

            if n_eval_rounds is None:
                n_eval_rounds = int(np.ceil(np.log2(n)))
            else:
                n_eval_rounds = int(np.ceil(max(np.log2(n), n_eval_rounds)))

            ltype_mdl_list = []
            for iter_ltype in try_linkages:
                iter_lhct = HClustTree.hclust_tree(dmat, linkage=iter_ltype)
                iter_nbp_cnt_list = iter_lhct.n_round_bipar_cnt(n_eval_rounds)
                iter_nbp_mdl_arr = np.array(list(map(
                    lambda x: MultinomialMdl(np.array(x)).mdl,
                    iter_nbp_cnt_list)))
                iter_nbp_mdl = np.sum(
                    iter_nbp_mdl_arr / np.arange(1, n_eval_rounds + 1))
                ltype_mdl_list.append(iter_nbp_mdl)

            linkage = try_linkages[ltype_mdl_list.index(max(ltype_mdl_list))]

            if verbose:
                print(linkage, tuple(zip(try_linkages, ltype_mdl_list)),
                      sep="\n")

        dmat_sf = spspatial.distance.squareform(dmat)
        hac_z = sch.linkage(dmat_sf, method=linkage,
                            optimal_ordering=optimal_ordering)
        return hac_z

    @staticmethod
    def hclust_tree(dmat, linkage="complete", n_eval_rounds=None,
                    is_euc_dist=False, optimal_ordering=False, verbose=False):
        hac_z = HClustTree.hclust_linkage(
            dmat=dmat, linkage=linkage, n_eval_rounds=n_eval_rounds,
            is_euc_dist=is_euc_dist, optimal_ordering=optimal_ordering,
            verbose=verbose)
        return HClustTree.hct_from_lkg(hac_z)

    @staticmethod
    def hct_from_lkg(hac_z):
        return HClustTree(sch.to_tree(hac_z))

    @staticmethod
    def sort_x_by_d(x, dmat=None, metric="cosine", linkage="auto",
                    n_eval_rounds=None, optimal_ordering=False,
                    nprocs=None, verbose=False):
        dmat = SampleDistanceMatrix(x, d=dmat, metric=metric,
                                    nprocs=nprocs)._d
        xhct = HClustTree.hclust_tree(dmat, linkage="auto",
                                      is_euc_dist=(metric == "euclidean"),
                                      optimal_ordering=optimal_ordering)
        return xhct.leaf_ids()
