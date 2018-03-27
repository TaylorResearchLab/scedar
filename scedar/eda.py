import numpy as np
import scipy.spatial
import sklearn.manifold

import pandas as pd

import matplotlib as mpl
mpl.use("agg", warn=False)
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.patches
import matplotlib.gridspec

import seaborn as sns
sns.set(style="ticks")

import sklearn as skl
import sklearn.metrics

import warnings

from . import utils


class SampleFeatureMatrix(object):
    """
    SampleFeatureMatrix is a (n_samples, n_features) matrix.

    In this package, we are only interested in float features as measured
    expression levels.
    
    Parameters
    ----------
    x : ndarray or list
        data matrix (n_samples, n_features)
    sids : homogenous list of int or string
        sample ids. Should not contain duplicated elements.
    fids : homogenous list of int or string
        feature ids. Should not contain duplicated elements.

    Attributes
    ----------
    _x : ndarray
        data matrix (n_samples, n_features)
    _d : ndarray
        distance matrix (n_samples, n_samples)
    _sids : ndarray
        sample ids.
    _fids : ndarray
        sample ids.
    """

    def __init__(self, x, sids=None, fids=None):
        super(SampleFeatureMatrix, self).__init__()
        if x is None:
            raise ValueError("x cannot be None")
        else:
            try:
                x = np.array(x, dtype="float64")
            except ValueError as e:
                raise ValueError("Features must be float. {}".format(e))

            if x.ndim != 2:
                raise ValueError("x has shape (n_samples, n_features)")

            if x.size == 0:
                raise ValueError("size of x cannot be 0")

        if sids is None:
            sids = list(range(x.shape[0]))
        else:
            self.check_is_valid_sfids(sids)
            if len(sids) != x.shape[0]:
                raise ValueError("x has shape (n_samples, n_features)")

        if fids is None:
            fids = list(range(x.shape[1]))
        else:
            self.check_is_valid_sfids(fids)
            if len(fids) != x.shape[1]:
                raise ValueError("x has shape (n_samples, n_features)")

        self._x = x
        self._sids = np.array(sids)
        self._fids = np.array(fids)

    @staticmethod
    def is_valid_sfid(sfid):
        return (type(sfid) == str) or (type(sfid) == int)

    @staticmethod
    def check_is_valid_sfids(sfids):
        if sfids is None:
            raise ValueError("[sf]ids cannot be None")

        if type(sfids) != list:
            raise ValueError("[sf]ids must be a homogenous list of int or str")

        if len(sfids) == 0:
            raise ValueError("[sf]ids must have >= 1 values")

        sid_types = tuple(map(type, sfids))
        if len(set(sid_types)) != 1:
            raise ValueError("[sf]ids must be a homogenous list of int or str")

        if not SampleFeatureMatrix.is_valid_sfid(sfids[0]):
            raise ValueError("[sf]ids must be a homogenous list of int or str")

        sfids = np.array(sfids)
        assert sfids.ndim == 1
        assert sfids.shape[0] > 0
        if not utils.is_uniq_np1darr(sfids):
            raise ValueError("[sf]ids must not contain duplicated values")

    def s_id_to_ind(self, selected_sids):
        """
        Convert a list of sample IDs into sample indices.
        """
        sid_list = self.sids
        return [sid_list.index(i) for i in selected_sids]

    def f_id_to_ind(self, selected_fids):
        """
        Convert a list of feature IDs into feature indices.
        """
        fid_list = self.fids
        return [fid_list.index(i) for i in selected_fids]

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
        subset: SampleFeatureMatrix
        """
        if selected_s_inds is None:
            selected_s_inds = slice(None, None)

        if selected_f_inds is None:
            selected_f_inds = slice(None, None)

        return SampleFeatureMatrix(
            x=self._x[selected_s_inds, :][:, selected_f_inds].copy(),
            sids=self._sids[selected_s_inds].tolist(),
            fids=self._fids[selected_f_inds].tolist())

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
        subset: SampleFeatureMatrix
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

    def s_ind_x_pair(self, xs_ind, ys_ind, feature_filter=None):
        x = self._x[xs_ind, :]
        y = self._x[ys_ind, :]

        if feature_filter is None:
            f_inds = slice(None, None)
        else:
            if callable(feature_filter):
                f_inds = [feature_filter(ix, iy) for ix, iy in zip(x, y)]
            else:
                f_inds = feature_filter
        xf = x[f_inds]
        yf = y[f_inds]
        return xf, yf
        
    def s_ind_regression_scatter(self, xs_ind, ys_ind, feature_filter=None,
                                 xlab=None, ylab=None, title=None,
                                 **kwargs):
        """
        Regression plot on two samples with xs_ind and ys_ind.

        Parameters
        ----------
        xs_ind: int
            Sample index of x.
        ys_ind: int
            Sample index of y.
        feature_filter: bool array, or int array, or callable(x, y)
            If feature_filter is bool / int array, directly select features 
            with it. If feature_filter is callable, it will be applied on each 
            (x, y) value tuple.
        xlab: str
        ylab: str
        title: str
        """
        xf, yf = self.s_ind_x_pair(xs_ind, ys_ind, feature_filter)
        if xlab is None:
            xlab = self._sids[xs_ind]

        if ylab is None:
            ylab = self._sids[ys_ind]
        
        return regression_scatter(x=xf, y=yf, xlab=xlab, ylab=ylab, 
                                  title=title, **kwargs)

    def s_id_regression_scatter(self, xs_id, ys_id, feature_filter=None,
                                 xlab=None, ylab=None, title=None, **kwargs):
        """
        Regression plot on two samples with xs_id and ys_id.

        Parameters
        ----------
        xs_ind: int
            Sample ID of x.
        ys_ind: int
            Sample ID of y.
        feature_filter: bool array, or int array, or callable(x, y)
            If feature_filter is bool / int array, directly select features 
            with it. If feature_filter is callable, it will be applied on each 
            (x, y) value tuple.
        xlab: str
        ylab: str
        title: str
        """
        xs_ind, ys_ind = self.s_id_to_ind([xs_id, ys_id])
        return self.s_ind_regression_scatter(xs_ind, ys_ind, 
                                             feature_filter=feature_filter, 
                                             xlab=xlab, ylab=ylab, title=title, 
                                             **kwargs)

    def f_ind_x_pair(self, xf_ind, yf_ind, sample_filter=None):
        x = self._x[:, xf_ind]
        y = self._x[:, yf_ind]

        if sample_filter is None:
            s_inds = slice(None, None)
        else:
            if callable(sample_filter):
                s_inds = [sample_filter(ix, iy) for ix, iy in zip(x, y)]
            else:
                s_inds = sample_filter
        
        xf = x[s_inds]
        yf = y[s_inds]
        return (xf, yf)
    
    def f_ind_regression_scatter(self, xf_ind, yf_ind, sample_filter=None,
                                 xlab=None, ylab=None, title=None,
                                 **kwargs):
        """
        Regression plot on two features with xf_ind and yf_ind.

        Parameters
        ----------
        xf_ind: int
            Sample index of x.
        yf_ind: int
            Sample index of y.
        sample_filter: bool array, or int array, or callable(x, y)
            If sample_filter is bool / int array, directly select features 
            with it. If sample_filter is callable, it will be applied on each 
            (x, y) value tuple.
        xlab: str
        ylab: str
        title: str
        """
        xf, yf = self.f_ind_x_pair(xf_ind, yf_ind, sample_filter)
        if xlab is None:
            xlab = self._fids[xf_ind]

        if ylab is None:
            ylab = self._fids[yf_ind]
        
        return regression_scatter(x=xf, y=yf, xlab=xlab, ylab=ylab, 
                                  title=title, **kwargs)

    def f_id_regression_scatter(self, xf_id, yf_id, sample_filter=None,
                                 xlab=None, ylab=None, title=None, **kwargs):
        """
        Regression plot on two features with xf_id and yf_id.

        Parameters
        ----------
        xf_id: int
            Sample ID of x.
        yf_ind: int
            Sample ID of y.
        sample_filter: bool array, or int array, or callable(x, y)
            If sample_filter is bool / int array, directly select features 
            with it. If sample_filter is callable, it will be applied on each 
            (x, y) value tuple.
        xlab: str
        ylab: str
        title: str
        """
        xf_ind, yf_ind = self.f_id_to_ind([xf_id, yf_id])
        return self.f_ind_regression_scatter(xf_ind, yf_ind, 
                                             sample_filter=sample_filter, 
                                             xlab=xlab, ylab=ylab, title=title, 
                                             **kwargs)

    def s_ind_x_vec(self, s_ind, feature_filter=None):
        """
        Access a single vector of a sample.         
        """
        x = self._x[s_ind, :]
        if feature_filter is None:
            f_inds = slice(None, None)
        else:
            if callable(feature_filter):
                f_inds = [feature_filter(ix) for ix in x]
            else:
                f_inds = feature_filter
        xf = x[f_inds]
        return xf
    
    def s_ind_dist(self, s_ind, feature_filter=None, xlab=None, ylab=None,
                   title=None, figsize=(5, 5), ax=None, **kwargs):
        xf = self.s_ind_x_vec(s_ind, feature_filter)
        return hist_dens_plot(xf, title=title, xlab=xlab, ylab=ylab, 
                              figsize=figsize, ax=ax, **kwargs)

    def s_id_dist(self, s_id, feature_filter=None, xlab=None, ylab=None,
                  title=None, figsize=(5, 5), ax=None, **kwargs):
        s_ind = self.s_id_to_ind([s_id])[0]
        return self.s_ind_dist(s_ind, title=title, xlab=xlab, ylab=ylab,
                               figsize=figsize, ax=ax, **kwargs)

    def f_ind_x_vec(self, f_ind, sample_filter=None):
        """
        Access a single vector of a sample.         
        """
        x = self._x[:, f_ind]
        if sample_filter is None:
            s_inds = slice(None, None)
        else:
            if callable(sample_filter):
                s_inds = [sample_filter(ix) for ix in x]
            else:
                s_inds = sample_filter
        xf = x[s_inds]
        return xf
    
    def f_ind_dist(self, f_ind, sample_filter=None, xlab=None, ylab=None,
                   title=None, figsize=(5, 5), ax=None, **kwargs):
        xf = self.f_ind_x_vec(f_ind, sample_filter)
        return hist_dens_plot(xf, title=title, xlab=xlab, ylab=ylab, 
                              figsize=figsize, ax=ax, **kwargs)

    def f_id_dist(self, f_id, sample_filter=None, xlab=None, ylab=None,
                  title=None, figsize=(5, 5), ax=None, **kwargs):
        f_ind = self.f_id_to_ind([f_id])[0]
        return self.f_ind_dist(f_ind, title=title, xlab=xlab, ylab=ylab,
                               figsize=figsize, ax=ax, **kwargs)

    @property
    def sids(self):
        return self._sids.tolist()

    @property
    def fids(self):
        return self._fids.tolist()

    @property
    def x(self):
        return self._x.tolist()


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
    """

    def __init__(self, x, d=None, metric="correlation",
                 sids=None, fids=None, nprocs=None):
        super(SampleDistanceMatrix, self).__init__(x=x, sids=sids, fids=fids)

        if d is not None:
            try:
                d = np.array(d, dtype="float64")
            except ValueError as e:
                raise ValueError("d must be float. {}".format(e))

            if ((d.ndim != 2) or
                (d.shape[0] != d.shape[1]) or
                (d.shape[0] != self._x.shape[0])):
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

    # numerically correct dmat
    @staticmethod
    def num_correct_dist_mat(dmat, upper_bound=None):
        if ((not isinstance(dmat, np.ndarray)) or
            (dmat.ndim != 2) or
            (dmat.shape[0] != dmat.shape[1])):
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

    # store_res : bool. Wheter to keep the tsne results in a dictionalry keyed
    # by the parameters.
    def tsne(self, store_res=True, **kwargs):
        # check input args
        if ("metric" in kwargs
            and kwargs["metric"] not in ("precomputed", self._metric)):
            raise ValueError("If you want to calculate t-SNE of a different "
                             "metric than the instance metric, create another "
                             "instance of the desired metric.")
        else:
            kwargs["metric"] = "precomputed"
        # look for cached tsne with param key
        curr_store_ind = len(self._tsne_lut) + 1
        tsne_params_key = (str(kwargs), curr_store_ind)
        for key, val in self._tsne_lut.items():
            if key[0] == tsne_params_key[0]:
                tsne_res = val
                break
        else:
            tsne_res = tsne(self._d, **kwargs)

        if store_res:
            tsne_res_copy = tsne_res.copy()
            self._tsne_lut[tsne_params_key] = tsne_res_copy
            self._lazy_load_last_tsne = tsne_res_copy

        return tsne_res

    def tsne_gradient_plot(self, gradient=None, labels=None, 
                           title=None, xlab=None, ylab=None,
                           figsize=(20, 20), add_legend=True, 
                           n_txt_per_cluster=3, alpha=1, s=0.5,
                           random_state=None, **kwargs):
        """
        Plot the last t-SNE projection with the provided gradient as color.
        """
        return cluster_scatter(self._last_tsne, labels=labels,
                               gradient=gradient,
                               title=title, xlab=xlab, ylab=ylab,
                               figsize=figsize, add_legend=add_legend,
                               n_txt_per_cluster=n_txt_per_cluster,
                               alpha=alpha, s=s, random_state=random_state,
                               **kwargs)

    def tsne_feature_gradient_plot(self, fid, labels=None,
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
        """
        if SampleFeatureMatrix.is_valid_sfid(fid):
            fid = [fid]
            f_ind = self.f_id_to_ind(fid)[0]
        else:
            raise ValueError("Invalid fid {}."
                             "Currently only support 1 "
                             "feature gradient plot.".format(fid))
            
        fx = self._x[:, f_ind]
        return cluster_scatter(self._last_tsne, labels=labels,
                               gradient=fx,
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

        return SampleDistanceMatrix(
            x=self._x[selected_s_inds, :][:, selected_f_inds].copy(),
            d=self._d[selected_s_inds, :][:, selected_s_inds].copy(),
            metric=self._metric,
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

    @property
    def d(self):
        return self._d.tolist()

    @property
    def _d(self):
        if self._lazy_load_d is None:
            self._lazy_load_d = self.num_correct_dist_mat(
                skl.metrics.pairwise.pairwise_distances(self._x,
                                                        metric=self._metric,
                                                        n_jobs=self._nprocs))
        return self._lazy_load_d

    @property
    def _last_tsne(self):
        if self._lazy_load_last_tsne is None:
            self._lazy_load_last_tsne = self.tsne()
        return self._lazy_load_last_tsne

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


class SingleLabelClassifiedSamples(SampleDistanceMatrix):
    """SingleLabelClassifiedSamples"""
    # sid, lab, fid, x

    def __init__(self, x, labs, sids=None, fids=None, d=None,
                 metric="correlation", nprocs=None):
        # sids: sample IDs. String or int.
        # labs: sample classified labels. String or int.
        # x: (n_samples, n_features)
        super(SingleLabelClassifiedSamples, self).__init__(x=x, d=d,
                                                           metric=metric,
                                                           sids=sids, fids=fids,
                                                           nprocs=nprocs)
        self.check_is_valid_labs(labs)
        labs = np.array(labs)
        if self._sids.shape[0] != labs.shape[0]:
            raise ValueError("sids must have the same length as labs")
        self._labs = labs

        self._uniq_labs, self._uniq_lab_cnts = np.unique(labs,
                                                         return_counts=True)

        sid_lut = {}
        for uniq_lab in self._uniq_labs:
            sid_lut[uniq_lab] = self._sids[labs == uniq_lab]
        self._sid_lut = sid_lut

        lab_lut = {}
        # sids only contain unique values
        for i in range(self._sids.shape[0]):
            lab_lut[self._sids[i]] = labs[i]
        self._lab_lut = lab_lut
        return

    @staticmethod
    def is_valid_lab(lab):
        return (type(lab) == str) or (type(lab) == int)

    @staticmethod
    def check_is_valid_labs(labs):
        if labs is None:
            raise ValueError("labs cannot be None")

        if type(labs) != list:
            raise ValueError("labs must be a homogenous list of int or str")

        if len(labs) == 0:
            raise ValueError("labs cannot be empty")

        if len(set(map(type, labs))) != 1:
            raise ValueError("labs must be a homogenous list of int or str")

        if not SingleLabelClassifiedSamples.is_valid_lab(labs[0]):
            raise ValueError("labs must be a homogenous list of int or str")

        labs = np.array(labs)
        assert labs.ndim == 1, "Labels must be 1D"
        assert labs.shape[0] > 0

    def filter_min_class_n(self, min_class_n):
        uniq_lab_cnts = np.unique(self._labs, return_counts=True)
        nf_sid_ind = np.in1d(
            self._labs, (uniq_lab_cnts[0])[uniq_lab_cnts[1] >= min_class_n])
        return self.ind_x(nf_sid_ind)

    def labs_to_sids(self, labs):
        return tuple(tuple(self._sid_lut[y].tolist()) for y in labs)

    def sids_to_labs(self, sids):
        return np.array([self._lab_lut[x] for x in sids])

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
        subset: SingleLabelClassifiedSamples
        """
        if selected_s_inds is None:
            selected_s_inds = slice(None, None)

        if selected_f_inds is None:
            selected_f_inds = slice(None, None)

        return SingleLabelClassifiedSamples(
            x=self._x[selected_s_inds, :][:, selected_f_inds].copy(),
            labs=self._labs[selected_s_inds].tolist(),
            d=self._d[selected_s_inds, :][:, selected_s_inds].copy(),
            sids=self._sids[selected_s_inds].tolist(),
            fids=self._fids[selected_f_inds].tolist(),
            metric=self._metric, nprocs=self._nprocs)

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
        subset: SingleLabelClassifiedSamples
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

    def tsne_gradient_plot(self, gradient=None, labels=None, 
                           title=None, xlab=None, ylab=None,
                           figsize=(20, 20), add_legend=True, 
                           n_txt_per_cluster=3, alpha=1, s=0.5,
                           random_state=None, **kwargs):
        """
        Plot the last t-SNE projection with the provided gradient as color.
        """
        return super(SingleLabelClassifiedSamples, 
                     self).tsne_gradient_plot(
                        labels=self.labs, gradient=gradient,
                        title=title, xlab=xlab, ylab=ylab,
                        figsize=figsize, 
                        add_legend=add_legend, 
                        n_txt_per_cluster=n_txt_per_cluster, 
                        alpha=alpha, s=s, 
                        random_state=random_state, 
                        **kwargs)

    def tsne_feature_gradient_plot(self, fid, labels=None, 
                                   title=None, xlab=None, ylab=None,
                                   figsize=(20, 20), add_legend=True, 
                                   n_txt_per_cluster=3, alpha=1, s=0.5,
                                   random_state=None, **kwargs):
        """
        Plot the last t-SNE projection with the provided gradient as color.
        """
        return super(SingleLabelClassifiedSamples, 
                     self).tsne_feature_gradient_plot(
                        fid=fid, labels=self.labs,
                        title=title, xlab=xlab, ylab=ylab,
                        figsize=figsize, 
                        add_legend=add_legend, 
                        n_txt_per_cluster=n_txt_per_cluster, 
                        alpha=alpha, s=s, 
                        random_state=random_state, 
                        **kwargs)

    @property
    def labs(self):
        return self._labs.tolist()

    # Sort the clustered sample_ids with the reference order of another.
    #
    # Sort sids according to labs
    # If ref_sid_order is not None:
    #   sort sids further according to ref_sid_order
    def lab_sorted_sids(self, ref_sid_order=None):
        sep_lab_sid_list = []
        sep_lab_list = []
        for iter_lab in sorted(self._sid_lut.keys()):
            iter_sid_arr = self._sid_lut[iter_lab]
            sep_lab_sid_list.append(iter_sid_arr)
            sep_lab_list.append(np.repeat(iter_lab, len(iter_sid_arr)))

        if ref_sid_order is not None:
            self.check_is_valid_sfids(ref_sid_order)
            ref_sid_order = np.array(ref_sid_order)
            # sort r according to q
            # assumes:
            # - r contains all elements in q
            # - r is 1d np array

            def sort_flat_sids(query_sids, ref_sids):
                return ref_sids[np.in1d(ref_sids, query_sids)]

            # sort inner sid list but maintains the order as sep_lab_list
            sep_lab_sid_list = [sort_flat_sids(x, ref_sid_order)
                                for x in sep_lab_sid_list]
            sep_lab_min_sid_list = [x[0] for x in sep_lab_sid_list]
            sorted_sep_lab_min_sid_list = list(
                sort_flat_sids(sep_lab_min_sid_list, ref_sid_order))
            min_sid_sorted_sep_lab_ind_list = [sep_lab_min_sid_list.index(x)
                                               for x in sorted_sep_lab_min_sid_list]
            sep_lab_list = [sep_lab_list[i]
                            for i in min_sid_sorted_sep_lab_ind_list]
            sep_lab_sid_list = [sep_lab_sid_list[i]
                                for i in min_sid_sorted_sep_lab_ind_list]

        lab_sorted_sid_arr = np.concatenate(sep_lab_sid_list)
        lab_sorted_lab_arr = np.concatenate(sep_lab_list)

        # check sorted sids are the same set as original
        np.testing.assert_array_equal(
            np.sort(lab_sorted_sid_arr), np.sort(self._sids))
        # check sorted labs are the same set as original
        np.testing.assert_array_equal(
            np.sort(lab_sorted_lab_arr), np.sort(self._labs))
        # check sorted (sid, lab) matchings are the same set as original
        np.testing.assert_array_equal(lab_sorted_lab_arr[np.argsort(lab_sorted_sid_arr)],
                                      self._labs[np.argsort(self._sids)])

        return (lab_sorted_sid_arr, lab_sorted_lab_arr)

    # See how two clustering criteria match with each other.
    # When given q_slc_samples is not None, sids and labs are ignored.
    # When q_slc_samples is None, sids and labs must be provided
    def cross_labs(self, q_slc_samples):
        if not isinstance(q_slc_samples, SingleLabelClassifiedSamples):
            raise TypeError("Query should be an instance of "
                            "SingleLabelClassifiedSamples")

        try:
            ref_labs = np.array([self._lab_lut[x]
                                 for x in q_slc_samples.sids])
        except KeyError as e:
            raise ValueError("query sid {} is not in ref sids.".format(e))

        query_labs = np.array(q_slc_samples.labs)

        uniq_rlabs, uniq_rlab_cnts = np.unique(ref_labs, return_counts=True)
        cross_lab_lut = {}
        for i in range(len(uniq_rlabs)):
            # ref cluster i. query unique labs.
            ref_ci_quniq = tuple(map(list, np.unique(
                query_labs[np.where(np.array(ref_labs) == uniq_rlabs[i])],
                return_counts=True)))
            cross_lab_lut[uniq_rlabs[i]] = (uniq_rlab_cnts[i],
                                            tuple(map(tuple, ref_ci_quniq)))

        return cross_lab_lut

    @staticmethod
    def labs_to_cmap(labels, return_lut=False):
        SingleLabelClassifiedSamples.check_is_valid_labs(labels)

        labels = np.array(labels)
        uniq_lab_arr = np.unique(labels)
        num_uniq_labs = len(uniq_lab_arr)

        lab_col_list = sns.hls_palette(num_uniq_labs)
        lab_cmap = mpl.colors.ListedColormap(lab_col_list)

        if return_lut:
            uniq_lab_lut = dict(zip(range(num_uniq_labs), uniq_lab_arr))
            uniq_ind_lut = dict(zip(uniq_lab_arr, range(num_uniq_labs)))

            lab_ind_arr = np.array([uniq_ind_lut[x] for x in labels])

            lab_col_lut = dict(zip([uniq_lab_lut[i]
                                    for i in range(len(uniq_lab_arr))],
                                   lab_col_list))
            return (lab_cmap, lab_ind_arr, lab_col_lut, uniq_lab_lut)
        else:
            return lab_cmap


def cluster_scatter(projection2d, labels=None, gradient=None, 
                    title=None, xlab=None, ylab=None,
                    figsize=(20, 20), add_legend=True, n_txt_per_cluster=3,
                    alpha=1, s=0.5, random_state=None, **kwargs):
    projection2d = np.array(projection2d, dtype="float")

    if (projection2d.ndim != 2) or (projection2d.shape[1] != 2):
        raise ValueError("projection2d matrix should have shape "
                         "(n_samples, 2). {}".format(projection2d))

    # TODO: optimize the if-else statement
    if labels is not None:
        SingleLabelClassifiedSamples.check_is_valid_labs(labels)
        labels = np.array(labels)
        if labels.shape[0] != projection2d.shape[0]:
            raise ValueError(
                "nrow(projection2d matrix) should be equal to len(labels)")
        uniq_labels = np.unique(labels)

        if gradient is not None:
            cmap = kwargs.pop("cmap", "viridis")
            plt.figure(figsize=figsize)
            # matplotlib.collections.PathCollection
            mpc = plt.scatter(x=projection2d[:, 0], y=projection2d[:, 1], 
                              c=gradient, cmap=cmap, s=s, alpha=alpha, 
                              **kwargs)
            if add_legend:
                plt.colorbar(mpc)
            fig = mpc.get_figure()
            ax = fig.get_axes()[0]
        else:
            fig, ax = plt.subplots(figsize=figsize)
            color_lut = dict(zip(uniq_labels,
                                 sns.color_palette("hls", len(uniq_labels))))
            ax.scatter(x=projection2d[:, 0], y=projection2d[:, 1],
                       c=[color_lut[lab] for lab in labels],
                       s=s, alpha=alpha, **kwargs)
            # Add legend
            # Shrink current axis by 20%
            if add_legend:
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                ax.legend(handles=[mpl.patches.Patch(color=color_lut[ulab], 
                                                     label=ulab)
                                   for ulab in uniq_labels],
                          bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        # randomly select labels for annotation
        if random_state is not None:
            np.random.seed(random_state)

        anno_ind = np.concatenate(
            [np.random.choice(np.where(labels == ulab)[0], n_txt_per_cluster)
             for ulab in uniq_labels])

        for i in map(int, anno_ind):
            ax.annotate(labels[i], (projection2d[i, 0], projection2d[i, 1]))
    else:
        if gradient is None:
            fig, ax = plt.subplots(figsize=figsize)
            ax.scatter(x=projection2d[:, 0], y=projection2d[:, 1], s=s,
                       alpha=alpha, **kwargs)
        else:
            cmap = kwargs.pop("cmap", "viridis")
            plt.clf()
            # matplotlib.collections.PathCollection
            mpc = plt.scatter(x=projection2d[:, 0], y=projection2d[:, 1], 
                              c=gradient, cmap=cmap, s=s, alpha=alpha, 
                              **kwargs)
            if add_legend:
                plt.colorbar(mpc)
            fig = mpc.get_figure()
            ax = fig.get_axes()[0]

    if title is not None:
        ax.set_title(title)

    if xlab is not None:
        ax.set_xlabel(xlab)

    if ylab is not None:
        ax.set_ylabel(ylab)
    plt.close()
    return fig

def regression_scatter(x, y, title=None, xlab=None, ylab=None, 
                       figsize=(5, 5), alpha=1, s=0.5, ax=None, **kwargs):
    """
    Paired vector scatter plot.
    """
    if xlab is not None:
        x = pd.Series(x, name=xlab)

    if ylab is not None:
        y = pd.Series(y, name=ylab)

    # initialize a new figure
    if ax is None:
        _, ax = plt.subplots()

    ax = sns.regplot(x=x, y=y, ax=ax, **kwargs)

    fig = ax.get_figure()

    if title is not None:
        ax.set_title(title)

    if xlab is not None:
        ax.set_xlabel(xlab)

    if ylab is not None:
        ax.set_ylabel(ylab)

    fig.set_size_inches(*figsize)
    plt.close()
    return fig

def hist_dens_plot(x, title=None, xlab=None, ylab=None, figsize=(5, 5), 
                   ax=None, **kwargs):
    """
    Plot histogram and density plot of x.
    """
    if ax is None:
        _, ax = plt.subplots()

    ax = sns.distplot(x, norm_hist=None, ax=ax, **kwargs)

    fig = ax.get_figure()

    if title is not None:
        ax.set_title(title)

    if xlab is not None:
        ax.set_xlabel(xlab)

    if ylab is not None:
        ax.set_ylabel(ylab)

    fig.set_size_inches(*figsize)
    plt.close()
    return fig

def heatmap(x, row_labels=None, col_labels=None, title=None, xlab=None,
            ylab=None, figsize=(20, 20), **kwargs):
    x = np.array(x, dtype="float")
    if x.ndim != 2:
        raise ValueError("x should be 2D array. {}".format(x))

    if x.size == 0:
        raise ValueError("x cannot be empty.")

    if row_labels is not None:
        SingleLabelClassifiedSamples.check_is_valid_labs(row_labels)
        if len(row_labels) != x.shape[0]:
            raise ValueError("length of row_labels should be the same as the "
                             "number of rows in x."
                             " row_labels: {}. x: {}".format(len(row_labels),
                                                             x.shape))

    if col_labels is not None:
        SingleLabelClassifiedSamples.check_is_valid_labs(col_labels)
        if len(col_labels) != x.shape[1]:
            raise ValueError("length of col_labels should be the same as the "
                             "number of rows in x."
                             " col_labels: {}. x: {}".format(len(col_labels),
                                                             x.shape))

    if "interpolation" not in kwargs:
        kwargs["interpolation"] = "nearest"

    fig = plt.figure(figsize=figsize)
    if title is not None:
        fig.suptitle(title)

    # outer 2x2 grid
    gs = mpl.gridspec.GridSpec(2, 2,
                               width_ratios=[1, 4],
                               height_ratios=[1, 4],
                               wspace=0.0, hspace=0.0)

    # inner upper right for color labels and legends
    ur_gs = mpl.gridspec.GridSpecFromSubplotSpec(2, 1,
                                                 height_ratios=[3, 1],
                                                 subplot_spec=gs[1],
                                                 wspace=0.0, hspace=0.0)

    # inner lower left for color labels and legends
    ll_gs = mpl.gridspec.GridSpecFromSubplotSpec(1, 2,
                                                 width_ratios=[3, 1],
                                                 subplot_spec=gs[2],
                                                 wspace=0.0, hspace=0.0)

    ax_lut = {
        'cb_ax': plt.subplot(gs[0]),
        'hm_ax': plt.subplot(gs[3]),
        'lcol_ax': plt.subplot(ll_gs[1]),
        'ucol_ax': plt.subplot(ur_gs[1]),
        'llgd_ax': plt.subplot(ll_gs[0]),
        'ulgd_ax': plt.subplot(ur_gs[0])
    }

    # remove frames and ticks
    for iax in ax_lut.values():
        iax.set_xticks([])
        iax.set_yticks([])
        iax.axis('off')

    # lower right heatmap
    imgp = ax_lut['hm_ax'].imshow(x, cmap='magma', aspect='auto', **kwargs)
    if xlab is not None:
        ax_lut['hm_ax'].set_xlabel(xlab)

    if ylab is not None:
        ax_lut['hm_ax'].set_ylabel(ylab)

    # upper left colorbar
    cb = plt.colorbar(imgp, cax=ax_lut['cb_ax'])
    ax_lut['cb_ax'].set_aspect(5, anchor='W')
    ax_lut['cb_ax'].yaxis.tick_left()
    ax_lut['cb_ax'].axis('on')

    # color labels and legends
    ax_lut['ucol_ax'].set_anchor('S')
    ax_lut['lcol_ax'].set_anchor('E')
    col_axs = (ax_lut['ucol_ax'], ax_lut['lcol_ax'])
    lgd_axs = (ax_lut['ulgd_ax'], ax_lut['llgd_ax'])
    cr_labs = (col_labels, row_labels)
    for i in range(2):
        if cr_labs[i] is not None:
            cmap, ind, ulab_col_lut, ulab_lut = SingleLabelClassifiedSamples.labs_to_cmap(
                cr_labs[i], return_lut=True)
            if i == 0:
                # col color labels
                ind_mat = ind.reshape(1, -1)
            else:
                # row color labels
                ind_mat = ind.reshape(-1, 1)
            col_axs[i].imshow(ind_mat, cmap=cmap, aspect='auto', **kwargs)

            lgd_patches = [mpl.patches.Patch(color=ulab_col_lut[ulab],
                                             label=ulab)
                           for ulab in sorted(ulab_lut.values())]

            if i == 0:
                # col color legend
                lgd_axs[i].legend(handles=lgd_patches, loc="center", ncol=6)
            else:
                # row color legend
                lgd_axs[i].legend(handles=lgd_patches, loc="upper center",
                                  ncol=1)
    plt.close()
    return fig
