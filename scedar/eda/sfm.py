import numpy as np

import scipy.sparse as spsp

from sklearn.preprocessing import StandardScaler

from scedar import utils

from scedar.eda.plot import regression_scatter
from scedar.eda.plot import hist_dens_plot
from scedar.eda import mtype
from scedar.eda import stats


class SampleFeatureMatrix(object):
    """
    SampleFeatureMatrix is a (n_samples, n_features) matrix.

    In this package, we are only interested in float features as measured
    expression levels.

    Parameters
    ----------
    x : {array-like, sparse matrix}
        data matrix (n_samples, n_features)
    sids : homogenous list of int or string
        sample ids. Should not contain duplicated elements.
    fids : homogenous list of int or string
        feature ids. Should not contain duplicated elements.

    Attributes
    ----------
    _x : {array-like, sparse matrix}
        data matrix (n_samples, n_features)
    _is_sparse: boolean
        whether the data matrix is sparse matrix or not
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
            if spsp.issparse(x):
                x = spsp.csr_matrix(x, dtype="float64")
            else:
                try:
                    x = np.array(x, copy=False, dtype="float64")
                except ValueError as e:
                    raise ValueError("Features must be float. {}".format(e))

            if x.ndim != 2:
                raise ValueError("x has shape (n_samples, n_features)")

        if sids is None:
            sids = list(range(x.shape[0]))
        else:
            mtype.check_is_valid_sfids(sids)
            if len(sids) != x.shape[0]:
                raise ValueError("x has shape (n_samples, n_features)")

        if fids is None:
            fids = list(range(x.shape[1]))
        else:
            mtype.check_is_valid_sfids(fids)
            if len(fids) != x.shape[1]:
                raise ValueError("x has shape (n_samples, n_features)")

        self._x = x
        self._sids = np.array(sids)
        self._fids = np.array(fids)

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
        selected_sids: id array
            ID array of selected samples. If is None, select all.
        selected_fids: id array
            ID array of selected features. If is None, select all.

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

    @staticmethod
    def filter_1d_inds(f, x):
        # f_inds filtered index
        if f is None:
            f_inds = slice(None, None)
        else:
            if callable(f):
                f_inds = [f(ix) for ix in x]
            else:
                f_inds = f
        return f_inds

    def s_ind_x_pair(self, xs_ind, ys_ind, feature_filter=None):
        x = self._x[xs_ind, :]
        y = self._x[ys_ind, :]
        if self._is_sparse:
            x = x.todense().A1
            y = y.todense().A1
        if callable(feature_filter):
            f_inds = self.filter_1d_inds(
                lambda pair: feature_filter(pair[0], pair[1]), zip(x, y))
        else:
            f_inds = self.filter_1d_inds(feature_filter, zip(x, y))
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
        if self._is_sparse:
            x = x.todense().A1
            y = y.todense().A1
        if callable(sample_filter):
            s_inds = self.filter_1d_inds(
                lambda pair: sample_filter(pair[0], pair[1]), zip(x, y))
        else:
            s_inds = self.filter_1d_inds(sample_filter, zip(x, y))
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
        if self._is_sparse:
            x = x.todense().A1
        f_inds = self.filter_1d_inds(feature_filter, x)
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
        return self.s_ind_dist(s_ind, feature_filter=feature_filter,
                               title=title, xlab=xlab, ylab=ylab,
                               figsize=figsize, ax=ax, **kwargs)

    def f_ind_x_vec(self, f_ind, sample_filter=None, transform=None):
        """
        Access a single vector of a sample.
        """
        x = self._x[:, f_ind]
        if self._is_sparse:
            x = x.todense().A1
        s_inds = self.filter_1d_inds(sample_filter, x)
        xf = x[s_inds]
        if transform is not None:
            if callable(transform):
                xf = np.array(list(map(transform, xf)))
            else:
                raise ValueError("transform must be a callable")
        return xf

    def f_id_x_vec(self, f_id, sample_filter=None):
        f_ind = self.f_id_to_ind([f_id])[0]
        return self.f_ind_x_vec(f_ind, sample_filter=sample_filter)

    def f_ind_dist(self, f_ind, sample_filter=None, xlab=None, ylab=None,
                   title=None, figsize=(5, 5), ax=None, **kwargs):
        xf = self.f_ind_x_vec(f_ind, sample_filter)
        return hist_dens_plot(xf, title=title, xlab=xlab, ylab=ylab,
                              figsize=figsize, ax=ax, **kwargs)

    def f_id_dist(self, f_id, sample_filter=None, xlab=None, ylab=None,
                  title=None, figsize=(5, 5), ax=None, **kwargs):
        f_ind = self.f_id_to_ind([f_id])[0]
        return self.f_ind_dist(f_ind, sample_filter=sample_filter,
                               title=title, xlab=xlab, ylab=ylab,
                               figsize=figsize, ax=ax, **kwargs)

    def f_sum(self, f_sum_filter=None):
        """
        For each sample, compute the sum of all features.

        Returns
        -------
        rowsum: float array
            (filtered_n_samples,)
        """
        rowsum = self._x.sum(axis=1)
        if self._is_sparse:
            rowsum = rowsum.A1
        s_inds = self.filter_1d_inds(f_sum_filter, rowsum)
        rowsumf = rowsum[s_inds]
        return rowsumf

    def f_sum_dist(self, f_sum_filter=None, xlab=None, ylab=None,
                   title=None, figsize=(5, 5), ax=None, **kwargs):
        """
        Plot the distribution of the feature sum of each sample, (n_samples,).
        """
        xf = self.f_sum(f_sum_filter)
        return hist_dens_plot(xf, title=title, xlab=xlab, ylab=ylab,
                              figsize=figsize, ax=ax, **kwargs)

    def s_sum(self, s_sum_filter=None):
        """
        For each feature, computer the sum of all samples.

        Returns
        -------
        xf: float array
            (filtered_n_features,)
        """
        colsum = self._x.sum(axis=0)
        if self._is_sparse:
            colsum = colsum.A1
        f_inds = self.filter_1d_inds(s_sum_filter, colsum)
        colsumf = colsum[f_inds]
        return colsumf

    def s_sum_dist(self, s_sum_filter=None, xlab=None, ylab=None,
                   title=None, figsize=(5, 5), ax=None, **kwargs):
        """
        Plot the distribution of the sample sum of each feature, (n_features,).
        """
        xf = self.s_sum(s_sum_filter)
        return hist_dens_plot(xf, title=title, xlab=xlab, ylab=ylab,
                              figsize=figsize, ax=ax, **kwargs)

    def f_cv(self, f_cv_filter=None):
        """
        For each sample, compute the coefficient of variation of all features.

        Returns
        -------
        xf: float array
            (filtered_n_samples,)
        """
        if self._x.shape[1] == 0:
            return np.repeat(np.nan, self._x.shape[0])
        ss = StandardScaler(with_mean=False).fit(self._x.T)
        n_fts = self._x.shape[1]
        rowsd = np.sqrt(ss.var_ * (n_fts / (n_fts - 1)))
        rowmean = ss.mean_
        rowcv = rowsd / rowmean
        s_inds = self.filter_1d_inds(f_cv_filter, rowcv)
        rowcvf = rowcv[s_inds]
        return rowcvf

    def f_cv_dist(self, f_cv_filter=None, xlab=None, ylab=None,
                  title=None, figsize=(5, 5), ax=None, **kwargs):
        """
        Plot the distribution of the feature sum of each sample, (n_samples,).
        """
        xf = self.f_cv(f_cv_filter)
        return hist_dens_plot(xf, title=title, xlab=xlab, ylab=ylab,
                              figsize=figsize, ax=ax, **kwargs)

    def s_cv(self, s_cv_filter=None):
        """
        For each feature, compute the coefficient of variation of all samples.

        Returns
        -------
        xf: float array
            (n_features,)
        """
        if self._x.shape[1] == 0:
            return np.array([])
        ss = StandardScaler(with_mean=False).fit(self._x)
        n_sps = self._x.shape[0]
        colsd = np.sqrt(ss.var_ * (n_sps / (n_sps - 1)))
        colmean = ss.mean_
        colcv = colsd / colmean
        f_inds = self.filter_1d_inds(s_cv_filter, colcv)
        colcvf = colcv[f_inds]
        return colcvf

    def s_cv_dist(self, s_cv_filter=None, xlab=None, ylab=None,
                  title=None, figsize=(5, 5), ax=None, **kwargs):
        """
        Plot the distribution of the sample coefficient of variation
        of each feature, (n_features,).
        """
        xf = self.s_cv(s_cv_filter)
        return hist_dens_plot(xf, title=title, xlab=xlab, ylab=ylab,
                              figsize=figsize, ax=ax, **kwargs)

    def f_n_above_threshold(self, closed_threshold):
        """
        For each sample, compute the number of features above a closed
        threshold.
        """
        row_ath_sum = (self._x >= closed_threshold).sum(axis=1)
        if self._is_sparse:
            row_ath_sum = row_ath_sum.A1
        return row_ath_sum

    def f_n_above_threshold_dist(self, closed_threshold, xlab=None, ylab=None,
                                 title=None, figsize=(5, 5), ax=None,
                                 **kwargs):
        """
        Plot the distribution of the the number of above threshold samples
        of each feature, (n_features,).
        """
        xf = self.f_n_above_threshold(closed_threshold)
        return hist_dens_plot(xf, title=title, xlab=xlab, ylab=ylab,
                              figsize=figsize, ax=ax, **kwargs)

    def s_n_above_threshold(self, closed_threshold):
        """
        For each feature, compute the number of samples above a closed
        threshold.
        """
        col_ath_sum = (self._x >= closed_threshold).sum(axis=0)
        if self._is_sparse:
            col_ath_sum = col_ath_sum.A1
        return col_ath_sum

    def s_n_above_threshold_dist(self, closed_threshold, xlab=None, ylab=None,
                                 title=None, figsize=(5, 5), ax=None,
                                 **kwargs):
        """
        Plot the distribution of the the number of above threshold samples
        of each feature, (n_features,).
        """
        xf = self.s_n_above_threshold(closed_threshold)
        return hist_dens_plot(xf, title=title, xlab=xlab, ylab=ylab,
                              figsize=figsize, ax=ax, **kwargs)

    def f_gc(self, f_gc_filter=None):
        """
        For each sample, compute the Gini coefficients of all features.

        Returns
        -------
        xf: float array
            (filtered_n_samples,)
        """
        rowgc = []
        for i in range(self._x.shape[0]):
            if self._is_sparse:
                i_x = self._x[i, :].todense().A1
            else:
                i_x = self._x[i, :]
            rowgc.append(stats.gc1d(i_x))
        rowgc = np.array(rowgc)
        s_inds = self.filter_1d_inds(f_gc_filter, rowgc)
        rowgcf = rowgc[s_inds]
        return rowgcf

    def f_gc_dist(self, f_gc_filter=None, xlab=None, ylab=None,
                  title=None, figsize=(5, 5), ax=None, **kwargs):
        """
        Plot the distribution of the feature Gini coefficient of each
        sample, (n_samples,).
        """
        xf = self.f_gc(f_gc_filter)
        return hist_dens_plot(xf, title=title, xlab=xlab, ylab=ylab,
                              figsize=figsize, ax=ax, **kwargs)

    def s_gc(self, s_gc_filter=None):
        """
        For each feature, compute the Gini coefficient of all samples.

        Returns
        -------
        xf: float array
            (n_features,)
        """
        colgc = []
        for i in range(self._x.shape[1]):
            if self._is_sparse:
                i_x = self._x[:, i].todense().A1
            else:
                i_x = self._x[:, i]
            colgc.append(stats.gc1d(i_x))
        colgc = np.array(colgc)
        f_inds = self.filter_1d_inds(s_gc_filter, colgc)
        colgcf = colgc[f_inds]
        return colgcf

    def s_gc_dist(self, s_gc_filter=None, xlab=None, ylab=None,
                  title=None, figsize=(5, 5), ax=None, **kwargs):
        """
        Plot the distribution of the sample Gini coefficients
        of each feature, (n_features,).
        """
        xf = self.s_gc(s_gc_filter)
        return hist_dens_plot(xf, title=title, xlab=xlab, ylab=ylab,
                              figsize=figsize, ax=ax, **kwargs)

    @property
    def sids(self):
        return self._sids.tolist()

    @property
    def fids(self):
        return self._fids.tolist()

    @property
    def x(self):
        if self._is_sparse:
            return self._x.copy()
        else:
            return self._x.tolist()

    @property
    def _is_sparse(self):
        return spsp.issparse(self._x)
