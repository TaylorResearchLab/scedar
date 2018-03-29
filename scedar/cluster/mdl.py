import numpy as np
import scipy.spatial as spspatial
import scipy.stats as spstats

from .. import utils
from .. import eda


class MultinomialMdl(object):
    """
    Encode discrete values using multinomial distribution

    Parameters
    ----------
    x: 1d float array
        Should be non-negative

    Notes
    -----
    When x only has 1 uniq value. Encode the the number of values only.
    """

    def __init__(self, x):
        super(MultinomialMdl, self).__init__()
        x = np.array(x)
        if x.ndim != 1:
            raise ValueError("x should be 1D array. "
                             "x.shape: {}".format(x.shape))
        self._x = x
        self._n = x.shape[0]
        self._mdl = self._mn_mdl()
        return

    def _mn_mdl(self):
        uniq_vals, uniq_val_cnts = np.unique(self._x, return_counts=True)
        if len(uniq_vals) > 1:
            return (-np.log(uniq_val_cnts / uniq_val_cnts.sum()) * uniq_val_cnts).sum()
        elif len(uniq_vals) == 1:
            return np.log(uniq_val_cnts)
        else:
            # len(x) == 0
            return 0

    @property
    def x(self):
        return self._x.tolist()

    @property
    def mdl(self):
        return self._mdl


class ZeroIdcGKdeMdl(object):
    """
    Zero indicator Gaussian KDE MDL

    Encode the 0s and non-0s using bernoulli distribution.
    Then, encode non-0s using gaussian kde. Finally, one ternary val indicates
    all 0s, all non-0s, or otherwise


    Parameters
    ----------
    x: 1d float array
        Should be non-negative
    bandwidth_method: string
        KDE bandwidth estimation method bing passed to
        `scipy.stats.gaussian_kde`.
        Types:
        * `"scott"`: Scott's rule of thumb.
        * `"silverman"`: Silverman"s rule of thumb.
        * `constant`: constant will be timed by x.std(ddof=1) internally,
        because scipy times bw_method value by std. "Scipy weights its
        bandwidth by the ovariance of the input data" [3].
        * `callable`: scipy calls the function on self

    References
    ----------
    [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html

    [2] https://en.wikipedia.org/wiki/Kernel_density_estimation

    [3] https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/

    [4] https://github.com/scipy/scipy/blob/v1.0.0/scipy/stats/kde.py#L42-L564

    """


    def __init__(self, x, kde_bw_method="silverman"):
        super(ZeroIdcGKdeMdl, self).__init__()

        if x.ndim != 1:
            raise ValueError("x should be 1D array. "
                             "x.shape: {}".format(x.shape))

        self._x = x
        self._n = x.shape[0]

        self._x_nonzero = x[np.nonzero(x)]
        self._k = self._x_nonzero.shape[0]

        self._bw_method = kde_bw_method

        if self._n != 0:
            self._zi_mdl = self._compute_zero_indicator_mdl()
            self._kde_mdl = self._compute_non_zero_val_mdl()
            self._mdl = self._zi_mdl + self._kde_mdl
        else:
            self._zi_mdl = 0
            self._kde_mdl = 0
            self._mdl = 0

    @staticmethod
    def gaussian_kde_logdens(x, bandwidth_method="silverman",
                             ret_kernel=False):
        """
        Estimate Gaussian kernel density estimation bandwidth for input `x`.

        Parameters
        ----------
        x: float array of shape `(n_samples)` or `(n_samples, n_features)`
            Data points for KDE estimation.
        bandwidth_method: string
            KDE bandwidth estimation method bing passed to
            `scipy.stats.gaussian_kde`.

        """

        # This package uses (n_samples, n_features) convention
        # scipy uses (n_featues, n_samples) convention
        # so it is necessary to reshape the data
        if x.ndim == 1:
            x = x.reshape(1, -1)
        elif x.ndim == 2:
            x = x.T
        else:
            raise ValueError("x should be 1/2D array. "
                             "x.shape: {}".format(x.shape))

        kde = spstats.gaussian_kde(x, bw_method=bandwidth_method)
        logdens = np.log(kde.evaluate(x))

        if ret_kernel:
            return (logdens, kde)
        else:
            return logdens

    def _compute_zero_indicator_mdl(self):
        if self._k == self._n or self._k == 0:
            zi_mdl = np.log(3)
        else:
            p = self._k / self._n
            zi_mdl = (np.log(3) - self._k * np.log(p) -
                      (self._n - self._k) * np.log(1-p))
        return zi_mdl

    def _compute_non_zero_val_mdl(self):
        if self._k == 0:
            kde = None
            logdens = None
            bw_factor = None
            # no non-zery vals. Indicator encoded by zi mdl.
            kde_mdl = 0
        else:
            try:
                logdens, kde = self.gaussian_kde_logdens(
                    self._x_nonzero, bandwidth_method=self._bw_method,
                    ret_kernel=True)
                kde_mdl = -logdens.sum() + np.log(2)
                bw_factor = kde.factor
            except Exception as e:
                kde = None
                logdens = None
                bw_factor = None
                # encode just single value or multiple values
                kde_mdl = MultinomialMdl(
                    (self._x_nonzero * 100).astype(int)).mdl

        self._bw_factor = bw_factor
        self._kde = kde
        self._logdens = logdens
        return kde_mdl

    @property
    def bandwidth(self):
        if self._bw_factor is None:
            return None
        else:
            return self._bw_factor * self._x_nonzero.std(ddof=1)

    @property
    def zi_mdl(self):
        return self._zi_mdl

    @property
    def kde_mdl(self):
        return self._kde_mdl

    @property
    def mdl(self):
        return self._mdl

    @property
    def x(self):
        return self._x.copy()

    @property
    def x_nonzero(self):
        return self._x_nonzero.copy()

class MDLSampleDistanceMatrix(eda.SingleLabelClassifiedSamples):
    """
    MDLSampleDistanceMatrix inherits SingleLabelClassifiedSamples to offer MDL
    operations.
    """

    def __init__(self, x, labs, sids=None, fids=None,
                 d=None, metric="correlation", nprocs=None):
        super(MDLSampleDistanceMatrix, self).__init__(x=x, labs=labs,
                                                      sids=sids, fids=fids,
                                                      d=d, metric=metric,
                                                      nprocs=nprocs)

    @staticmethod
    def per_column_zigkmdl(x, nprocs=1, verbose=False, ret_internal=False):
        # verbose is not implemented
        if x.ndim != 2:
            raise ValueError("x should have shape (n_samples, n_features)."
                             "x.shape: {}".format(x.shape))

        nprocs = max(int(nprocs), 1)

        # apply to each feature
        if nprocs != 1:
            col_mdl_list = utils.parmap(lambda x1d: ZeroIdcGKdeMdl(x1d),
                                        x.T, nprocs)
        else:
            col_mdl_list = list(map(lambda x1d: ZeroIdcGKdeMdl(x1d), x.T))

        col_mdl_sum = sum(map(lambda zkmdl: zkmdl.mdl, col_mdl_list))
        if ret_internal:
            return col_mdl_sum, col_mdl_list
        else:
            return col_mdl_sum

    def no_lab_mdl(self, nprocs=1, verbose=False):
        # verbose is not implemented
        col_mdl_sum = self.per_column_zigkmdl(self._x, nprocs, verbose)
        return col_mdl_sum

    def lab_mdl(self, cl_mdl_scale_factor=1, nprocs=1, verbose=False,
                ret_internal=False):
        n_uniq_labs = self._uniq_labs.shape[0]
        ulab_s_ind_list = [np.where(self._labs == ulab)[0].tolist()
                           for ulab in self._uniq_labs]

        ulab_x_list = [self._x[i, :] for i in ulab_s_ind_list]

        ulab_cnt_ratios = self._uniq_lab_cnts / self._x.shape[0]

        # MDL for points in each cluster
        pts_mdl_list = [self.per_column_zigkmdl(x, nprocs, verbose)
                        for x in ulab_x_list]

        # Additional MDL for encoding the cluster:
        # - labels are encoded by multinomial distribution
        # - KDE bandwidth factors are encoded by 32bit float
        #   np.log(2**32) = 22.18070977791825
        # - scaled by factor
        cluster_mdl = ((MultinomialMdl(self._labs).mdl
                        + 22.18070977791825 * n_uniq_labs)
                       * cl_mdl_scale_factor)

        ulab_mdl_list = [pts_mdl_list[i] + cluster_mdl * ulab_cnt_ratios[i]
                         for i in range(n_uniq_labs)]

        if ret_internal:
            return (ulab_s_ind_list, self._uniq_lab_cnts.tolist(),
                    ulab_mdl_list, cluster_mdl, pts_mdl_list)
        else:
            return (ulab_s_ind_list, self._uniq_lab_cnts.tolist(),
                    ulab_mdl_list, cluster_mdl)

