import numpy as np
import scipy.spatial as spspatial
import scipy.stats as spstats


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


class GKdeMdl(object):
    """docstring for GKdeMdl"""
    def __init__(self, x, kde_bw_method="silverman"):
        super(GKdeMdl, self).__init__()

        if x.ndim != 1:
            raise ValueError("x should be 1D array. "
                             "x.shape: {}".format(x.shape))

        self._x = x
        self._n = x.shape[0]

        self._bw_method = kde_bw_method

        self._mdl = self._kde_mdl()

    def _kde_mdl(self):
        if self._n == 0:
            kde = None
            logdens = None
            bw_factor = None
            # no non-zery vals. Indicator encoded by zi mdl.
            kde_mdl = 0
        else:
            try:
                logdens, kde = self.gaussian_kde_logdens(
                    self._x, bandwidth_method=self._bw_method,
                    ret_kernel=True)
                kde_mdl = -logdens.sum() + np.log(2)
                bw_factor = kde.factor
            except Exception as e:
                kde = None
                logdens = None
                bw_factor = None
                # encode just single value or multiple values
                kde_mdl = MultinomialMdl(
                    (self._x * 100).astype(int)).mdl

        self._bw_factor = bw_factor
        self._kde = kde
        self._logdens = logdens
        return kde_mdl

    @property
    def bandwidth(self):
        if self._bw_factor is None:
            return None
        else:
            return self._bw_factor * self._x.std(ddof=1)

    @property
    def mdl(self):
        return self._mdl

    @property
    def x(self):
        return self._x.copy()

    @property
    def kde(self):
        return self._kde

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


class ZeroIGKdeMdl(object):
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
        super(ZeroIGKdeMdl, self).__init__()

        if x.ndim != 1:
            raise ValueError("x should be 1D array. "
                             "x.shape: {}".format(x.shape))

        self._x = x
        self._n = x.shape[0]

        self._x_nonzero = x[np.nonzero(x)]
        self._k = self._x_nonzero.shape[0]

        self._bw_method = kde_bw_method

        self._zi_mdl = self._compute_zero_indicator_mdl()
        self._kde_mdl_obj = GKdeMdl(self._x_nonzero, kde_bw_method)
        self._kde_mdl = self._kde_mdl_obj.mdl
        self._mdl = self._zi_mdl + self._kde_mdl

    def _compute_zero_indicator_mdl(self):
        if self._n == 0:
            zi_mdl = 0
        elif self._k == self._n or self._k == 0:
            zi_mdl = np.log(3)
        else:
            p = self._k / self._n
            zi_mdl = (np.log(3) - self._k * np.log(p) -
                      (self._n - self._k) * np.log(1-p))
        return zi_mdl

    @property
    def bandwidth(self):
        return self._kde_mdl_obj.bandwidth

    @property
    def kde(self):
        return self._kde_mdl_obj.kde

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
