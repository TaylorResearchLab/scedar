import numpy as np
import scipy.stats as spstats
from abc import ABC, abstractmethod


# TODO: implement histogram mdl with proper discretization.

def np_number_1d(x, dtype=np.dtype("f8"), copy=True):
    """Convert x to 1d np number array

    Args:
        x (1d sequence of values convertable to np.number)
        dtype (np number type): default to 64-bit float
        copy (bool): passed to np.array()

    Returns:
        xarr (1d np.number array)

    Raises:
        ValueError: If x is not convertable to provided dtype or non-1d.
            If dtype is not subdtype of np number.
    """
    if not np.issubdtype(dtype, np.number):
        raise ValueError("dtype must be a type of numpy number")

    xarr = np.array(x, dtype=dtype, copy=copy)
    if xarr.ndim != 1:
        raise ValueError("x should be 1D array. "
                         "x.shape: {}".format(xarr.shape))
    return xarr


class Mdl(ABC):
    """Minimum description length abstract base class

    Interface of various mdl schemas. Subclasses must implement mdl property
        and encode method.

    Attributes:
        _x (1D np.number array): data used for fit mdl
        _n (np.int): number of points in x
    """
    @abstractmethod
    def __init__(self, x, dtype=np.dtype("f8"), copy=True):
        """Initialize

        Args:
            x (1D np.number array): data used for fit mdl
            dtype (np.dtype): default to 64-bit float
            copy (bool): passed to np.array()
        """
        self._x = np_number_1d(x, dtype=dtype, copy=copy)
        # avoid divide by 0 exception
        self._n = np.int_(self._x.shape[0])

    @abstractmethod
    def encode(self, x):
        """Encode another 1D number array with fitted code
        Args:
            x (1D np.number array): data to encode
        """
        raise NotImplementedError

    @property
    def x(self):
        return self._x.copy()

    @property
    @abstractmethod
    def mdl(self):
        raise NotImplementedError


class MultinomialMdl(Mdl):
    """ Encode discrete values using multinomial distribution

    Args:
        x (1D np.number array): data used for fit mdl
        dtype (np.dtype): default to 64-bit float
        copy (bool): passed to np.array()

    Note:
        When x only has 1 uniq value. Encode the the number of values only.
    """

    def __init__(self, x, dtype=np.dtype("f8"), copy=True):
        super().__init__(x, dtype=dtype, copy=copy)

        uniq_vals, uniq_val_cnts = np.unique(x, return_counts=True)
        self._n_uniq = len(uniq_vals)
        self._uniq_vals = uniq_vals
        self._uniq_val_cnts = uniq_val_cnts
        # make division by 0 valid.
        self._uniq_val_ps = uniq_val_cnts / self._n
        # create a lut for unique vals and ps
        self._uniq_val_p_lut = dict(zip(uniq_vals, self._uniq_val_ps))

        if len(self._uniq_vals) > 1:
            mdl = (-np.log(self._uniq_val_ps) * self._uniq_val_cnts).sum()
        elif len(self._uniq_vals) == 1:
            mdl = np.log(self._n)
        else:
            # len(x) == 0
            mdl = 0

        self._mdl = mdl
        return

    def encode(self, qx, use_adjescent_when_absent=False):
        """Encode another 1D float array with fitted code

        Args:
            qx (1d float array): query data
            use_adjescent_when_absent (bool): whether to use adjascent value
                to compute query mdl. If not, uniform mdl is used. If
                adjascent values have same distance to query value, choose the
                one with smaller mdl.

        Returns:
            qmdl (float)
        """
        qx = np_number_1d(qx, copy=False)
        if qx.size == 0:
            return 0

        # Encode with 32bit float
        unif_q_val_mdl = np.log(max(np.max(np.abs(qx))*2, 1))
        if self._n == 0:
            # uniform
            return qx.size * unif_q_val_mdl

        q_uniq_vals, q_uniq_val_cnts = np.unique(qx, return_counts=True)
        q_mdl = 0
        for uval, ucnt in zip(q_uniq_vals, q_uniq_val_cnts):
            uval_p = self._uniq_val_p_lut.get(uval)
            if uval_p is None:
                if use_adjescent_when_absent:
                    uind = np.searchsorted(self._uniq_vals, uval)
                    if uind <= 0:
                        # uval lower than minimum
                        uval_p = self._uniq_val_ps[0]
                    elif uind >= self._n_uniq:
                        # uval higher than maximum
                        uval_p = self._uniq_val_ps[-1]
                    else:
                        # uval within range [1, _n_uniq-1]
                        # abs diff between uval and left val
                        l_diff = np.abs(self._uniq_vals[uind-1] - uval)
                        # abs diff between uval and right avl
                        r_diff = np.abs(self._uniq_vals[uind] - uval)
                        if l_diff < r_diff:
                            # closer to left
                            uval_p = self._uniq_val_ps[uind-1]
                        elif l_diff > r_diff:
                            # closer to right
                            uval_p = self._uniq_val_ps[uind]
                        else:
                            # same distance, choose max p
                            uval_p = max(self._uniq_val_ps[uind-1],
                                         self._uniq_val_ps[uind])
                    uval_mdl = -np.log(uval_p)
                else:
                    uval_mdl = unif_q_val_mdl
            else:
                uval_mdl = -np.log(uval_p)
            q_mdl += uval_mdl * ucnt
        return q_mdl

    @property
    def mdl(self):
        return self._mdl


class ZeroIMdl(Mdl):
    """Encode an indicator vector of 0s and non-0s
    """
    def __init__(self, x, dtype=np.dtype("f8"), copy=True):
        super().__init__(x, dtype=dtype, copy=copy)
        self._x_equal_zero = self._x == 0
        self._mn_encoder = MultinomialMdl(self._x == 0, dtype=np.dtype("i1"),
                                          copy=False)
        # log(3) to encode 3 conditions:
        # - empty
        # - all non-0s or 0s
        # - mixed non-0s and 0s
        self._mdl = self._mn_encoder.mdl + np.log(3)

    def encode(self, qx):
        qx = np_number_1d(qx, copy=False)
        return self._mn_encoder.encode(qx == 0) + np.log(3)

    @property
    def mdl(self):
        return self._mdl


class ZeroIMultinomialMdl(Mdl):
    def __init__(self, x, dtype=np.dtype("f8"), copy=True):
        super().__init__(x, dtype=dtype, copy=copy)
        self._x_nonzero = self._x[np.nonzero(self._x)]
        self._k = self._x_nonzero.shape[0]

        self._zi_encoder = ZeroIMdl(self._x, dtype=dtype, copy=False)
        self._zi_mdl = self._zi_encoder.mdl

        self._mn_encoder = MultinomialMdl(self._x_nonzero, dtype=dtype,
                                          copy=False)
        self._mn_mdl = self._mn_encoder.mdl
        self._mdl = self._zi_mdl + self._mn_mdl

    def encode(self, qx, use_adjescent_when_absent=False):
        qx = np_number_1d(qx)
        qx_nonzero = qx[np.nonzero(qx)]
        qzi_mdl = self._zi_encoder.encode(qx)
        qmn_mdl = self._mn_encoder.encode(
            qx_nonzero, use_adjescent_when_absent=use_adjescent_when_absent)
        return qzi_mdl + qmn_mdl

    @property
    def mdl(self):
        return self._mdl


class GKdeMdl(Mdl):
    """Use Gaussian kernel density estimation to compute mdl

    Args:
        x (1D np.number array): data used for fit mdl
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
        dtype (np.dtype): default to 64-bit float
        copy (bool): passed to np.array()

    Attributes:
        _x (1d float array): data to fit
        _n (int): number of elements in data
        _bw_method (str): bandwidth method
        _kde (:obj:`scipy kde`)
        _logdens (1d float array): log density
    """

    def __init__(self, x, kde_bw_method="scott", dtype=np.dtype("f8"),
                 copy=True):
        super().__init__(x, dtype=dtype, copy=copy)

        self._bw_method = kde_bw_method

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
                # log(2) to encode kde or not
                kde_mdl = -logdens.sum() + np.log(2)
                bw_factor = kde.factor
            except Exception as e:
                kde = None
                logdens = None
                bw_factor = None
                # encode just single value or multiple values
                # fall back on uniform encoding
                unif_sval_mdl = np.log(max(np.max(np.abs(self._x))*2, 1))
                kde_mdl = unif_sval_mdl * self._n

        self._bw_factor = bw_factor
        self._kde = kde
        self._logdens = logdens
        self._mdl = kde_mdl

    def encode(self, qx, mdl_scale_factor=1):
        """Encode query data using fitted KDE code

        Args:
            qx (1d float array)
            mdl_scale_factor (number): times mdl by this number

        Returns:
            float: mdl
        """
        qx = np_number_1d(qx, copy=False)
        if qx.size == 0:
            return 0

        unif_sval_mdl = np.log(max(np.max(np.abs(qx))*2, 1))
        if self._kde is None:
            return unif_sval_mdl * len(qx) * mdl_scale_factor
        else:
            logdens = self._kde.logpdf(qx)
            return -logdens.sum() * mdl_scale_factor

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
    def kde(self):
        return self._kde

    @staticmethod
    def gaussian_kde_logdens(x, bandwidth_method="scott",
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


class ZeroIGKdeMdl(Mdl):
    """
    Zero indicator Gaussian KDE MDL

    Encode the 0s and non-0s using bernoulli distribution.
    Then, encode non-0s using gaussian kde. Finally, one ternary val indicates
    all 0s, all non-0s, or otherwise


    Args:
        x (1D np.number array): data used for fit mdl
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
        dtype (np.dtype): default to 64-bit float
        copy (bool): passed to np.array()

    References:
        [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html

        [2] https://en.wikipedia.org/wiki/Kernel_density_estimation

        [3] https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/

        [4] https://github.com/scipy/scipy/blob/v1.0.0/scipy/stats/kde.py#L42-L564
    """  # noqa

    def __init__(self, x, kde_bw_method="scott", dtype=np.dtype("f8"),
                 copy=True):
        super().__init__(x, dtype=dtype, copy=copy)

        self._x_nonzero = self._x[np.nonzero(self._x)]
        self._k = self._x_nonzero.shape[0]

        self._bw_method = kde_bw_method

        self._zi_encoder = ZeroIMdl(self._x, dtype=dtype, copy=False)
        self._zi_mdl = self._zi_encoder.mdl

        self._kde_encoder = GKdeMdl(self._x_nonzero, kde_bw_method,
                                    dtype=dtype, copy=False)
        self._kde_mdl = self._kde_encoder.mdl
        self._mdl = self._zi_mdl + self._kde_mdl

    def encode(self, qx):
        """Encode qx

        Args:
            qx (1d np number array)

        Returns:
            mdl (float)
        """
        qx = np_number_1d(qx, copy=False)
        qx_nonzero = qx[np.nonzero(qx)]
        qzi_mdl = self._zi_encoder.encode(qx)
        qkde_mdl = self._kde_encoder.encode(qx_nonzero)
        return qzi_mdl + qkde_mdl

    @property
    def zi_mdl(self):
        return self._zi_mdl

    @property
    def bandwidth(self):
        return self._kde_encoder.bandwidth

    @property
    def kde(self):
        return self._kde_encoder.kde

    @property
    def kde_mdl(self):
        return self._kde_mdl

    @property
    def mdl(self):
        return self._mdl

    @property
    def x_nonzero(self):
        return self._x_nonzero.copy()
