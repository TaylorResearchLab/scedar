import numpy as np

from scedar import eda
from scedar import utils


class RareSampleDetection(object):
    """
    K nearest neighbor detection of rare samples

    Perform the rare sample detection procedure in parallel, with each
    combination of parameters as a process. Because this procedure runs
    iteratively, parallelizing each individual parameter combination run is
    not implemented.

    Stores the results for further lookup.

    Parameters
    ----------
    sdm: SampleDistanceMatrix or its subclass

    Attributes
    ----------
    _sdm: SampleDistanceMatrix
    _res_lut: dict
        lookup table of KNN rare sample detection results
    """
    def __init__(self, sdm):
        super(RareSampleDetection, self).__init__()
        self._sdm = sdm
        self._res_lut = {}

    def _rare_sample_detection_runner(self, k, d_cutoff, n_iter):
        """
        KNN rare sample detection with scalar tuple of parameters.
        """
        # TODO: copy lookup results
        param_key = (k, d_cutoff, n_iter)
        if param_key in self._res_lut:
            return self._res_lut[param_key]

        d_max = self._sdm._d.max()
        # curr_dist_mat being reduced by removing samples, but the data matrix
        # is not edited. Thus, no need to copy.
        curr_dist_mat = self._sdm._d
        curr_s_inds = np.arange(curr_dist_mat.shape[0])
        progress_list = [curr_s_inds.tolist()]

        for i in range(1, n_iter+1):
            i_d_cutoff = (d_cutoff +
                          (n_iter - i) / n_iter * max(0, d_max - d_cutoff))
            i_k = min(curr_dist_mat.shape[0]-1, k)
            kept_curr_s_inds = []
            # each column is sorted. Not in-place.
            sorted_curr_dist_mat = np.sort(curr_dist_mat, axis=0)
            # size of curr_dist_mat is updated each iteration
            for s_ind in range(curr_dist_mat.shape[0]):
                # i_k'th neighbor distance of sample s_ind
                if sorted_curr_dist_mat[i_k, s_ind] <= i_d_cutoff:
                    kept_curr_s_inds.append(s_ind)
            # no change of value. No need to copy.
            curr_dist_mat = curr_dist_mat[np.ix_(kept_curr_s_inds,
                                                 kept_curr_s_inds)]
            curr_s_inds = curr_s_inds[kept_curr_s_inds]
            progress_list.append(curr_s_inds.tolist())
        # thes last one of progress_list is equal to the curr_s_inds
        return curr_s_inds.tolist(), progress_list

    def detect_rare_samples(self, k, d_cutoff, n_iter, nprocs=1):
        """
        KNN rare sample detection with multiple parameter combinations

        Assuming that there are at least k samples look similar in this
        dataset, the samples with less than k similar neighbors may be
        rare. The rare samples can either be really distinct from the general
        populaton or caused by technical errors.

        This procedure iteratively detects samples according to their k-th
        nearest neighbors. The samples most distinct from its k-th nearest
        neighbors are detected first. Then, the left samples are detected
        by less stringent distance cutoff. The distance cutoff decreases
        linearly from maximum distance to d_cutoff with n_iter iterations.

        Parameters
        ----------
        k: int list or scalar
            K nearest neighbors to detect rare samples.
        d_cutoff: float list or scalar
            Samples with >= d_cutoff distances are distinct from each other.
            Minimum (>=) distance to be called as rare.
        n_iter: int list or scalar
            N progressive iNN detections on the dataset.
        nproces: int
            N processes to run all parameter tuples.

        Returns
        -------
        res_list
            Filtered SampleDistanceMatrix of each corresponding parameter
            tuple.

        Notes
        -----
        If parameters are provided as lists of equal length n, the n
        corresponding parameter tuples will be executed parallely.

        Example:

        `k = [10, 15, 20]`

        `d_cutoff = [1, 2, 3]`

        `n_iter = [10, 20, 30]`

        `(k, d_cutoff, n_iter)` tuples `(10, 1, 10), (15, 2, 20), (20, 3, 30)`
        will be tried parallely with nprocs.
        """
        # Convert scalar to list
        if np.isscalar(k):
            k_list = [k]
        else:
            k_list = list(k)

        if np.isscalar(d_cutoff):
            d_cutoff_list = [d_cutoff]
        else:
            d_cutoff_list = list(d_cutoff)

        if np.isscalar(n_iter):
            n_iter_list = [n_iter]
        else:
            n_iter_list = list(n_iter)
        # Check all param lists have the same length
        if not (len(k_list) == len(d_cutoff_list) == len(n_iter_list)):
            raise ValueError("Parameter should have the same length."
                             "k: {}, d_cutoff: {}, n_iter: {}.".format(
                                k, d_cutoff, n_iter))
        n_param_tups = len(k_list)
        # type check all parameters
        for i in range(n_param_tups):
            if k_list[i] < 1 or k_list[i] > self._sdm._x.shape[0] - 1:
                raise ValueError("k should be >= 1 and <= n_samples-1. "
                                 "k: {}".format(k))
            else:
                k_list[i] = int(k_list[i])

            if d_cutoff_list[i] <= 0:
                raise ValueError("d_cutoff should be > 0. "
                                 "d_cutoff: {}".format(d_cutoff))
            else:
                d_cutoff_list[i] = float(d_cutoff_list[i])

            if n_iter_list[i] < 1:
                raise ValueError("n_iter should be >= 1. "
                                 "n_iter: {}".format(n_iter))
            else:
                n_iter_list[i] = int(n_iter_list[i])

        param_tups = [(k_list[i], d_cutoff_list[i], n_iter_list[i])
                      for i in range(n_param_tups)]
        nprocs = int(nprocs)
        nprocs = min(nprocs, n_param_tups)

        # returns (filtered_sdm, progress_list (list of kept indices))
        res_list = utils.parmap(
            lambda ptup: self._rare_sample_detection_runner(*ptup),
            param_tups, nprocs)

        for i in range(n_param_tups):
            if param_tups[i] not in self._res_lut:
                self._res_lut[param_tups[i]] = res_list[i]

        return [res[0] for res in res_list]
