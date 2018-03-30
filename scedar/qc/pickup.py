import numpy as np

from .. import utils
from .. import eda

import time


class FeatureKNNPickUp(object):
    """
    "Pick-up" dropped out features using K nearest neighbors (KNN) approach.

    If the value of a feature is below min_present_val in a sample, and all
    its KNNs have above min_present_val, replace the value with the median
    of KNN above threshold values.

    Attributes
    ----------
    _sdm: SampleDistanceMatrix
    _res_lut: dict
        lookup table of the results.
        `{(k, n_do, min_present_val, n_iter): (pu_sdm, pu_idc_arr, stats), \
          ...}`
    """
    def __init__(self, sdm):
        super(FeatureKNNPickUp, self).__init__()
        self._sdm = sdm
        self._res_lut = {}

    def _knn_pickup_features_runner(self, k, n_do, min_present_val, n_iter):
        """
        Runs KNN pick-up on single parameter set in one process.
        """
        param_key = (k, n_do, min_present_val, n_iter)
        if param_key in self._res_lut:
            return self._res_lut[param_key]

        start_time = time.time()
        n_samples, n_features = self._sdm._x.shape
        # {sample_ind : [1st_NN_ind(neq sample_ind),
        #                 2nd_NN_ind, ..., nth_NN_ind], ...}
        knn_ordered_ind_dict = self._sdm.s_knn_ind_lut(k)
        # curr_x_arr is only accessed but not edited
        curr_x_arr = self._sdm._x
        next_x_arr = np.copy(curr_x_arr)
        curr_x_present_arr = curr_x_arr >= min_present_val
        curr_x_absent_arr = np.logical_not(curr_x_present_arr)
        # next_x_arr is edited
        # Indicator matrix of whether an entry is picked up.
        pickup_idc_arr = np.zeros(curr_x_arr.shape, dtype=int)

        stats = "Preparation: {:.2f}s\n".format(time.time() - start_time)

        for i in range(1, n_iter + 1):
            iter_start_time = time.time()
            # iteratively decreases n_do_i from k to n_do
            # Pick-up easy ones first
            n_do_i = n_do + int(np.ceil((n_iter - i) / n_iter * (k - n_do)))
            for s_ind in range(n_samples):
                # below threshold feature indices
                f_absent_inds = np.where(curr_x_absent_arr[s_ind, :])[0]
                knn_ordered_inds = knn_ordered_ind_dict[s_ind]
                # (n_features,) number of KNNs have feature present
                f_n_knn_present_arr = np.sum(
                    curr_x_present_arr[knn_ordered_inds, :], axis=0)

                for fai in f_absent_inds:
                    # feature fai present in >= n_do_i NNs
                    if  f_n_knn_present_arr[fai] >= n_do_i:
                        # Mark (s_ind, fai) as picked up
                        pickup_idc_arr[s_ind, fai] = i
                        # Replace val in (s_ind, fai) with median of knn
                        knn_x_arr = curr_x_arr[knn_ordered_inds, fai]
                        knn_x_present_med = np.median(
                            knn_x_arr[knn_x_arr >= min_present_val])
                        next_x_arr[s_ind, fai] = knn_x_present_med

            curr_x_arr = next_x_arr
            next_x_arr = np.copy(curr_x_arr)
            curr_x_present_arr = curr_x_arr >= min_present_val
            curr_x_absent_arr = np.logical_not(curr_x_present_arr)

            n_pu_features_per_s = np.sum(pickup_idc_arr, axis=1)
            n_pu_entries = np.sum(n_pu_features_per_s)
            # number of samples with feature being picked up.
            n_samples_wfpu = np.sum(n_pu_features_per_s > 0)
            n_pu_entries_ratio = n_pu_entries / pickup_idc_arr.size

            iter_time = time.time() - iter_start_time
            stats += str.format("Iteration {} ({:.2f}s): picked up {} total "
                                "features in {} ({:.1%}) cells. {:.1%} among "
                                "samples * features.\n", i, iter_time,
                                n_pu_entries, n_samples_wfpu,
                                n_samples_wfpu / n_samples,
                                n_pu_entries_ratio)

        pu_sdm = eda.SampleDistanceMatrix(curr_x_arr,
                                          metric=self._sdm._metric,
                                          sids=self._sdm.sids,
                                          fids=self._sdm.fids,
                                          nprocs=self._sdm._nprocs)

        stats += "Complete in {:.2f}s\n".format(time.time() - start_time)

        return pu_sdm, pickup_idc_arr, stats

    def knn_pickup_features(self, k, n_do, min_present_val, n_iter, nprocs=1):
        """
        Runs KNN pick-up on multiple parameter sets parallely.

        Each parameter set will be executed in one process.

        Parameters
        ----------
        k: int
            Look at k nearest neighbors to decide whether to pickup or not.
        n_do: int
            Minimum (`>=`) number of above min_present_val neighbors among KNN
            to be callsed as drop-out, so that pick-up will be performed.
        min_present_val: float
            Minimum (`>=`) values of a feature to be called as present.
        n_iter: int
            The number of iterations to run.

        Returns
        -------
        resl: list
            list of results, `[(pu_sdm, pu_idc_arr, stats), ...]`.

            pu_sdm: SampleDistanceMatrix
                SampleDistanceMatrix after pick-up
            pu_idc_arr: array of shape (n_samples, n_features)
                Indicator matrix of the ith iteration an entry is being
                picked up.
            stats: str
                Stats of the run.


        Notes
        -----
        If parameters are provided as lists of equal length n, the n
        corresponding parameter tuples will be executed parallely.

        Example
        -------

        If `k = [10, 15]`, `n_do = [1, 2]`, `min_present_val = [5, 6]`, and
        `n_iter = [10, 20]`, `(k, n_do, min_present_val, n_iter)` tuples
        `(10, 1, 5, 10) and (15, 2, 6, 20)` will be tried parallely
        with nprocs.

        n_do, min_present_val, n_iter
        """
        if np.isscalar(k):
            k_list = [k]
        else:
            k_list = list(k)

        if np.isscalar(n_do):
            n_do_list = [n_do]
        else:
            n_do_list = list(n_do)

        if np.isscalar(min_present_val):
            min_present_val_list = [min_present_val]
        else:
            min_present_val_list = list(min_present_val)

        if np.isscalar(n_iter):
            n_iter_list = [n_iter]
        else:
            n_iter_list = list(n_iter)

        # Check all param lists have the same length
        if not (len(k_list) == len(n_do_list) ==
                len(min_present_val_list) == len(n_iter_list)):
            raise ValueError("Parameter should have the same length."
                             "k: {}, n_do: {}, min_present_val: {}, "
                             "n_iter: {}.".format(k, n_do, min_present_val,
                                                  n_iter))
        n_param_tups = len(k_list)
        # type check all parameters
        for i in range(n_param_tups):
            if k_list[i] < 1 or k_list[i] >= self._sdm._x.shape[0]:
                raise ValueError("k should be >= 1 and < n_samples. "
                                 "k: {}".format(k))
            else:
                k_list[i] = int(k_list[i])

            if n_do_list[i] > k_list[i] or n_do_list[i] < 1:
                raise ValueError("n_do should be  <= k and >= 1. "
                                 "n_do: {}".format(n_do))
            else:
                n_do_list[i] = int(n_do_list[i])

            min_present_val_list[i] = float(min_present_val_list[i])

            if n_iter_list[i] < 1:
                raise ValueError("n_iter should be >= 1. "
                                 "n_iter: {}".format(n_iter))
            else:
                n_iter_list[i] = int(n_iter_list[i])

        param_tups = [(k_list[i], n_do_list[i], min_present_val_list[i],
                       n_iter_list[i])
                      for i in range(n_param_tups)]
        nprocs = int(nprocs)
        nprocs = min(nprocs, n_param_tups)

        f = lambda ptup: self._knn_pickup_features_runner(*ptup)

        if nprocs != 1:
            res_list = utils.parmap(f, param_tups, nprocs)
        else:
            res_list = list(map(f, param_tups))

        for i in range(n_param_tups):
            if param_tups[i] not in self._res_lut:
                self._res_lut[param_tups[i]] = res_list[i]

        return [res[0] for res in res_list]
