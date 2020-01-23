import numpy as np

import scipy.sparse as spsp

from scedar import eda
from scedar import utils

import time

import pickle


class FeatureImputation(object):
    """
    Impute dropped out features using K nearest neighbors approach

    If the value of a feature is below min_present_val in a sample, and all
    its KNNs have above min_present_val, replace the value with the summary
    statistic (default is median) of KNN above threshold values.

    Attributes
    ----------
    _sdm: SampleDistanceMatrix
    _res_lut: dict
        lookup table of the results.
        `{(k, n_do, min_present_val, n_iter): (pu_sdm, pu_idc_arr, stats), \
          ...}`
    """
    def __init__(self, sdm):
        super(FeatureImputation, self).__init__()
        self._sdm = sdm
        self._res_lut = {}

    # TODO: clean up np.array and scipy.sparse distinctions.
    @staticmethod
    def _impute_features_runner(pb_x, knn_ordered_ind_dict, k, n_do,
                                min_present_val, n_iter,
                                statistic_fun=np.median, verbose=False):
        """
        Runs KNN dropout imputation on single parameter set in one process.
        """
        start_time = time.time()
        curr_x_arr = pb_x
        if spsp.issparse(curr_x_arr):
            if not spsp.isspmatrix_csr(curr_x_arr):
                curr_x_arr = curr_x_arr.tocsr()
        # curr_x_arr is only accessed but not edited
        n_samples, n_features = curr_x_arr.shape
        # Indicator matrix of whether an entry is imputed.
        impute_idc_arr = spsp.lil_matrix(curr_x_arr.shape, dtype=int)
        stats = "Preparation: {:.2f}s\n".format(time.time() - start_time)
        if verbose:
            print(stats)


        for i in range(1, n_iter + 1):

            iter_start_time = time.time()
            # next_x_arr is edited
            if spsp.issparse(curr_x_arr):
                next_x_arr = curr_x_arr.tolil(copy=True)
            else:
                next_x_arr = curr_x_arr.copy()
            curr_x_present_arr = curr_x_arr >= min_present_val
            # iteratively decreases n_do_i from k to n_do
            # Pick-up easy ones first
            n_do_i = n_do + int(np.ceil((n_iter - i) / n_iter * (k - n_do)))
            # print('------', n_do_i)

            ipt_s_inds = np.array([], dtype=int)
            ipt_fais = np.array([], dtype=int)
            ipt_vals = np.array([], dtype=int)
            for s_ind in range(n_samples):
                # print(s_ind)
                knn_ordered_inds = knn_ordered_ind_dict[s_ind]
                s_parr = curr_x_present_arr[s_ind, :]
                knn_parr = curr_x_present_arr[knn_ordered_inds, :]
                if spsp.issparse(knn_parr):
                    # above threshold
                    knn_at_mask = (knn_parr.sum(axis=0) >= n_do_i).A1
                    # absent
                    s_f_a_mask = s_parr.todense().A1 == 0
                else:
                    knn_at_mask = knn_parr.sum(axis=0) >= n_do_i
                    s_f_a_mask = s_parr == 0
                # print(knn_at_mask, s_f_a_mask)
                ipt_f_mask = np.logical_and(knn_at_mask, s_f_a_mask)

                if spsp.issparse(curr_x_arr):
                    # summary stat array
                    f_n_knn_ss_arr = statistic_fun(
                        curr_x_arr[knn_ordered_inds, :].tocsc()[:, ipt_f_mask].A, axis=0)
                else:
                    f_n_knn_ss_arr = statistic_fun(
                        curr_x_arr[knn_ordered_inds, :][:, ipt_f_mask], axis=0)
                s_ipt_f_inds = np.where(ipt_f_mask)[0]

                ipt_fais = np.concatenate((ipt_fais, s_ipt_f_inds))
                ipt_s_inds = np.concatenate(
                    (ipt_s_inds, np.repeat(s_ind, len(s_ipt_f_inds))))
                ipt_vals = np.concatenate((ipt_vals, f_n_knn_ss_arr))

            next_x_arr[ipt_s_inds, ipt_fais] = ipt_vals

            impute_idc_arr[ipt_s_inds, ipt_fais] = i
            if spsp.issparse(next_x_arr):
                curr_x_arr = next_x_arr.tocsr()
            else:
                curr_x_arr = next_x_arr

            n_pu_features_per_s = np.sum(impute_idc_arr > 0, axis=1).A1
            n_pu_entries = np.sum(n_pu_features_per_s)
            # number of samples with feature being picked up.
            n_samples_wfpu = np.sum(n_pu_features_per_s > 0)
            n_pu_entries_ratio = n_pu_entries / (n_samples * n_features)

            iter_time = time.time() - iter_start_time
            stats_update = str.format(
                "Iteration {} ({:.2f}s): picked up {} total "
                "features in {} ({:.1%}) cells. {:.1%} among "
                "samples * features.\n", i, iter_time,
                n_pu_entries, n_samples_wfpu,
                n_samples_wfpu / n_samples,
                n_pu_entries_ratio)
            stats += stats_update
            if verbose:
                print(stats_update)
        curr_x_pb = curr_x_arr
        impute_idc_pb = impute_idc_arr
        stats_update = "Complete in {:.2f}s\n".format(
            time.time() - start_time)
        if verbose:
            print(stats_update)
        stats += stats_update
        return curr_x_pb, impute_idc_pb, stats

    def impute_features(self, k, n_do, min_present_val, n_iter, nprocs=1,
                        statistic_fun=np.median, metric=None, use_pca=False,
                        use_hnsw=False, index_params=None, query_params=None,
                        verbose=False):
        """
        Runs KNN imputation on multiple parameter sets parallely.

        Each parameter set will be executed in one process.

        Parameters
        ----------
        k: int
            Look at k nearest neighbors to decide whether to impute or not.
        n_do: int
            Minimum (`>=`) number of above min_present_val neighbors among KNN
            to be callsed as dropout, so that imputation will be performed.
        min_present_val: float
            Minimum (`>=`) values of a feature to be called as present.
        n_iter: int
            The number of iterations to run.
        statistic_fun: callable
            The summary statistic used to correct gene dropouts. Default is
            median.

        Returns
        -------
        resl: list
            list of results, `[(pu_sdm, pu_idc_arr, stats), ...]`.

            pu_sdm: SampleDistanceMatrix
                SampleDistanceMatrix after imputation
            pu_idc_arr: array of shape (n_samples, n_features)
                Indicator matrix of the ith iteration an entry is being
                imputed.
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
        start_time = time.time()
        try:
            # make sure that the function runs on list of numbers
            if not np.isscalar(np.isreal(statistic_fun([0, 1, 2]))):
                raise ValueError("statistic_fun should be a function of a"
                                 "list of numbers that returns a scalar.")
        except Exception:
            raise ValueError("statistic_fun should be a function of a"
                             "list of numbers that returns a scalar.")

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
                       n_iter_list[i], statistic_fun)
                      for i in range(n_param_tups)]
        res_list = []
        # use cached results with the following procedure
        # 1. put cached results to res_list, with not cached ones as None
        # 2. run not cached ones
        # 3. after running, cache the results results and fill res_list
        # same as filter
        # TODO: abstract the running pattern into a function

        # parameter tuples without cached results for running
        run_param_tups = []
        # indices of results to be filled after running
        res_list_run_inds = []
        for i, ptup in enumerate(param_tups):
            if ptup in self._res_lut:
                res_list.append(self._res_lut[ptup])
            else:
                run_param_tups.append(ptup)
                res_list.append(None)
                res_list_run_inds.append(i)
        # set up parameters for running
        run_param_setup_tups = []
        for ptup in run_param_tups:
            # assumes that the first element of the ptup is k
            s_knn_ind_lut = self._sdm.s_knn_ind_lut(
                ptup[0], metric=metric, use_pca=use_pca,
                use_hnsw=use_hnsw,
                index_params=index_params, query_params=query_params,
                verbose=verbose)
            run_param_setup_tups.append(
                (self._sdm._x, s_knn_ind_lut) + ptup)

        stats = "Compute KNNs: {:.2f}s\n".format(time.time() - start_time)
        if verbose:
            print(stats)

        nprocs = int(nprocs)
        nprocs = min(nprocs, n_param_tups)
        run_res_list = utils.parmap(
            lambda ptup: self._impute_features_runner(*ptup, verbose=verbose),
            run_param_setup_tups, nprocs)

        for i, param_tup in enumerate(run_param_tups):
            # cache results
            if param_tup in self._res_lut:
                raise NotImplementedError("Unexpected scenario encountered")
            res_x = run_res_list[i][0]
            res_idc = run_res_list[i][1]
            res_tup = (res_x, res_idc, run_res_list[i][2])
            self._res_lut[param_tup] = res_tup
            # fill res_list
            if res_list[res_list_run_inds[i]] is not None:
                raise NotImplementedError("Unexpected scenario encountered")
            res_list[res_list_run_inds[i]] = res_tup

        kpu_sdm_list = []
        for res in res_list:
            kpu_x = res[0]
            kpu_sdm = eda.SampleDistanceMatrix(
                kpu_x,
                metric=self._sdm._metric,
                sids=self._sdm.sids,
                fids=self._sdm.fids,
                nprocs=self._sdm._nprocs)
            kpu_sdm_list.append(kpu_sdm)
        return kpu_sdm_list
