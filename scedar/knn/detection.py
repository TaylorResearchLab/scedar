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

    def _no_pdist_rare_s_detect(self, k, d_cutoff, n_iter, metric=None,
                                use_pca=False, use_hnsw=False,
                                index_params=None, query_params=None):
        param_key = (k, d_cutoff, n_iter)
        if param_key in self._res_lut:
            return self._res_lut[param_key]

        curr_sdm = self._sdm
        curr_knn_s_inds, curr_knn_distances = curr_sdm.s_knns(
            k, metric=metric, use_pca=use_pca, use_hnsw=use_hnsw,
            index_params=index_params, query_params=query_params)
        d_max = max([x[-1] for x in curr_knn_distances])
        # curr_dist_mat being reduced by removing samples, but the data matrix
        # is not edited. Thus, no need to copy.
        # curr_dist_mat = self._sdm._d
        # TODO: optimize to keep track of the removed samples.
        curr_s_inds = np.arange(len(curr_knn_s_inds))
        progress_list = [curr_s_inds.tolist()]

        for i in range(1, n_iter+1):
            i_d_cutoff = (d_cutoff +
                          (n_iter - i) / n_iter * max(0, d_max - d_cutoff))
            # print(i_d_cutoff)
            # print(curr_knn_distances)
            # i_k = min(curr_dist_mat.shape[0]-1, k)
            kept_curr_s_inds = []
            # each column is sorted. Not in-place.
            # sorted_curr_dist_mat = np.sort(curr_dist_mat, axis=0)
            # size of curr_dist_mat is updated each iteration
            for s_ind in range(len(curr_knn_s_inds)):
                # i_k'th neighbor distance of sample s_ind
                if curr_knn_distances[s_ind][-1] <= i_d_cutoff:
                    kept_curr_s_inds.append(s_ind)
            curr_s_inds = curr_s_inds[kept_curr_s_inds]
            progress_list.append(curr_s_inds.tolist())
            if len(curr_s_inds) == 0:
                return curr_s_inds.tolist(), progress_list
            # no change of value. No need to copy.
            curr_sdm = eda.SampleDistanceMatrix(
                curr_sdm._x[kept_curr_s_inds, :], d=None,
                metric=curr_sdm._metric, use_pdist=False,
                nprocs=curr_sdm._nprocs)
            curr_knn_s_inds, curr_knn_distances = curr_sdm.s_knns(
                k, metric=metric, use_pca=use_pca, use_hnsw=use_hnsw,
                index_params=index_params, query_params=query_params)
        # thes last one of progress_list is equal to the curr_s_inds
        return curr_s_inds.tolist(), progress_list

    def _pdist_rare_s_detect(self, k, d_cutoff, n_iter, metric=None,
                             use_pca=False, use_hnsw=False,
                             index_params=None, query_params=None):
        """
        KNN rare sample detection with scalar tuple of parameters.

        Keep the same interface as the _no_pdist_rare_s_detect. The following
        parameters are not used, [metric, use_pca, use_hnsw, index_params,
        query_params].
        """
        # TODO: copy lookup results
        param_key = (k, d_cutoff, n_iter)
        if param_key in self._res_lut:
            return self._res_lut[param_key]

        # curr_dist_mat being reduced by removing samples, but the data matrix
        # is not edited. Thus, no need to copy.
        curr_dist_mat = self._sdm._d
        sorted_curr_dist_mat = np.sort(curr_dist_mat, axis=0)
        d_max = sorted_curr_dist_mat[k, :].max()

        curr_s_inds = np.arange(curr_dist_mat.shape[0])
        progress_list = [curr_s_inds.tolist()]

        for i in range(1, n_iter+1):
            i_d_cutoff = (d_cutoff +
                          (n_iter - i) / n_iter * max(0, d_max - d_cutoff))
            i_k = min(curr_dist_mat.shape[0]-1, k)
            kept_curr_s_inds = []
            # each column is sorted. Not in-place.
            # size of curr_dist_mat is updated each iteration
            for s_ind in range(curr_dist_mat.shape[0]):
                # i_k'th neighbor distance of sample s_ind
                if sorted_curr_dist_mat[i_k, s_ind] <= i_d_cutoff:
                    kept_curr_s_inds.append(s_ind)
            # no change of value. No need to copy.
            curr_dist_mat = curr_dist_mat[np.ix_(kept_curr_s_inds,
                                                 kept_curr_s_inds)]
            sorted_curr_dist_mat = np.sort(curr_dist_mat, axis=0)
            curr_s_inds = curr_s_inds[kept_curr_s_inds]
            progress_list.append(curr_s_inds.tolist())
        # thes last one of progress_list is equal to the curr_s_inds
        return curr_s_inds.tolist(), progress_list

    def detect_rare_samples(self, k, d_cutoff, n_iter, nprocs=1, metric=None,
                            use_pca=False, use_hnsw=False,
                            index_params=None, query_params=None):
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
        metric: {'cosine', 'euclidean', None}
            If none, self._sdm._metric is used.
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
        nprocs: int
            N processes to run all parameter tuples.

        Returns
        -------
        res_list
            Indices of non-rare samples of each corresponding parameter
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

        param_tups = [(k_list[i], d_cutoff_list[i], n_iter_list[i],
                       metric, use_pca, use_hnsw, index_params,
                       query_params)
                      for i in range(n_param_tups)]
        nprocs = int(nprocs)
        nprocs = min(nprocs, n_param_tups)

        # returns (filtered_sdm, progress_list (list of kept indices))
        if self._sdm._use_pdist:
            res_list = utils.parmap(
                lambda ptup: self._pdist_rare_s_detect(*ptup),
                param_tups, nprocs)
        else:
            res_list = utils.parmap(
                lambda ptup: self._no_pdist_rare_s_detect(*ptup),
                param_tups, nprocs)


        for i in range(n_param_tups):
            # only use k, d, and n_iter for res saving
            param_key = param_tups[i][:3]
            if param_key not in self._res_lut:
                self._res_lut[param_key] = res_list[i]
        # print(res_list)
        return [res[0] for res in res_list]
