import numpy as np

from .. import eda
from .. import utils


class SampleKNNFilter(object):
    """
    K nearest neighbor filter on sample distance matrix. 

    Perform filter with multiple filters in parallel, with each combination 
    of parameters as a process. Because this filter runs iteratively, 
    parallelizing each individual parameter combination run is not implemented.

    Stores the results for further lookup.

    Parameters
    ----------
    sdm: SampleDistanceMatrix or its subclass
    
    Attributes
    ----------
    _sdm: SampleDistanceMatrix
    _filter_res_lut: dict
        lookup table of KNN filter results
    """
    def __init__(self, sdm):
        super(SampleKNNFilter, self).__init__()
        self._sdm = sdm
        self._res_lut = {}

    def _ind_param_tup_knn_filter_samples(self, k, d_cutoff, n_iter):
        """
        KNN filter on samples with scalar tuple of parameters. 
        """
        d_max = self._sdm._d.max()
        curr_dist_mat = self._sdm._d.copy()
        curr_ind = np.arange(curr_dist_mat.shape[0])
        progress_list = []
        progress_list.append((0, curr_ind))

        for i in range(1, n_iter+1):
            i_d_cutoff = d_cutoff + (n_iter - i) / n_iter * (d_max - d_cutoff)
            i_k = min(curr_dist_mat.shape[0], k)
            kept_curr_cell_ind = []

            sorted_curr_dist_mat = np.sort(curr_dist_mat, axis=0)
            for j in range(curr_dist_mat.shape[0]):
                if sorted_curr_dist_mat[i_k, j] <= i_d_cutoff:
                    kept_curr_cell_ind.append(j)

            curr_dist_mat = curr_dist_mat[np.ix_(
                kept_curr_cell_ind, kept_curr_cell_ind)].copy()
            curr_ind = curr_ind[kept_curr_cell_ind]
            progress_list.append((i, curr_ind.tolist()))

        return (curr_ind.tolist(), progress_list)

    def knn_filter_samples(self, k, d_cutoff, n_iter, nprocs=1):
        """
        KNN filter on samples with multiple parameter combinations. 

        Assuming that there are at least k samples look similar in this 
        dataset, the samples with less than k similar neighbors may be 
        outliers. The outliers can either be really distinct from the general 
        populaton or caused by technical errors. 

        This filter iteratively filters samples according to their k-th nearest
        neighbors. The samples most distinct from its k-th nearest neighbors 
        are removed first. Then, the left samples are filtered by less 
        stringent distance cutoff. The distance cutoff decreases linearly 
        from maximum distance to d_cutoff with n_iter iterations.

        Parameters
        ----------
        k: int list or scalar
            K-th neighbor to cutoff
        d_cutoff: float list or scalar
            Minimum (>=) distance to be called as distinct. Samples with >=
            d_cutoff distances are distinct from each other.
        n_iter: int list or scalar
            N progressive iNN filters on the dataset. See description for more 
            details.
        nproces: int
            N processes to run all parameter tuples. 

        Returns
        -------
        res_list
            Filtered indices of each corresponding parameter tuple.

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

        This filter can be applied on filters by transforming the data matrix 
        when creating SampleDistanceMatrix, but it might not be
        very meaningful. For example, if there is 100 features all have values
        of 1 across all samples, they will always be kept with KNN filter
        strategy with k < 100. 
        """
        # Convert scalar to list
        try:
            k_list = list(k)
        except TypeError:
            k_list = [k]
        try:
            d_cutoff_list = list(d_cutoff)
        except TypeError:
            d_cutoff_list = [d_cutoff]
        try:
            n_iter_list = list(n_iter)
        except TypeError:
            n_iter_list = [n_iter]
        # Check all param lists have the same length
        if not (len(k_list) == len(d_cutoff_list) == len(n_iter_list)):
            raise ValueError("Parameter should have the same length."
                             "k: {}, d_cutoff: {}, n_iter: {}.".format(
                                k, d_cutoff, n_iter))
        n_param_tups = len(k_list)
        # type check all parameters
        for i in range(n_param_tups):
            if k_list[i] < 1:
                raise ValueError("k should be >= 1. k: {}".format(k))
            else:
                k_list[i] = int(k_list[i])

            if d_cutoff_list[i] <= 0:
                raise ValueError("d_cutoff should be >0 1. "
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
        # TODO: implement cache access
        f = lambda ptup: self._ind_param_tup_knn_filter_samples(*ptup)

        if nprocs != 1:
            res_list = utils.parmap(f, param_tups, nprocs)
        else:
            res_list = list(map(f, param_tups))

        for i in range(n_param_tups):
            self._res_lut[param_tups[i]] = res_list[i][1]

        return [res_list[i][0] for i in range(n_param_tups)]


def remove_constant_features(sfm):
    """
    Remove features that are constant across all samples
    """
    # boolean matrix of whether x == first column (feature)
    x_not_equal_to_1st_row = sfm._x != sfm._x[0]
    non_const_f_bool_ind = x_not_equal_to_1st_row.sum(axis=0) >= 1
    return sfm.ind_x(selected_f_inds=non_const_f_bool_ind)
