import numpy as np

import scipy.cluster.hierarchy as sph
import scipy.spatial as spspatial
import scipy.stats as spstats

from . import utils
from . import eda


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


class HClustTree(object):
    """
    Hierarchical clustering tree. 

    Implement simple tree operation routines. HCT is binary unbalanced tree.

    Attributes
    ----------
    node : scipy.cluster.hierarchy.ClusterNode
        current node
    prev : HClustTree
        parent of current node

    """

    def __init__(self, node, prev=None):
        super(HClustTree, self).__init__()
        self._node = node
        self._prev = prev

        if node is None:
            left = None
            right = None
        else:
            left = node.get_left()
            right = node.get_right()

        self._left = left
        self._right = right

    @property
    def prev(self):
        return self._prev

    def count(self):
        if self._node is None:
            return 0
        else:
            return self._node.get_count()

    def left(self):
        return HClustTree(self._left, self)

    def left_count(self):
        return self.left().count()

    def right(self):
        return HClustTree(self._right, self)

    def right_count(self):
        return self.right().count()

    def leaf_ids(self):
        if self._node is None:
            return None
        else:
            return self._node.pre_order(lambda xn: xn.get_id())

    def left_leaf_ids(self):
        return self.left().leaf_ids()

    def right_leaf_ids(self):
        return self.right().leaf_ids()

    def bi_partition(self, return_subtrees=False):
        labs, sids = self.cluster_id_to_lab_list([self.left_leaf_ids(),
                                                        self.right_leaf_ids()],
                                                       self.leaf_ids())
        if return_subtrees:
            return labs, sids, self.left(), self.right()
        else:
            return labs, sids

    def n_round_bipar_cnt(self, n):
        assert n > 0
        nr_bipar_cnt_list = []
        curr_hct_list = [self]
        curr_hct_cnt_list = []
        next_hct_list = []
        for i in range(n):
            for iter_hct in curr_hct_list:
                iter_left = iter_hct.left()
                iter_right = iter_hct.right()
                next_hct_list += [iter_left, iter_right]
                curr_hct_cnt_list += [iter_left.count(), iter_right.count()]
            nr_bipar_cnt_list.append(curr_hct_cnt_list)

            curr_hct_list = next_hct_list
            next_hct_list = []
            curr_hct_cnt_list = []
        return nr_bipar_cnt_list

    @staticmethod
    def cluster_id_to_lab_list(cl_sid_list, sid_list):
        """
        Convert nested clustered ID list into cluster label list.
        
        For example, convert `[[0, 1, 2], [3,4]]` to `[0, 0, 0, 1, 1]`
        according to id_arr `[0, 1, 2, 3, 4]`

        Parameters

        cl_sid_list: list[list[id]]
            Nested list with each sublist as a sert of IDs from a cluster.
        sid_list: list[id]
            Flat list of lists.

        """

        # checks uniqueness
        # This guarantees that clusters are all non-empty
        eda.SampleFeatureMatrix.check_is_valid_sfids(sid_list)

        if type(cl_sid_list) != list:
            raise ValueError(
                "cl_sid_list must be a list: {}".format(cl_sid_list))

        for x in cl_sid_list:
            eda.SampleFeatureMatrix.check_is_valid_sfids(x)

        cl_id_mlist = np.concatenate(cl_sid_list).tolist()
        eda.SampleFeatureMatrix.check_is_valid_sfids(cl_id_mlist)

        # np.unique returns sorted unique values
        if sorted(sid_list) != sorted(cl_id_mlist):
            raise ValueError(
                "sid_list should have the same ids as cl_sid_list.")

        cl_ind_lut = {}
        # iter_cl_ind : cluster index
        # iter_cl_sids: individual cluster list
        for iter_cl_ind, iter_cl_sids in enumerate(cl_sid_list):
            for sid in iter_cl_sids:
                cl_ind_lut[sid] = iter_cl_ind

        lab_list = [cl_ind_lut[x] for x in sid_list]
        return lab_list, sid_list

    @staticmethod
    def hclust_tree(dmat, linkage="complete", n_eval_rounds=None,
                    is_euc_dist=False, optimal_ordering=False, verbose=False):
        dmat = np.array(dmat, dtype="float")
        dmat = eda.SampleDistanceMatrix.num_correct_dist_mat(dmat)

        n = dmat.shape[0]

        if linkage == "auto":
            try_linkages = ("single", "complete", "average", "weighted")

            if is_euc_dist:
                try_linkages += ("centroid", "median", "ward")

            if n_eval_rounds is None:
                n_eval_rounds = int(np.ceil(np.log2(n)))
            else:
                n_eval_rounds = int(np.ceil(max(np.log2(n), n_eval_rounds)))

            ltype_mdl_list = []
            for iter_ltype in try_linkages:
                iter_lhct = HClustTree.hclust_tree(dmat, linkage=iter_ltype)
                iter_nbp_cnt_list = iter_lhct.n_round_bipar_cnt(n_eval_rounds)
                iter_nbp_mdl_arr = np.array(list(map(
                    lambda x: MultinomialMdl(np.array(x)).mdl,
                    iter_nbp_cnt_list)))
                iter_nbp_mdl = np.sum(iter_nbp_mdl_arr
                                      / np.arange(1, n_eval_rounds + 1))
                ltype_mdl_list.append(iter_nbp_mdl)

            linkage = try_linkages[ltype_mdl_list.index(max(ltype_mdl_list))]

            if verbose:
                print(linkage, tuple(zip(try_linkages, ltype_mdl_list)),
                      sep="\n")

        dmat_sf = spspatial.distance.squareform(dmat)
        hac_z = sph.linkage(dmat_sf, method=linkage,
                            optimal_ordering=optimal_ordering)
        return HClustTree(sph.to_tree(hac_z))


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


class MIRAC(object):
    """
    MIRAC: MDL iteratively regularized agglomerative clustering.

    Parameters
    ----------
    x: float array
        Data.
    d: float array
        Distance matrix.
    metric: str
        Type of distance metric.
    sids: sid list
        List of sample ids.
    fids: fid list
        List of feature ids.
    nprocs: int
        Number of processes to run MIRAC parallely.
    cl_mdl_scale_factor: float
        Scale factor of cluster overhead mdl. 
    minimax_n: int
        Estimated minimum # samples in a cluster
    maxmini_n: int
        Estimated max # samples in a cluster. If none, 10 * minimax is used.
    linkage: str
        Linkage type for generating the hierarchy. 
    verbose: bool
        Print stats for each iteration.

    Attributes
    ----------
    _sdm: SampleDistanceMatrix
        Data and distance matrices. 
    _minimax_n: int
        Stored parameter.
    _maxmini_n: int
        Stored parameter.
    _cluster_s_ind: int array
        Clustered sample indices of the data matrix.
    _cluster_labs: label array
        Labels of clustered samples. 1-to-1 matching to _cluster_s_ind.
    _hac_tree_root: HClustTree
        Root node of the hierarchical agglomerative clustering tree.
    _run_log: str
        String containing the log of the MIRAC run. 

    """

    def __init__(self, x, d=None, metric=None, sids=None, fids=None,
                 nprocs=1, cl_mdl_scale_factor=1, minimax_n=25,
                 maxmini_n=None, linkage="complete", verbose=False):
        super(MIRAC, self).__init__()

        nprocs = int(np.ceil(nprocs))

        self._sdm = eda.SampleDistanceMatrix(x=x, d=d, metric=metric, sids=sids,
                                             fids=fids, nprocs=nprocs)

        is_euc_dist = (metric == "euclidean")

        if cl_mdl_scale_factor < 0:
            raise ValueError("cl_mdl_scale_factor should >= 0",
                             "cl_mdl_scale_factor: {}".format(cl_mdl_scale_factor))

        # assert minimax_n > 0
        minimax_n = int(minimax_n)
        if minimax_n <= 0:
            raise ValueError("minimax_n shoud > 0. "
                             "minimax_n: {}".format(minimax_n))

        if maxmini_n is None:
            maxmini_n = 10 * minimax_n
        else:
            # assert maxmini_n >= minimax_n
            maxmini_n = int(maxmini_n)
            if minimax_n > maxmini_n:
                raise ValueError("minimax_n should be <= maxmini_n"
                                 "minimax_n={}"
                                 "maxmini_n={}".format(minimax_n, maxmini_n))

        self._minimax_n = minimax_n
        self._maxmini_n = maxmini_n
        self._cluster_s_ind, self._cluster_labs = self._mirac(
            cl_mdl_scale_factor=cl_mdl_scale_factor,
            minimax_n=minimax_n, maxmini_n=maxmini_n, linkage=linkage,
            is_euc_dist=is_euc_dist, nprocs=nprocs, verbose=verbose)

    # lower upper bound
    # start, end, lb, ub should all be scalar
    @staticmethod
    def bidir_ReLU(x, start, end, lb=0, ub=1):
        if start > end:
            raise ValueError("start should <= end"
                             "start: {}. end: {}.".format(start, end))

        if lb > ub:
            raise ValueError("lb should <= ub"
                             "lower bound: {}. "
                             "upper bound: {}. ".format(start, end))

        if start < end:
            width = end - start
            height = ub - lb
            return np.clip(a=height * (x - start) / width + lb,
                           a_min=lb, a_max=ub)
        else:
            # start == end
            return np.where(x >= start, ub, lb)

    # S-shaped function
    # MDL upper bound of a homogenous cluster.
    #   Below -> stop slit.
    #   Above -> keep split.
    # Upper bound is corrected by the clustering status.
    # if even split -> upper bound is not changed.
    # if uneven split -> bigger sub-cl hMDL upper bound gets lower (more likely
    #                    to be heterogenous)
    #                 -> smaller sub-cl hMDL upper bound gets higher
    # shrink_factor: if n >> minimax_n, more likely to split.
    @staticmethod
    def spl_mdl_ratio(ind_cl_n, n, no_spl_mdl, minimax_n=25):
        if ind_cl_n > n or ind_cl_n <= 0:
            raise ValueError("ind_cl_n={} should <= n={} and > 0".format(
                ind_cl_n, n))

        if minimax_n <= 0:
            raise ValueError("minimax_n shoud > 0. "
                             "minimax_n: {}".format(minimax_n))

        shrink_factor = MIRAC.bidir_ReLU(n / minimax_n, 2, 20)

        ind_cl_r_to_n = ind_cl_n / n
        ind_cl_r = (ind_cl_n - n/2) / (n/2)

        if no_spl_mdl <= 0:
            ind_cl_corrected_r = ((ind_cl_r
                                   + ind_cl_r * (1 - np.abs(ind_cl_r))) / 2
                                  + 0.5)
        else:
            ind_cl_corrected_r = ((1 - np.sqrt(-np.abs(ind_cl_r) + 1))
                                  * np.sign(ind_cl_r) / 2 + 0.5)

        shr_ind_cl_corrected_r = ((ind_cl_corrected_r - ind_cl_r_to_n)
                                  * shrink_factor
                                  + ind_cl_r_to_n)

        shr_ind_cl_corrected_n = shr_ind_cl_corrected_r * n
        if shr_ind_cl_corrected_n < 1:
            shr_ind_cl_corrected_n = np.ceil(shr_ind_cl_corrected_n)
        elif shr_ind_cl_corrected_n > n - 1:
            shr_ind_cl_corrected_n = np.floor(shr_ind_cl_corrected_n)

        return shr_ind_cl_corrected_r * (1 - 1 / shr_ind_cl_corrected_n)

    # bisplit: split both subtrees if labeled mdl sum > threshold
    # if n is large, bisplit threshold increases. Prefer splitting.
    # maxmini_n: estimate of smallest large cluster size
    # minimax_n: estimate of largest smallest cluster size
    @staticmethod
    def bi_split_compensation_factor(subtree_leaf_cnt, n,
                                     minimax_n, maxmini_n):
        # compensation factor: large when iter_n >> minimax
        # and iter_n close to n
        if subtree_leaf_cnt <= 0:
            raise ValueError("subtree_leaf_cnt shoud > 0. "
                             "subtree_leaf_cnt: {}".format(subtree_leaf_cnt))
        if subtree_leaf_cnt > n:
            raise ValueError("subtree_leaf_cnt shoud < n. "
                             "subtree_leaf_cnt: {}. n: {}".format(
                                 subtree_leaf_cnt, n))
        # assert minimax_n > 0
        if minimax_n <= 0:
            raise ValueError("minimax_n shoud > 0. "
                             "minimax_n: {}".format(minimax_n))

        # assert maxmini_n >= minimax_n
        if minimax_n > maxmini_n:
            raise ValueError("minimax_n should be <= maxmini_n"
                             "minimax_n={}"
                             "maxmini_n={}".format(minimax_n, maxmini_n))

        lin_grow_start_ratio = maxmini_n / minimax_n
        lin_grow_end_ratio = lin_grow_start_ratio ** 2
        ratio_linear_grow_width = lin_grow_end_ratio - lin_grow_start_ratio

        stc_minimaxn_ratio = subtree_leaf_cnt / minimax_n
        subtree_complexity = MIRAC.bidir_ReLU(stc_minimaxn_ratio,
                                              lin_grow_start_ratio,
                                              lin_grow_end_ratio)

        split_progress_factor = ((subtree_leaf_cnt / n) ** 2)

        subtree_sum_spl_comp_factor = (0.5 * subtree_complexity
                                       * split_progress_factor)
        return subtree_sum_spl_comp_factor

    # Split by individual cluster
    # if sub_cl_mdl_sum < no_lab_mdl:
    #     split_into_all_sub_cl_with_minimax_to_final
    # else:
    #     if sub_cl_mdl < no_lab_mdl * threshold_ratio:
    #         split_with_minimax_to_final
    #     else:
    #         no_spl
    #
    # if all_sub_cl_no_spl:
    #     add_together_as_a_final_cl
    # else:
    #     add_individual_cluster_as_a_final_cl
    def _mirac(self, cl_mdl_scale_factor=1, minimax_n=25,
               maxmini_n=None, linkage="complete",
               is_euc_dist=False, nprocs=1, verbose=False):

        # iterative hierarchical agglomerative clustering
        # Input:
        # - cl_mdl_scale_factor: scale cluster overhead mdl
        # - minimax_n: estimated minimum # samples in a cluster
        # - maxmini_n: estimated max # samples in a cluster.
        #   If none, 10 * minimax is used.

        n_samples = self._sdm._x.shape[0]
        n_features = self._sdm._x.shape[1]

        hac_tree = HClustTree.hclust_tree(self._sdm._d, linkage=linkage,
                                          is_euc_dist=is_euc_dist,
                                          verbose=verbose)
        self._hac_tree_root = hac_tree
        self._run_log = ""

        curr_trees = [hac_tree]

        next_trees = []

        final_s_inds = []
        final_labs = []
        curr_final_lab = 0

        while len(curr_trees) != 0:
            # Split each of the hac tree in curr_trees
            for i in range(len(curr_trees)):
                # lst, rst: left_sub_tree, right_sub_tree
                labs, s_inds, lst, rst = curr_trees[i].bi_partition(
                    return_subtrees=True)
                s_cnt = len(s_inds)

                subtrees = [lst, rst]
                n_subtrees = len(subtrees)

                subtree_split_list = []
                subtree_split_type = None

                mdl_sdm = MDLSampleDistanceMatrix(
                    x=self._sdm._x[s_inds, :],
                    labs=labs,
                    metric=self._sdm._metric)

                no_lab_mdl = mdl_sdm.no_lab_mdl(nprocs, verbose=verbose)

                # subtree_subset_s_ind_list starts from 0 to mdl_sdm.shape[0]
                (subtree_subset_s_ind_list,
                 subtree_s_cnt_list,
                 subtree_mdl_list,
                 cluster_mdl) = mdl_sdm.lab_mdl(cl_mdl_scale_factor, nprocs,
                                                verbose)

                subtree_s_ind_list = [[s_inds[i] for i in x] 
                                      for x in subtree_subset_s_ind_list]
                # compensation factor: large when s_cnt >> minimax and s_cnt
                # close to n_samples
                bi_spl_cmps_factor = self.bi_split_compensation_factor(
                    s_cnt, n_samples, self._minimax_n, self._maxmini_n)

                if (sum(subtree_mdl_list)
                    < (no_lab_mdl
                       * (self.spl_mdl_ratio(s_cnt, s_cnt, no_lab_mdl,
                                             self._minimax_n)
                          + bi_spl_cmps_factor))):
                    subtree_split_type = "bi-spl"
                    for st_ind in range(n_subtrees):
                        st_n = subtree_s_cnt_list[st_ind]
                        if st_n <= self._minimax_n:
                            final_s_inds += subtree_s_ind_list[st_ind]
                            final_labs += [curr_final_lab] * st_n
                            curr_final_lab += 1
                            subtree_split_list.append("spl-minimax")
                        else:
                            next_trees.append(subtrees[st_ind])
                            subtree_split_list.append("spl")
                else:
                    subtree_split_type = "ind-spl"
                    for st_ind in range(n_subtrees):
                        st_n = subtree_s_cnt_list[st_ind]
                        st_spl_ratio = self.spl_mdl_ratio(st_n,
                                                          s_cnt,
                                                          no_lab_mdl,
                                                          self._minimax_n)
                        if (subtree_mdl_list[st_ind]
                                < (no_lab_mdl * st_spl_ratio)):
                            if st_n <= self._minimax_n:
                                subtree_split_list.append("no-spl-minimax")
                            else:
                                subtree_split_list.append("no-spl")
                        else:
                            if st_n <= self._minimax_n:
                                subtree_split_list.append("spl-minimax")
                            else:
                                subtree_split_list.append("spl")

                    # if both subcls are "no-spl", add them together as a single cluster
                    if (np.all(np.in1d(subtree_split_list,
                                       ("spl", "spl-minimax")))
                        or np.all(np.in1d(subtree_split_list,
                                          ("no-spl", "no-spl-minimax")))):
                        final_s_inds += s_inds
                        final_labs += [curr_final_lab] * s_cnt
                        curr_final_lab += 1
                    else:
                        for st_ind in range(n_subtrees):
                            st_n = subtree_s_cnt_list[st_ind]
                            if subtree_split_list[st_ind] in ("no-spl", "no-spl-minimax"):
                                final_s_inds += subtree_s_ind_list[st_ind]
                                final_labs += [curr_final_lab] * st_n
                                curr_final_lab += 1
                            else:
                                if subtree_split_list[st_ind] == "spl-minimax":
                                    final_s_inds += subtree_s_ind_list[st_ind]
                                    final_labs += [curr_final_lab] * st_n
                                    curr_final_lab += 1
                                else:
                                    # "spl"
                                    next_trees.append(subtrees[st_ind])

                curr_iter_run_log = str.format(
                    "no lab mdl: {}, subtree mdl: {}, "
                    "cluster_mdl: {}, subtree_n: {}, "
                    "split type: {}, "
                    "split: {}.\n",
                    no_lab_mdl,
                    subtree_mdl_list,
                    cluster_mdl,
                    subtree_s_cnt_list,
                    subtree_split_type,
                    subtree_split_list)
                self._run_log += curr_iter_run_log
                if verbose:
                    print(curr_iter_run_log, end="")

            curr_trees = next_trees
            next_trees = []

        return (np.array(final_s_inds), np.array(final_labs))
