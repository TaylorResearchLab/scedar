import numpy as np
from .. import eda

from .mdl import MultinomialMdl, ZeroIdcGKdeMdl, MDLSampleDistanceMatrix
from .hct import HClustTree


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

    TODO
    ----
    * Dendrogram representation of the splitting process.

    * Take HCTree as parameter. Computing it is non-trivial when n is large.

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
