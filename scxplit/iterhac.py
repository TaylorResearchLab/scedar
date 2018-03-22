import numpy as np

import scipy.cluster.hierarchy as sph
import scipy.spatial as spspatial
import scipy.stats as spstats

from . import utils
from . import eda

class MultinomialMdl(object):
    """
    Encode discrete values using multinomial distribution
    Input:
    x: 1d array. Should be non-negative
    
    Notes:
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
    ZeroIdcGKdeMdl. Encode the 0s and non-0s using bernoulli distribution.
    Then, encode non-0s using gaussian kde. Finally, one ternary val indicates 
    all 0s, all non-0s, or otherwise

    Parameters:
    -----------
    x: 1d array. Should be non-negative

    Methods defined here:
    ---------------------
    gaussian_kde_bw(x, method="silverman") : 
        Estimate Gaussian kernel density estimation bandwidth for input x.
        x: 
            float array of shape (n_samples) or (n_samples, n_features)
        bandwidth_method: 
            passed to scipy.stats.gaussian_kde param bw_method
            'scott': Scott's rule of thumb.
            'silverman': Silverman's rule of thumb.
            constant: constant will be timed by x.std(ddof=1) internally, 
                because scipy times bw_method value by std. "Scipy weights its 
                bandwidth by the ovariance of the input data" [3].
            callable: scipy calls the function on self
        Ref: 
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
            self._zi_mdl = self._compute_zi_mdl()
            self._kde_mdl = self._compute_gaussian_kde_mdl()
            self._mdl = self._zi_mdl + self._kde_mdl
        else:
            self._zi_mdl = 0
            self._kde_mdl = 0
            self._mdl = 0

    @staticmethod
    def gaussian_kde_logdens(x, bandwidth_method="silverman",
                             ret_kernel=False):
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

    def _compute_zi_mdl(self):
        if self._k == self._n or self._k == 0:
            zi_mdl = np.log(3)
        else:
            p = self._k / self._n
            zi_mdl = (np.log(3) - self._k * np.log(p) -
                      (self._n - self._k) * np.log(1-p))
        return zi_mdl
    
    def _compute_gaussian_kde_mdl(self):
        if self._k == 0:
            kde = None
            logdens = None
            bw_factor = None
            # no non-zery vals. Indicator encoded by zi mdl. 
            kde_mdl = 0
        elif self._k == 1:
            kde = None
            logdens = None
            bw_factor = None
            # encode just single value or multiple values
            kde_mdl = np.log(2)
        else:
            logdens, kde = self.gaussian_kde_logdens(
                self._x_nonzero, bandwidth_method=self._bw_method,
                ret_kernel=True)
            kde_mdl = -logdens.sum() + np.log(2)
            bw_factor = kde.factor
        
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
    Hierarchical clustering tree. Implement simple tree operation routines. 
    HCT is binary unbalanced tree. 

    Attributes:
    -----------
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
        clabs, cids = self.cluster_id_list_to_clab_array([self.left_leaf_ids(), 
                                                          self.right_leaf_ids()],
                                                         self.leaf_ids())
        if return_subtrees:
            return clabs, cids, self.left(), self.right()
        else:
            return clabs, cids

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
    def cluster_id_list_to_clab_array(cl_id_list, id_list):
        # convert [[0, 1, 2], [3,4]] to np.array([0, 0, 0, 1, 1])
        # according to id_arr [0, 1, 2, 3, 4]

        # checks uniqueness
        eda.SampleFeatureMatrix.check_is_valid_sfids(id_list)

        if type(cl_id_list) != list:
            raise ValueError("cl_id_list must be a list: {}".format(cl_id_list))

        for x in cl_id_list:
            eda.SampleFeatureMatrix.check_is_valid_sfids(x)

        cl_id_mlist = np.concatenate(cl_id_list).tolist()
        eda.SampleFeatureMatrix.check_is_valid_sfids(cl_id_mlist)

        # np.unique returns sorted unique values
        if np.all(sorted(id_list) != sorted(cl_id_mlist)):
            raise ValueError("id_list should have the same ids as cl_id_list.")

        cl_id_dict = {}
        # iter_cl_ind : cluster index
        # iter_cl_ids: individual cluster list
        for iter_cl_ind, iter_cl_ids in enumerate(cl_id_list):
            assert len(iter_cl_ids) > 0
            for cid in iter_cl_ids:
                cl_id_dict[cid] = iter_cl_ind

        clab_list = [cl_id_dict[x] for x in id_list]
        return clab_list, id_list

    @staticmethod
    def hclust_tree(dmat, linkage="complete", n_eval_rounds = None, 
                    is_euc_dist=False, verbose = False):
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
                iter_lhct = HClustTree.hclust_tree(dmat, linkage = iter_ltype)
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
                      sep = "\n")

        dmat_sf = spspatial.distance.squareform(dmat)
        hac_z = sph.linkage(dmat_sf, method=linkage, optimal_ordering=False)
        return HClustTree(sph.to_tree(hac_z))


class MDLSampleDistanceMatrix(eda.SingleLabelClassifiedSamples):
    """
    MDLSampleDistanceMatrix inherits SingleLabelClassifiedSamples to offer MDL 
    operations. 
    """
    def __init__(self, x, labs, sids=None, fids=None, 
                 d=None, metric="correlation", nprocs=None):
        super(MDLSampleDistanceMatrix, self).__init__(x=x, labs=labs, 
                                                      sids=sid, fids=fid, 
                                                      d=d, metric=metric, 
                                                      nprocs=nprocs)

    @staticmethod
    def per_column_zigkmdl(x, nprocs=1, verbose=False):
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

        return col_mdl_list

    def no_lab_mdl(self, nprocs=1, verbose=False, ret_internal=False):
        # verbose is not implemented
        col_mdl_list = self.per_column_zigkmdl(self._x, nprocs, verbose)
        col_mdl_sum = sum(map(lambda zkmdl: zkmdl.mdl, col_mdl_list))
        if ret_internal:
            return (pc_zkmdl_sum, col_mdl_list)
        else:
            return pc_zkmdl_sum

    def lab_mdl(self, nprocs=1, verbose=False, ret_internal=False):
        n_uniq_labels = self._uniq_labs.shape[0]

        ulab_x_list = [self._x[self._labs == ulab, :] 
                       for ulab in self._uniq_labs]

        # MDL for points
        pts_mdl_list = [self.per_column_zigkmdl(x, nprocs, verbose)
                        for x in ulab_x_list]

        ulab_pts_mdl_sum_list = [sum(map(lambda zkmdl: zkmdl.mdl, 
                                         ulab_pts_mdl_list))
                                 for ulab_pts_mdl_list in pts_mdl_list]
        
        pts_mdl_sum = sum(ulab_pts_mdl_sum_list)
        
        lab_mdl = MultinomialMdl(self._labs)

        # bandwidth
        # np.log(2**32) = 22.18070977791825
        param_mdl = 22.18070977791825 * n_uniq_labels
        
        mdl = param_mdl + lab_mdl.mdl + pts_mdl_sum

        if ret_internal:
            return (mdl, (pts_mdl_list, ulab_pts_mdl_sum_list, pts_mdl_sum, 
                          lab_mdl, lab_mdl.mdl, param_mdl))
        else:
            return mdl


class MIRCH(eda.SampleDistanceMatrix):
    """
    MIRCH: MDL iteratively regularized clustering based on hierarchy.
    """
    def __init__(self, x, d=None, metric=None, sids=None, fids=None, nprocs=None):
        super(MIRCH, self).__init__(x=x, d=d, metric=metric, sids=sid, 
                                    fids=fid, nprocs=nprocs)

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
            return np.clip(a = height * (x - start) / width + lb,
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
        if ind_cl_n >= n or ind_cl_n <= 0:
            raise ValueError("ind_cl_n={} should < n={} and > 0".format(ind_cl_n, n))

        if minimax_n <= 0:
            raise ValueError("minimax_n shoud > 0. "
                             "minimax_n: {}".format(minimax_n))

        shrink_factor = MIRCH.bidir_ReLU(n / minimax_n, 2, 20)

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

        if minimax_n <= 0:
            raise ValueError("minimax_n shoud > 0. "
                             "minimax_n: {}".format(minimax_n))

        if maxmini_n <= 0:
            raise ValueError("maxmini_n shoud > 0. "
                             "maxmini_n: {}".format(maxmini_n))

        if minimax_n > maxmini_n:
            raise ValueError("minimax_n should be <= maxmini_n"
                             "minimax_n={}"
                             "maxmini_n={}".format(minimax_n, maxmini_n))

        lin_grow_start_ratio = maxmini_n / minimax_n
        lin_grow_end_ratio = lin_grow_start_ratio ** 2
        ratio_linear_grow_width = lin_grow_end_ratio - lin_grow_start_ratio

        stc_minimaxn_ratio = subtree_leaf_cnt / minimax_n
        subtree_complexity = MIRCH.bidir_ReLU(stc_minimaxn_ratio, 
                                              lin_grow_start_ratio, 
                                              lin_grow_end_ratio)

        split_progress_factor = ((subtree_leaf_cnt / n) ** 2)

        subtree_sum_spl_comp_factor = (0.5 * subtree_complexity 
                                       * split_progress_factor)
        return subtree_sum_spl_comp_factor
