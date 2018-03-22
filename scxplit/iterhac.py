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
            kde, logdens = self.gaussian_kde_logdens(
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


