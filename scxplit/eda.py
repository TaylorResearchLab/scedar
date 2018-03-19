import numpy as np
import scipy.spatial
import sklearn.manifold

import matplotlib as mpl
import matplotlib.colors
import matplotlib.gridspec
import seaborn as sns

from . import utils

class SampleFeatureMatrix(object):
    """
    SampleFeatureMatrix is a (n_samples, n_features) matrix. 
    In this package, we are only interested in float features as measured
    expression levels. 
    """
    def __init__(self, x, sids=None, fids=None):
        super(SampleFeatureMatrix, self).__init__()
        if x is None:
            raise ValueError("x cannot be None")
        else:
            try:
                x = np.array(x, dtype="float64")
            except ValueError as e:
                raise ValueError("Features must be float. {}".format(e))
            
            if x.ndim != 2:
                raise ValueError("x has shape (n_samples, n_features)")

            if x.size == 0:
                raise ValueError("size of x cannot be 0")

        if sids is None:
            sids = list(range(x.shape[0]))
        else:
            self.check_is_valid_sfids(sids)
            if len(sids) != x.shape[0]:
                raise ValueError("x has shape (n_samples, n_features)")

        if fids is None:
            fids = list(range(x.shape[1]))
        else:
            self.check_is_valid_sfids(fids)
            if len(fids) != x.shape[1]:
                raise ValueError("x has shape (n_samples, n_features)")

        self._x = x
        self._sids = np.array(sids)
        self._fids = np.array(fids)

    @staticmethod
    def is_valid_sfid(sid):
        return (type(sid) == str) or (type(sid) == int)

    @staticmethod
    def check_is_valid_sfids(sids):
        if sids is None:
            raise ValueError("[sf]ids cannot be None")

        if type(sids) != list:
            raise ValueError("[sf]ids must be a homogenous list of int or str")

        if len(sids) == 0:
            raise ValueError("[sf]ids must have >= 1 values")

        sid_types = tuple(map(type, sids))
        if len(set(sid_types)) != 1:
            raise ValueError("[sf]ids must be a homogenous list of int or str")

        if not SampleFeatureMatrix.is_valid_sfid(sids[0]):
            raise ValueError("[sf]ids must be a homogenous list of int or str")

        sids = np.array(sids)
        assert sids.ndim == 1
        assert sids.shape[0] > 0
        if not utils.is_uniq_np1darr(sids):
            raise ValueError("[sf]ids must not contain duplicated values")

    def get_sids(self):
        return self._sids.copy()

    def get_fids(self):
        return self._fids.copy()

    def get_x(self):
        return self._x.copy()



class SampleDistanceMatrix(object):
    """docstring for SampleDistanceMatrix"""
    def __init__(self, x, is_dist_mat=False, sids=None, nprocs=1):
        super(SampleDistanceMatrix, self).__init__()
        

class SingleLabelClassifiedSamples(SampleFeatureMatrix):
    """docstring for SingleLabelClassifiedSamples"""
    # sid, lab, fid, x
    def __init__(self, x, labs, sids=None, fids=None):
        # sids: sample IDs. String or int.
        # labs: sample classified labels. String or int. 
        # x: (n_samples, n_features)
        super(SingleLabelClassifiedSamples, self).__init__(x=x, sids=sids, 
                                                           fids=fids)
        self.check_is_valid_labs(labs)
        labs = np.array(labs)
        if self._sids.shape[0] != labs.shape[0]:
            raise ValueError("sids must have the same length as labs")
        self._labs = labs

        sid_lut = {}
        for uniq_lab in np.unique(labs):
            sid_lut[uniq_lab] = self._sids[labs == uniq_lab]
        self._sid_lut = sid_lut

        lab_lut = {}
        # sids only contain unique values
        for i in range(self._sids.shape[0]):
            lab_lut[self._sids[i]] = labs[i]
        self._lab_lut = lab_lut
        return

    @staticmethod
    def is_valid_lab(lab):
        return (type(lab) == str) or (type(lab) == int)

    @staticmethod
    def check_is_valid_labs(labs):
        if labs is None:
            raise ValueError("labs cannot be None")

        if type(labs) != list:
            raise ValueError("labs must be a homogenous list of int or str")
        
        if len(labs) == 0:
            raise ValueError("labs cannot be empty")

        if len(set(map(type, labs))) != 1:
            raise ValueError("labs must be a homogenous list of int or str")

        if not SingleLabelClassifiedSamples.is_valid_lab(labs[0]):
            raise ValueError("labs must be a homogenous list of int or str")

        labs = np.array(labs)
        assert labs.ndim == 1, "Labels must be 1D"
        assert labs.shape[0] > 0
        
    def filter_min_class_n(self, min_class_n):
        uniq_lab_cnts = np.unique(self._labs, return_counts=True)
        nf_sid_ind = np.in1d(self._labs, 
                             (uniq_lab_cnts[0])[uniq_lab_cnts[1] >= min_class_n])
        return (self._sids[nf_sid_ind], self._labs[nf_sid_ind])

    def labs_to_sids(self, labs):
        return tuple(tuple(self._sid_lut[y].copy()) for y in labs)

    def sids_to_labs(self, sids):
        return np.array([self._lab_lut[x] for x in sids])
    
    def get_labs(self):
        return self._labs.copy()

    # Sort the clustered sample_ids with the reference order of another. 
    # 
    # Sort sids according to labs
    # If ref_sid_order is not None:
    #   sort sids further according to ref_sid_order
    def lab_sorted_sids(self, ref_sid_order=None):
        sep_lab_sid_list = []
        sep_lab_list = []
        for iter_lab in sorted(self._sid_lut.keys()):
            iter_sid_arr = self._sid_lut[iter_lab]
            sep_lab_sid_list.append(iter_sid_arr)
            sep_lab_list.append(np.repeat(iter_lab, len(iter_sid_arr)))

        if ref_sid_order is not None:
            self.check_is_valid_sfids(ref_sid_order)
            ref_sid_order = np.array(ref_sid_order)
            # sort r according to q
            # assumes:
            # - r contains all elements in q
            # - r is 1d np array
            def sort_flat_sids(query_sids, ref_sids):
                return ref_sids[np.in1d(ref_sids, query_sids)]

            # sort inner sid list but maintains the order as sep_lab_list
            sep_lab_sid_list = [sort_flat_sids(x, ref_sid_order)
                                for x in sep_lab_sid_list]
            sep_lab_min_sid_list = [x[0] for x in sep_lab_sid_list]
            sorted_sep_lab_min_sid_list = list(sort_flat_sids(sep_lab_min_sid_list,
                                                              ref_sid_order))
            min_sid_sorted_sep_lab_ind_list = [sep_lab_min_sid_list.index(x)
                                               for x in sorted_sep_lab_min_sid_list]
            sep_lab_list = [sep_lab_list[i] for i in min_sid_sorted_sep_lab_ind_list]
            sep_lab_sid_list = [sep_lab_sid_list[i] for i in min_sid_sorted_sep_lab_ind_list]

        lab_sorted_sid_arr = np.concatenate(sep_lab_sid_list)
        lab_sorted_lab_arr = np.concatenate(sep_lab_list)
        
        # check sorted sids are the same set as original    
        assert np.all(np.sort(lab_sorted_sid_arr) == np.sort(self._sids))
        # check sorted labs are the same set as original
        assert np.all(np.sort(lab_sorted_lab_arr) == np.sort(self._labs))
        # check sorted (sid, lab) matchings are the same set as original
        assert np.all(lab_sorted_lab_arr[np.argsort(lab_sorted_sid_arr)] 
            == self._labs[np.argsort(self._sids)])

        return (lab_sorted_sid_arr, lab_sorted_lab_arr)

    # See how two clustering criteria match with each other.
    # When given q_slc_samples is not None, sids and labs are ignored. 
    # When q_slc_samples is None, sids and labs must be provided
    def cross_labs(self, q_slc_samples):
        if not isinstance(q_slc_samples, SingleLabelClassifiedSamples):
            raise TypeError("Query should be an instance of "
                            "SingleLabelClassifiedSamples")
        
        try:
            ref_labs = np.array([self._lab_lut[x] 
                                 for x in q_slc_samples.get_sids()])
        except KeyError as e:
            raise ValueError("query sid {} is not in ref sids.".format(e))

        query_labs = q_slc_samples.get_labs()
        
        uniq_rlabs, uniq_rlab_cnts = np.unique(ref_labs, return_counts=True)
        cross_lab_lut = {}
        for i in range(len(uniq_rlabs)):
            # ref cluster i. query unique labs.
            ref_ci_quniq = tuple(map(list, np.unique(
                query_labs[np.where(np.array(ref_labs) == uniq_rlabs[i])],
                return_counts=True)))
            cross_lab_lut[uniq_rlabs[i]] = (uniq_rlab_cnts[i], tuple(map(tuple, ref_ci_quniq)))

        return cross_lab_lut

    def labs_to_cmap(self, return_lut=False):
        uniq_lab_arr = np.unique(self._labs)
        num_uniq_labs = len(uniq_lab_arr)

        uniq_lab_lut = dict(zip(range(num_uniq_labs), uniq_lab_arr))
        uniq_ind_lut = dict(zip(uniq_lab_arr, range(num_uniq_labs)))
        
        lab_ind_arr = np.array([uniq_ind_lut[x] for x in self._labs])

        lab_col_list = sns.hls_palette(num_uniq_labs)
        lab_cmap = mpl.colors.ListedColormap(lab_col_list)

        lab_col_lut = dict(zip([uniq_lab_lut[i] for i in range(len(uniq_lab_arr))],
                               lab_col_list))

        if return_lut:
            return (lab_cmap, lab_ind_arr, lab_col_lut, uniq_lab_lut)
        else:
            return lab_cmap


