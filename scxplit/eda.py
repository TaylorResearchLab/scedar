import numpy as np
import scipy.spatial
import sklearn.manifold

import matplotlib as mpl
import matplotlib.gridspec

from . import utils

class SingleLabelClassifiedSamples(object):
    """docstring for SingleLabelClassifiedSamples"""
    def __init__(self, sids, labs, x=None):
        # sids: sample IDs. String or int.
        # labs: sample classified labels. String or int. 
        # x: (n_samples, n_features)
        super(SingleLabelClassifiedSamples, self).__init__()
        self.assert_is_valid_sids(sids)
        self.assert_is_valid_labs(labs)
        sids = np.array(sids)
        labs = np.array(labs)
        assert sids.shape[0] == labs.shape[0]
        if x is not None:
            x = np.array(x)
            assert len(x.shape) == 2
            assert x.shape[0] == sids

        self._n = sids.shape[0]
        self._sids = sids
        self._labs = labs
        self._x = x

        sid_lut = {}
        for uniq_lab in np.unique(labs):
            sid_lut[uniq_lab] = sids[labs == uniq_lab]
        self._sid_lut = sid_lut

        lab_lut = {}
        # sids only contain unique values
        for i in range(sids.shape[0]):
            lab_lut[sids[i]] = labs[i]
        self._lab_lut = lab_lut
        return

    @staticmethod
    def is_valid_sid(sid):
        return (type(sid) == str) or (type(sid) == int)

    @staticmethod
    def assert_is_valid_sids(sids):
        assert sids is not None
        assert type(sids) == list
        assert len(sids) > 0
        sid_types = tuple(map(type, sids))
        assert len(set(sid_types)) == 1
        assert SingleLabelClassifiedSamples.is_valid_sid(sids[0])
        sids = np.array(sids)
        assert sids.ndim == 1
        assert sids.shape[0] > 0
        assert utils.is_uniq_np1darr(sids), 'Sample IDs must be 1D uniq array'

    @staticmethod
    def is_valid_lab(lab):
        return (type(lab) == str) or (type(lab) == int)

    @staticmethod
    def assert_is_valid_labs(labs):
        assert labs is not None
        assert type(labs) == list
        assert len(labs) > 0
        lab_types = tuple(map(type, labs))
        assert len(set(lab_types)) == 1
        assert SingleLabelClassifiedSamples.is_valid_lab(labs[0])
        labs = np.array(labs)
        assert labs.ndim == 1, 'Labels must be 1D'
        assert labs.shape[0] > 0
        
    def filter_min_class_n(self, min_class_n):
        uniq_lab_cnts = np.unique(self._labs, return_counts=True)
        nf_sid_ind = np.in1d(self._labs, 
                             (uniq_lab_cnts[0])[uniq_lab_cnts[1] >= min_class_n])
        return (self._sids[nf_sid_ind], self._labs[nf_sid_ind])

    def labs_to_sids(labs):
        return np.array([self._sid_lut[y].copy() for y in labs])

    def sids_to_labs(sids):
        return np.array([self._lab_lut[x] for x in labs])
    
    def get_sids(self):
        return self._sids.copy()

    def get_labs(self):
        return self._labs.copy()

    def get_x(self):
        return self._x.copy()

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
            self.assert_is_valid_sids(ref_sid_order)
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
    def cross_labs(self, q_slc_samples=None, qsids=None, qlabs=None):
        if q_slc_samples is None:
            q_slc_samples = SingleLabelClassifiedSamples(qsids, qlabs)
            
        if not isinstance(q_slc_samples, SingleLabelClassifiedSamples):
            raise TypeError('Query should be an instance of '
                            'SingleLabelClassifiedSamples')
        
        try:
            ref_labs = np.array([self._lab_lut[x] 
                                 for x in q_slc_samples.get_sids()])
        except KeyError as e:
            raise ValueError('query sid {} is not in ref sids.'.format(e))

        query_labs = q_slc_samples.get_labs()
        
        uniq_rlabs, uniq_rlab_cnts = np.unique(ref_labs, return_counts=True)
        cross_lab_lut = {}
        for i in range(len(uniq_rlabs)):
            ref_ci_quniq = tuple(map(list,
                                     np.unique(query_labs[np.where(np.array(ref_labs) == uniq_rlabs[i])],
                                               return_counts=True)))
            cross_lab_lut[uniq_rlabs[i]] = (uniq_rlab_cnts[i], tuple(map(tuple, ref_ci_quniq)))

        return cross_lab_lut

