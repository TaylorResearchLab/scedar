import numpy as np
import scipy.spatial
import sklearn.manifold

import matplotlib as mpl
import matplotlib.gridspec


# Sort the clustered sample_ids with the reference order of another. 
def sort_sids(sids, clabs, ref_sids):
    assert isinstance(sids, np.ndarray)
    assert isinstance(clabs, np.ndarray)
    assert isinstance(ref_sids, np.ndarray)
    
    assert len(sids.shape) == 1
    assert len(clabs.shape) == 1
    assert len(ref_sids.shape) == 1
    
    uniq_sids = np.unique(sids)
    uniq_ref_sids = np.unique(ref_sids)
    assert len(sids) == len(uniq_sids)
    assert len(ref_sids) == len(uniq_ref_sids)
    assert len(clabs) == len(sids)
    
    assert np.all(uniq_sids == uniq_ref_sids)
    
    uniq_clabs = np.unique(clabs)
    
    sep_cl_sid_list = []
    sep_cl_clab_list = []
    sep_cl_single_sid_list = []
    
    def sort_flat_sids(query_sids, ref_sids):
        return ref_sids[np.in1d(ref_sids, query_sids)]
    
    # Break qsids and qclabs by qclusters
    # Sort each qcluster according to rsids
    # Keep the min sid of each qcluster for further sorting of the individual
    # clusters. 
    for iter_clab in uniq_clabs:
        iter_cl_sids = sids[clabs == iter_clab]
        sep_cl_sid_list.append(sort_flat_sids(iter_cl_sids, ref_sids))
        sep_cl_clab_list.append([iter_clab] * len(iter_cl_sids))
        sep_cl_single_sid_list.append(iter_cl_sids[0])

    ref_ordered_sep_cl_single_sid_list = list(sort_flat_sids(
        sep_cl_single_sid_list, ref_sids))
    
    sep_cl_ref_ordered_ind_arr = np.array([
        sep_cl_single_sid_list.index(x)
        for x in ref_ordered_sep_cl_single_sid_list])
    
    ref_sorted_clabs = np.concatenate(np.array(
        sep_cl_clab_list)[sep_cl_ref_ordered_ind_arr])
    ref_sorted_sids = np.concatenate(np.array(
        sep_cl_sid_list)[sep_cl_ref_ordered_ind_arr])
    
    assert np.all(clabs[np.argsort(sids)] 
        == ref_sorted_clabs[np.argsort(ref_sorted_sids)])
    assert np.all(np.unique(ref_sorted_sids) == uniq_sids)
    
    return (ref_sorted_sids, ref_sorted_clabs)


# See how two clustering criteria match with each other. 
def cross_labs(ref_labs, query_labs):
    ref_labs = np.array(ref_labs)
    query_labs = np.array(query_labs)
    
    uniq_rlabs, uniq_rlab_cnts = np.unique(ref_labs, return_counts=True)
    cross_lab_lut = {}
    for i in range(len(uniq_rlabs)):
        ref_ci_quniq = tuple(map(list,
                                 np.unique(query_labs[np.where(np.array(ref_labs) == uniq_rlabs[i])],
                                           return_counts=True)))
        cross_lab_lut[uniq_rlabs[i]] = (uniq_rlab_cnts[i], tuple(map(tuple, ref_ci_quniq)))

    return cross_lab_lut


def filter_min_cl_n(sids, labs, min_cl_n):
    uniq_lab_cnts = np.unique(labs, return_counts=True)
    sids, labds = np.array(sids), np.array(labs)
    nf_sid_ind = np.in1d(labs, (uniq_lab_cnts[0])[uniq_lab_cnts[1] >= min_cl_n])
    return (sids[nf_sid_ind], labs[nf_sid_ind])

