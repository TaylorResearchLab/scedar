import numpy as np
import scxplit.eda as eda

def test_sort_sids():
    qsids = np.array([0, 1, 5, 3, 2, 4])
    qlabs = np.array([0, 0, 2, 1, 1, 1,])
    rsids = np.array([3, 4, 2, 5, 1, 0])
    rs_qsids, rs_qlabs = eda.sort_sids(qsids, qlabs, rsids)
    assert np.all(rs_qsids == np.array([3, 4, 2, 5, 1, 0]))
    assert np.all(rs_qlabs == np.array([1, 1, 1, 2, 0, 0]))


def test_filter_min_cl_n():
    sids = np.array([0, 1, 2, 3, 4, 5])
    labs = np.array([0, 0, 0, 1, 2, 2])
    min_cl_n = 2
    mcnf_sids, mcnf_labs = eda.filter_min_cl_n(sids, labs, min_cl_n)
    assert np.all(mcnf_sids == np.array([0, 1, 2, 4, 5]))
    assert np.all(mcnf_labs == np.array([0, 0, 0, 2, 2]))

def test_cross_labs():
    rlabs = np.array([0, 0, 0, 1, 1])
    qlabs = np.array([1, 1, 0, 2, 3])
    cross_lab_lut = eda.cross_labs(rlabs, qlabs)
    test_lut = {
        0 : (3, ((0, 1), (1, 2))),
        1 : (2, ((2, 3), (1, 1)))
    }
    assert cross_lab_lut == test_lut