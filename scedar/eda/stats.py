import numpy as np


def gc1d(x):
    """
    Compute Gini Index for 1D array.

    References
    ----------
    [1] http://mathworld.wolfram.com/GiniCoefficient.html

    [2] Damgaard, C. and Weiner, J. "Describing Inequality in Plant Size
    or Fecundity." Ecology 81, 1139-1142, 2000.

    [3] Dixon, P. M.; Weiner, J.; Mitchell-Olds, T.; and Woodley, R.
    "Bootstrapping the Gini Coefficient of Inequality."
    Ecology 68, 1548-1551, 1987.

    [4] Dixon, P. M.; Weiner, J.; Mitchell-Olds, T.; and Woodley, R.
    "Erratum to 'Bootstrapping the Gini Coefficient of Inequality.' "
    Ecology 69, 1307, 1988.

    [5] https://en.wikipedia.org/wiki/Gini_coefficient

    [6] https://github.com/oliviaguest/gini/blob/master/gini.py
    """
    x = np.array(x, dtype="float64")
    if x.ndim != 1:
        raise ValueError("Only support 1D array.")
    if x.shape[0] == 0:
        raise ValueError("Array size cannot be 0.")
    if x.shape[0] == 1:
        return np.nan
    # sort into ascending order
    xs = np.sort(x)
    if xs[0] == xs[-1]:
        return 0
    # Wrap with np.int64 to prevent division by 0
    n = np.int64(x.shape[0])
    xs_ranks = np.arange(1, n+1)
    xmean = x.mean()
    gc = ((2 * xs_ranks - n - 1) * xs).sum() / (n * (n-1) * xmean)
    return gc


# Multiple testing correction adapted from
# https://github.com/CoBiG2/cobig_misc_scripts/blob/master/FDR.py
#
# Following is the lincense info:
# Copyright 2017 Francisco Pina Martins <f.pinamartins@gmail.com>
# This file is part of structure_threader.
# structure_threader is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# structure_threader is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with structure_threader. If not, see <http://www.gnu.org/licenses/>.
#
# Taken from https://stackoverflow.com/a/21739593/3091595, ported to python 3
# and improved readability.
def multiple_testing_correction(pvalues, correction_type="FDR"):
    """
    Consistent with R.

    correct_pvalues_for_multiple_testing([0.0, 0.01, 0.029, 0.03, 0.031, 0.05,
                                          0.069, 0.07, 0.071, 0.09, 0.1])
    """
    pvalues = np.array(pvalues)
    sample_size = pvalues.shape[0]
    qvalues = np.empty(sample_size)
    if correction_type == "Bonferroni":
        # Bonferroni correction
        qvalues = sample_size * pvalues
    elif correction_type == "FDR":
        # Benjamini-Hochberg, AKA - FDR test
        values = [(pvalue, i) for i, pvalue in enumerate(pvalues)]
        values.sort()
        values.reverse()
        new_values = []
        for i, vals in enumerate(values):
            rank = sample_size - i
            pvalue, index = vals
            new_values.append((sample_size/rank) * pvalue)
        for i in range(0, int(sample_size)-1):
            if new_values[i] < new_values[i+1]:
                new_values[i+1] = new_values[i]
        for i, vals in enumerate(values):
            pvalue, index = vals
            qvalues[index] = new_values[i]
    else:
        raise ValueError("Not supported correction type "
                         "{}".format(correction_type))
    return np.clip(qvalues, 0, 1)


# lower upper bound
# start, end, lb, ub should all be scalar
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
