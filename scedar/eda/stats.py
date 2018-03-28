import numpy as np


def gc1d(x):
    """
    Compute Gini Index for 1D array.

    Refs
    ----
    [1] <http://mathworld.wolfram.com/GiniCoefficient.html>

    [2] Damgaard, C. and Weiner, J. "Describing Inequality in Plant Size 
    or Fecundity." Ecology 81, 1139-1142, 2000.

    [3] Dixon, P. M.; Weiner, J.; Mitchell-Olds, T.; and Woodley, R. 
    "Bootstrapping the Gini Coefficient of Inequality." 
    Ecology 68, 1548-1551, 1987.

    [4] Dixon, P. M.; Weiner, J.; Mitchell-Olds, T.; and Woodley, R. 
    "Erratum to 'Bootstrapping the Gini Coefficient of Inequality.' " 
    Ecology 69, 1307, 1988.

    [5] <https://en.wikipedia.org/wiki/Gini_coefficient>
    """
    x = np.array(x, dtype="float64")
    if x.ndim != 1:
        raise ValueError("Only support 1D array.")
    if x.shape[0] == 0:
        raise ValueError("Array size cannot be 0.")
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
