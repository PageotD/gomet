"""
genetic_algorithm: A Genetic Algorithm optimization as described in
Global Optimization Methods In Geophysical Inversion, Sen & Stoffa (2013)
"""

import numpy as np
from scipy.optimize import OptimizeResult
from scipy.optimize.optimize import _status_message

from scipy._lib._util import check_random_state

def genetic_algorithm(func, bounds, funcargs=(), selection='fitest',
                      mutation='standard', crossover='standard'):
    """

    Parameters
    ----------
    func : TYPE
        DESCRIPTION.
    bounds : TYPE
        DESCRIPTION.
    funcargs : TYPE, optional
        DESCRIPTION. The default is ().
    selection : TYPE, optional
        DESCRIPTION. The default is 'fitest'.
    mutation : TYPE, optional
        DESCRIPTION. The default is 'standard'.
    crossover : TYPE, optional
        DESCRIPTION. The default is 'standard'.

    Returns
    -------
    None.

    """
    
    pass