from numpy.testing import assert_, assert_raises
from scipy.optimize import rosen
import numpy as np

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 
                                                '../../gomet')))

from gomet.genetic_algorithm import GeneticAlgorithm as ga

class TestGeneticAlgorithm:
    
    # ------------------------------------------------------------------------
    # >> MODE AVALAIBLES
    # ------------------------------------------------------------------------
    def test_selection_modes(self):
        assert('tournament' in ga._selection_modes)
        assert('roulette' in ga._selection_modes)
        
    def test_crossover_modes(self):
        assert('simple' in ga._crossover_modes)
        assert('multiple' in ga._crossover_modes)
        
    def test_mutation_modes(self):
        assert('standard' in ga._mutation_modes)
        assert('linear' in ga._mutation_modes)
        assert('exponential' in ga._mutation_modes)
        
    # ------------------------------------------------------------------------
    # >> INITIALIZATION
    # ------------------------------------------------------------------------
    def test_initialization(self):
        bounds = [(-1, 1), (-1, 1)]
        ga.__init__(self, func=rosen, bounds=bounds)
        assert(self.bounds == bounds)
        
    # ------------------------------------------------------------------------
    # >> NEAREST POWER OF TWO
    # ------------------------------------------------------------------------
    def test_nearest_power2(self):
        assert(ga._nearest_power2(self, 240) == 256)
        assert(ga._nearest_power2(self, 260) == 256)
        assert(ga._nearest_power2(self, 256) == 256)
     
    # ------------------------------------------------------------------------
    # >> INTEGER TO BINARY CONVERSION
    # ------------------------------------------------------------------------
    def test_integer2binary(self):
        assert(ga._integer2binary(self, 0, 256) == '00000000')
        assert(ga._integer2binary(self, 44, 256) == '00101100')
        assert(ga._integer2binary(self, 255, 256) == '11111111')
        
if __name__ == "__main__" :
    np.testing.run_module_suite()