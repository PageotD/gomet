from numpy.testing import assert_, assert_raises
import numpy as np

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 
                                                '../../gomet')))

from gomet.genetic_algorithm import GeneticAlgorithm as ga

class TestGeneticAlgorithm:
    
    def test_power2_lower(self):
        assert(ga._power2(self, 240) == 256)
        
    def test_power2_higher(self):
        assert(ga._power2(self, 260) == 512)
        
    def test_power2_exact(self):
        assert(ga._power2(self, 260) == 512)
        
if __name__ == "__main__" :
    np.testing.run_module_suite()