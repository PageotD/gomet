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
    # >> MODES AVALAIBLES
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
        bounds = [(-10, 10, 32), (-10, 10, 32)]
        ga2 = ga(rosen, bounds, popsize=100)
        assert(ga2.bounds == bounds)
        
    def test_initialization_strategies(self):
        bounds = [(-10, 10, 32), (-10, 10, 32)]
        ga2 = ga(rosen, bounds, selection='TourNaMEnt',
                 crossover='sIMplE', mutation='LINEAR')
        assert(ga2._selection == 'tournament')
        assert(ga2._crossover == 'simple')
        assert(ga2._mutation == 'linear')
        assert(ga2.popsize == 100)
        
    def test_initialization_popsize(self):
        bounds = [(-10, 10, 32), (-10, 10, 32)]
        ga2 = ga(rosen, bounds, popsize=500)
        assert(ga2.popsize == 500)
     
    def test_initialization_maxiter(self):
        bounds = [(-10, 10, 32), (-10, 10, 32)]
        ga2 = ga(rosen, bounds, maxiter=200)
        assert(ga2.maxiter == 200)
        
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

    # ------------------------------------------------------------------------
    # >> BINARY TO INTEGER CONVERSION
    # ------------------------------------------------------------------------
    def test_binary2integer(self):
        assert(ga._binary2integer(self, '00000000') == 0)
        assert(ga._binary2integer(self, '00101100') == 44)
        assert(ga._binary2integer(self, '11111111') == 255)
        
    # ------------------------------------------------------------------------
    # >> INITIALIZE CHROMOSOME POOL
    # ------------------------------------------------------------------------
    def test_initialize_pool(self):
        # Initialize random number generator
        np.random.seed(0)
        output = ['01100:01111:10101', 
                  '00000:00011:11011', 
                  '00011:00111:01001', 
                  '10011:10101:10010']
        bounds = [(-10, 10, 32), (-10, 10, 32), (-10, 10, 32)]
        ga2 = ga(rosen, bounds, popsize=4)
        ga2._initialize_pool()
        assert(ga2.current == output)
        
if __name__ == "__main__" :
    np.testing.run_module_suite()