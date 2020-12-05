from numpy.testing import assert_, assert_raises, assert_almost_equal
from scipy.optimize import rosen
import numpy as np

import sys
import os
import operator
#from candidate import Candidate

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 
                                                '../../gomet')))

from gomet.genetic_algorithm import GeneticAlgorithm as ga

class TestGeneticAlgorithm:
    
    # ------------------------------------------------------------------------
    # >> MODES AVALAIBLES
    # ------------------------------------------------------------------------
    # def test_selection_modes(self):
    #     assert('tournament' in ga._selection)
    #     assert('roulette' in ga._selection)
        
    # def test_crossover_modes(self):
    #     assert('single' in ga._crossover)
    #     assert('multiple' in ga._crossover)
        
    # def test_mutation_modes(self):
    #     assert('uniform' in ga._mutation)
    #     assert('linear' in ga._mutation)
    #     assert('exponential' in ga._mutation)
        
    # ------------------------------------------------------------------------
    # >> INITIALIZATION
    # ------------------------------------------------------------------------
    def test_initialization(self):
        bounds = [(-10, 10, 32), (-10, 10, 32)]
        ga2 = ga(rosen, bounds, popsize=100)
        assert(ga2.bounds == bounds)
        
    def test_initialization_strategies(self):
        bounds = [(-10, 10, 32), (-10, 10, 32)]
        ga2 = ga(rosen, bounds, selection='tournament',
                 crossover='single', mutation='linear')
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
        chromosomes = [obj.chromosome for obj in ga2.pool]
        assert(chromosomes == output)
        
    # ------------------------------------------------------------------------
    # >> EVALUATE POOL
    # ------------------------------------------------------------------------
    def test_evaluate_pool(self):
        # Initialize random number generator
        np.random.seed(0)
        output = [4137.901143558185, 1499969.7544094827, 
                  614853.8981593271, 12300.445016409998]
        bounds = [(-10, 10, 32), (-10, 10, 32), (-10, 10, 32)]
        ga2 = ga(rosen, bounds, popsize=4, maxiter=1)
        ga2._initialize_pool()
        ga2._evaluate_pool()
        fitness = [obj.fitness for obj in ga2.pool]
        assert_almost_equal(fitness, output, decimal=6)
        
    # ------------------------------------------------------------------------
    # >> SELECTION STRATEGIES
    # ------------------------------------------------------------------------
    def test_tournament_selection(self):
        # Initialize random number generator
        np.random.seed(0)
        output = ['00100:10111:00110', '00011:00111:01001', 
                  '00000:00011:11011', '00000:00011:11011', 
                  '00011:00111:01001', '00011:00111:01001', 
                  '00000:00011:11011', '00000:00011:11011', 
                  '11010:00001:00110', '00011:00111:01001']
        bounds = [(-10, 10, 32), (-10, 10, 32), (-10, 10, 32)]
        ga2 = ga(rosen, bounds, popsize=10, maxiter=1)
        ga2._initialize_pool()
        ga2._evaluate_pool()
        parents = ga2._selection()
        newpop = [obj.chromosome for obj in parents]
        assert(newpop == output)
        
    # ------------------------------------------------------------------------
    # >> CROSSOVER STRATEGIES
    # ------------------------------------------------------------------------
    def test_simple_crossover(self):
        # Initialize random number generator
        np.random.seed(8)
        parent1 = '00000:00000:00000'
        parent2 = '11111:11111:11111'
        output1 = '00000:00011:11111'
        output2 = '11111:11100:00000'
        bounds = [(-10, 10, 32), (-10, 10, 32), (-10, 10, 32)]
        ga2 = ga(rosen, bounds, popsize=4, maxiter=1, pc=1.0)
        children1, children2 = ga2._crossover(parent1, parent2)
        assert([children1, children2] == [output1, output2])
        
    # ------------------------------------------------------------------------
    # >> MUTATION STRATEGIES
    # ------------------------------------------------------------------------
    def test_standard_mutation(self):
        # Initialize random number generator
        np.random.seed(8)
        input1 = '01010:10001:10000'
        output = '10101:01110:01111'
        bounds = [(-10, 10, 32), (-10, 10, 32), (-10, 10, 32)]
        ga2 = ga(rosen, bounds, popsize=4, maxiter=1, pm=1.0)
        children1 = ga2._mutation(input1)
        assert(children1 == output)
        
if __name__ == "__main__" :
    np.testing.run_module_suite()