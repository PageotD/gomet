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
    # >> CODING
    # ------------------------------------------------------------------------
    
    # >> Nearest power of 2
    
    def test_nearPower2_low(self):
        bounds = [(-10, 10, 256), (-10, 10, 256)]
        test = ga(rosen, bounds, popsize=100)
        assert(test.nearPower2(240) == 256)
    
    def test_nearPower2_high(self):
        bounds = [(-10, 10, 256), (-10, 10, 256)]
        test = ga(rosen, bounds, popsize=100)
        assert(test.nearPower2(260) == 256)
        
    def test_nearPower2_exact(self):
        bounds = [(-10, 10, 256), (-10, 10, 256)]
        test = ga(rosen, bounds, popsize=100)
        assert(test.nearPower2(256) == 256)
    
    # >> Integer to binary string conversion
    
    def test_codInt2bin_min(self):
        bounds = [(-10, 10, 256), (-10, 10, 256)]
        test = ga(rosen, bounds, popsize=100)
        assert(test.codInt2bin(0, nsample=256) == '00000000')
         
    def test_codInt2bin_max(self):
        bounds = [(-10, 10, 256), (-10, 10, 256)]
        test = ga(rosen, bounds, popsize=100)
        assert(test.codInt2bin(255, nsample=256) == '11111111')
    
    def test_codInt2bin_intermediate(self):
        bounds = [(-10, 10, 256), (-10, 10, 256)]
        test = ga(rosen, bounds, popsize=100)
        assert(test.codInt2bin(44, nsample=256) == '00101100')
        
    # >> Binary string to integer conversion
    
    def test_codBin2int_min(self):
        bounds = [(-10, 10, 256), (-10, 10, 256)]
        test = ga(rosen, bounds, popsize=100)
        assert(test.codBin2int('00000000') == 0)
        
    def test_codBin2int_max(self):
        bounds = [(-10, 10, 256), (-10, 10, 256)]
        test = ga(rosen, bounds, popsize=100)
        assert(test.codBin2int('11111111') == 255)
    
    def test_codBin2int_intermediate(self):
        bounds = [(-10, 10, 256), (-10, 10, 256)]
        test = ga(rosen, bounds, popsize=100)
        assert(test.codBin2int('00101100') == 44)
        
    # >> Binary string to gray code conversion
    
    def test_codBin2gray_min(self):
        bounds = [(-10, 10, 256), (-10, 10, 256)]
        test = ga(rosen, bounds, popsize=100)
        assert(test.codBin2gray('00000000') == '00000000')
        
    def test_codBin2gray_max(self):
        bounds = [(-10, 10, 256), (-10, 10, 256)]
        test = ga(rosen, bounds, popsize=100)
        assert(test.codBin2gray('11111111') == '10000000')
    
    def test_codBin2gray_intermediate(self):
        bounds = [(-10, 10, 256), (-10, 10, 256)]
        test = ga(rosen, bounds, popsize=100)
        assert(test.codBin2gray('00101100') == '00111010')
        
    # >> Gray code to binary string conversion
    
    def test_codGray2bin_min(self):
        bounds = [(-10, 10, 256), (-10, 10, 256)]
        test = ga(rosen, bounds, popsize=100)
        assert(test.codGray2bin('00000000') == '00000000')
        
    def test_codGray2bin_max(self):
        bounds = [(-10, 10, 256), (-10, 10, 256)]
        test = ga(rosen, bounds, popsize=100)
        assert(test.codGray2bin('10000000') == '11111111')
    
    def test_codGray2bin_intermediate(self):
        bounds = [(-10, 10, 256), (-10, 10, 256)]
        test = ga(rosen, bounds, popsize=100)
        assert(test.codGray2bin('00111010') == '00101100')
        
    # >> Integer to gray code conversion
    
    def test_codInt2gray_min(self):
        bounds = [(-10, 10, 256), (-10, 10, 256)]
        test = ga(rosen, bounds, popsize=100)
        assert(test.codInt2gray(0, nsample=256) == '00000000')
         
    def test_codInt2gray_max(self):
        bounds = [(-10, 10, 32), (-10, 10, 32)]
        test = ga(rosen, bounds, popsize=100)
        assert(test.codInt2gray(255, nsample=256) == '10000000')
    
    def test_codInt2gray_intermediate(self):
        bounds = [(-10, 10, 32), (-10, 10, 32)]
        test = ga(rosen, bounds, popsize=100)
        assert(test.codInt2gray(44, nsample=256) == '00111010')
        
    # >> Gray code to integer conversion
    
    def test_codGray2int_min(self):
        bounds = [(-10, 10, 32), (-10, 10, 32)]
        test = ga(rosen, bounds, popsize=100)
        assert(test.codGray2int('00000000') == 0)
        
    def test_codGray2int_max(self):
        bounds = [(-10, 10, 32), (-10, 10, 32)]
        test = ga(rosen, bounds, popsize=100)
        assert(test.codGray2int('10000000') == 255)
    
    def test_codGray2int_intermediate(self):
        bounds = [(-10, 10, 32), (-10, 10, 32)]
        test = ga(rosen, bounds, popsize=100)
        assert(test.codGray2int('00111010') == 44)
    
    # ------------------------------------------------------------------------
    # >> POPULATION
    # ------------------------------------------------------------------------
    
# class TestGeneticAlgorithm:
    
#     # ------------------------------------------------------------------------
#     # >> MODES AVALAIBLES
#     # ------------------------------------------------------------------------
#     # def test_selection_modes(self):
#     #     assert('tournament' in ga._selection)
#     #     assert('roulette' in ga._selection)
        
#     # def test_crossover_modes(self):
#     #     assert('single' in ga._crossover)
#     #     assert('multiple' in ga._crossover)
        
#     # def test_mutation_modes(self):
#     #     assert('uniform' in ga._mutation)
#     #     assert('linear' in ga._mutation)
#     #     assert('exponential' in ga._mutation)
        
#     # ------------------------------------------------------------------------
#     # >> INITIALIZATION
#     # ------------------------------------------------------------------------
#     def test_initialization(self):
#         bounds = [(-10, 10, 32), (-10, 10, 32)]
#         ga2 = ga(rosen, bounds, popsize=100)
#         assert(ga2.bounds == bounds)
        
#     def test_initialization_strategies(self):
#         bounds = [(-10, 10, 32), (-10, 10, 32)]
#         ga2 = ga(rosen, bounds, selection='tournament',
#                  crossover='single', mutation='linear')
#         assert(ga2.popsize == 100)
        
#     def test_initialization_popsize(self):
#         bounds = [(-10, 10, 32), (-10, 10, 32)]
#         ga2 = ga(rosen, bounds, popsize=500)
#         assert(ga2.popsize == 500)
     
#     def test_initialization_maxiter(self):
#         bounds = [(-10, 10, 32), (-10, 10, 32)]
#         ga2 = ga(rosen, bounds, maxiter=200)
#         assert(ga2.maxiter == 200)

#     # ------------------------------------------------------------------------
#     # >> INITIALIZE CHROMOSOME POOL
#     # ------------------------------------------------------------------------
#     def test_initialize_pool(self):
#         # Initialize random number generator
#         np.random.seed(0)
#         output = ['01100:01111:10101', 
#                   '00000:00011:11011', 
#                   '00011:00111:01001', 
#                   '10011:10101:10010']
#         bounds = [(-10, 10, 32), (-10, 10, 32), (-10, 10, 32)]
#         ga2 = ga(rosen, bounds, popsize=4)
#         ga2._initialize_pool()
#         assert(ga2.current == output)
#         chromosomes = [obj.chromosome for obj in ga2.pool]
#         assert(chromosomes == output)
        
#     # ------------------------------------------------------------------------
#     # >> EVALUATE POOL
#     # ------------------------------------------------------------------------
#     def test_evaluate_pool(self):
#         # Initialize random number generator
#         np.random.seed(0)
#         output = [4137.901143558185, 1499969.7544094827, 
#                   614853.8981593271, 12300.445016409998]
#         bounds = [(-10, 10, 32), (-10, 10, 32), (-10, 10, 32)]
#         ga2 = ga(rosen, bounds, popsize=4, maxiter=1)
#         ga2._initialize_pool()
#         ga2._evaluate_pool()
#         fitness = [obj.fitness for obj in ga2.pool]
#         assert_almost_equal(fitness, output, decimal=6)
        
#     # ------------------------------------------------------------------------
#     # >> SELECTION STRATEGIES
#     # ------------------------------------------------------------------------
#     def test_tournament_selection(self):
#         # Initialize random number generator
#         np.random.seed(0)
#         output = ['00100:10111:00110', '00011:00111:01001', 
#                   '00000:00011:11011', '00000:00011:11011', 
#                   '00011:00111:01001', '00011:00111:01001', 
#                   '00000:00011:11011', '00000:00011:11011', 
#                   '11010:00001:00110', '00011:00111:01001']
#         bounds = [(-10, 10, 32), (-10, 10, 32), (-10, 10, 32)]
#         ga2 = ga(rosen, bounds, popsize=10, maxiter=1)
#         ga2._initialize_pool()
#         ga2._evaluate_pool()
#         parents = ga2._selection()
#         newpop = [obj.chromosome for obj in parents]
#         assert(newpop == output)
        
#     # ------------------------------------------------------------------------
#     # >> CROSSOVER STRATEGIES
#     # ------------------------------------------------------------------------
#     def test_simple_crossover(self):
#         # Initialize random number generator
#         np.random.seed(8)
#         parent1 = '00000:00000:00000'
#         parent2 = '11111:11111:11111'
#         output1 = '00000:00011:11111'
#         output2 = '11111:11100:00000'
#         bounds = [(-10, 10, 32), (-10, 10, 32), (-10, 10, 32)]
#         ga2 = ga(rosen, bounds, popsize=4, maxiter=1, pc=1.0)
#         children1, children2 = ga2._crossover(parent1, parent2)
#         assert([children1, children2] == [output1, output2])
        
#     # ------------------------------------------------------------------------
#     # >> MUTATION STRATEGIES
#     # ------------------------------------------------------------------------
#     def test_standard_mutation(self):
#         # Initialize random number generator
#         np.random.seed(8)
#         input1 = '01010:10001:10000'
#         output = '10101:01110:01111'
#         bounds = [(-10, 10, 32), (-10, 10, 32), (-10, 10, 32)]
#         ga2 = ga(rosen, bounds, popsize=4, maxiter=1, pm=1.0)
#         children1 = ga2._mutation(input1)
#         assert(children1 == output)
        
if __name__ == "__main__" :
    np.testing.run_module_suite()