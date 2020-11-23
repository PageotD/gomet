"""
genetic_algorithm: A Genetic Algorithm optimization as described in
Global Optimization Methods In Geophysical Inversion, Sen & Stoffa (2013)
"""

import math
import numpy as np
from scipy.optimize import OptimizeResult
from scipy.optimize.optimize import _status_message

from scipy._lib._util import check_random_state

class GeneticAlgorithm():
    """
    A standard Genetic Algorithm implementation according to 
    Sen & Stoffa (1992), Sambridge & Gallagher (1993), 
    Gallagher & Sambridge (1994).
    
    Gallagher, K. & Sambridge, M. Genetic algorithms: a powerful tool for 
    large-scale nonlinear optimization problems Computers & Geosciences, 
    Elsevier, 1994, 20, 1229-1236
    
    Sambridge, M. & Gallagher, K. Earthquake hypocenter location using genetic
    algorithms Bulletin of the Seismological Society of America, 
    Seismological Society of America, 1993, 83, 1467-1491
    
    Sen, M. K. & Stoffa, P. L. Rapid sampling of model space using genetic 
    algorithms: examples from seismic waveform inversion Geophysical 
    Journal International, Oxford University Press, 1992, 108, 281-292
    """
    
    # ------------------------------------------------------------------------
    # >> Define avalaible selection modes, crossover modes and mutation modes 
    #    in lists
    # ------------------------------------------------------------------------
    _selection_modes = ['tournament', 'roulette']
    _crossover_modes = ['simple', 'multiple']
    _mutation_modes  = ['standard', 'linear', 'exponential']

    
    # ------------------------------------------------------------------------
    # >> INITIALIZE
    # ------------------------------------------------------------------------
    def __init__(self, func, bounds, fargs=(), popsize=100, maxiter=100, 
                 selection='tournament', crossover='simple', 
                 mutation='standard'):
       
        # Initiate class
        self.func = func
        self.fargs = fargs
        self.maxiter = maxiter
        self.bounds = bounds
          
        # Check bounds
        if isinstance(bounds, list):
            if isinstance(bounds[0], tuple) and len(bounds[0]) != 3:
                raise Exception("bounds must be a tuple or a list of tuples (min, max number of samples)")
        elif isinstance(bounds, tuple) and len(bounds) != 3:
            raise Exception("bounds must be a tuple or a list of tuples (min, max number of samples)")
        else:
            self.bounds = bounds
            
        # Check population
        if (popsize % 2 == 0 and popsize >= 2):
            self.popsize = popsize
        else:
            raise ValueError("Population must be a multiple of 2 and >= 2")
            
        # Check selection mode
        if selection.lower() in self._selection_modes:
            self._selection = selection.lower()
        else:
            raise ValueError("Please select a valid selection strategy")
        
        # Check crossover mode
        if crossover.lower() in self._crossover_modes:
            self._crossover = crossover.lower()
        else:
            raise ValueError("Please select a valid crossover strategy")
            
        # Check crossover mode
        if mutation.lower() in self._mutation_modes:
            self._mutation = mutation.lower()
        else:
            raise ValueError("Please select a valid mutation strategy")
        
    # ------------------------------------------------------------------------
    # >> NEAREST POWER OF 2
    # ------------------------------------------------------------------------
    def _nearest_power2(self, n):
        """
        Get the nearest power of 2 of an integer value.

        Parameters
        ----------
        n : int
            Integer value.

        Returns
        -------
        int
            Nearest power of 2.

        """
        
        # Get the log of n in base 2 and get the closest integer to n
        npower = round(math.log(n, 2))
        
        # Return the next power of 2
        return int(math.pow(2, npower))
    
    # ------------------------------------------------------------------------
    # >> INTEGER TO BINARY CONVERSION
    # ------------------------------------------------------------------------
    def _integer2binary(self, rgene, nsamples):
        """
        Return the gene (integer) in binary format with the appropriate lenght

        """

        # Calculate the maximum lenght of the bit-string
        # Old version : nelements = 1./np.log(2.)*np.log(float(nsamples))
        nelements = len(np.binary_repr(nsamples-1))
        
        # Convert the gene in binary format
        bgene = np.binary_repr(rgene, width=nelements)

        return bgene
    
    # ------------------------------------------------------------------------
    # >> BINARY TO INTEGER CONVERSION
    # ------------------------------------------------------------------------
    def _binary2integer(self, bingene):
        """
        Return the integer value of a binary string

        """
        
        # Convert the gene in integer
        intgene = np.int(bingene, 2)

        return intgene
    
    # ------------------------------------------------------------------------
    # >> INITIALIZE CHROMOSOME POOL
    # ------------------------------------------------------------------------
    def _initialize_pool(self):
        """
        Initialize the chromosome pool.

        Returns
        -------
        None.

        """
        
        # Get the number of genes from bounds
        if isinstance(self.bounds, list):
            self.ngenes = len(self.bounds)
        else:
            self.ngenes = 1

        # Initialize population and misfit
        self.current = []
        self.misfit = []
        
        # Loop over chromosomes in population
        for ichromo in range(self.popsize):
            # Initialise gene_list
            chromosome = []
            # Loop over number of genes
            for igene in range(self.ngenes):
                # Get the corresponding bounds and number of samples
                if isinstance(self.bounds, list):
                    (bmin, bmax, nsamples) = self.bounds[igene]
                else:
                    (bmin, bmax, nsamples) = self.bounds
                # Check if the number of samples is a power of 2
                self._nearest_power2(nsamples)
                # Draw an integer at random in [0, nsamples]
                rgene = np.random.randint(0, high=nsamples)
                # Convert in binary format with the appropriate lenght
                bgene = self._integer2binary(rgene, nsamples)
                chromosome.append(bgene)
                
            # Add chromosome to the current pool
            self.current.append(':'.join(chromosome))
       
    # ------------------------------------------------------------------------
    # >> EVALUATE POOL
    # ------------------------------------------------------------------------
    def _evaluate_pool(self):
        # Loop over chromosomes
        self.fitness = []
        for ipop in range(self.popsize):
            # Split chromosome in genes
            genes = self.current[ipop].split(':')
            # Convert binary chromosomes into real parameter values
            param = []
            for igene in range(len(genes)):
                bmin, bmax, bsamp = self.bounds[igene] 
                value = bmin+int(genes[igene], 2)/(bsamp-1)*(bmax-bmin)
                param.append(value)
            # Evaluate chromosomes using the external function
            if not self.fargs:
                result = self.func(param)
            else:
                result = self.func(param, self.fargs)
            self.fitness.append(result)
            
    # ------------------------------------------------------------------------
    # >> SELECTION STRATEGIES
    # ------------------------------------------------------------------------
    def _tournament_selection(self):
        raise NotImplementedError("This function is not implemented yet.")
    
    def _roulette_selection(self):
        raise NotImplementedError("This function is not implemented yet.")
    
    # ------------------------------------------------------------------------
    # >> CROSSOVER STRATEGIES
    # ------------------------------------------------------------------------
    def _simple_crossover(self):
        raise NotImplementedError("This function is not implemented yet.")
    
    def _multiple_crossover(self):
        raise NotImplementedError("This function is not implemented yet.")
    
    # ------------------------------------------------------------------------
    # >> MUTATION STRATEGIES
    # ------------------------------------------------------------------------
    def _standard_mutation(self):
        raise NotImplementedError("This function is not implemented yet.")
    
    def _linear_mutation(self):
        raise NotImplementedError("This function is not implemented yet.")
    
    def _exponential_mutation(self):
        raise NotImplementedError("This function is not implemented yet.")
    
    # ------------------------------------------------------------------------
    # >> SOLVE
    # ------------------------------------------------------------------------
    def solve(self):
        # Generate pool
        self._initialize_pool()
        # Evaluate chromosomes using the external function
        self._evaluate_pool()
        # Loop over iteration
        for iteration in range(self.maxiter):
            # Selection
            # Crossover
            # Mutation
            self._evaluate_pool()