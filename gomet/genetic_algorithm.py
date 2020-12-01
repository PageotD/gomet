import math
import numpy as np
from scipy.optimize import OptimizeResult, Bounds
from scipy.optimize.optimize import _status_message

from scipy._lib._util import check_random_state

def genetic_algorithm(func):
    """
    Find the global minimum of a function using Genetic Algorithm.
    
    Parameters
    ----------
    func : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    Notes
    -----
    Genetic Algorithm is a population-based, stochastic optimization method 
    that mimics some evolution processes and natural selection. At each 
    iteration, pairs of chromosomes are selected (based on their fitness) to 
    pass their genes to the next generation. Each pairs of selected 
    chromosomes will produce offspring for the next generation through 
    crossover and mutation phases.
    
    This implementation is originally based on [1]_ and is improved following 
    [2]_ and [3]_.
    
    Examples
    --------
    Let us consider the problem of minimizing the Rosenbrock function. This
    function is implemented in `rosen` in `scipy.optimize`.
    
    >>> from scipy.optimize import rosen, differential_evolution
    
    References
    ----------
    .. [1] Gallagher, K. & Sambridge, M. Genetic algorithms: a powerful tool 
           for large-scale nonlinear optimization problems Computers & 
           Geosciences, Elsevier, 1994, 20, 1229-1236.
           
    .. [2] Sambridge, M. & Gallagher, K. Earthquake hypocenter location using 
           genetic algorithms Bulletin of the Seismological Society of America,
           Seismological Society of America, 1993, 83, 1467-1491
    
    .. [3] Sen, M. K. & Stoffa, P. L. Rapid sampling of model space using 
           genetic algorithms: examples from seismic waveform inversion 
           Geophysical Journal International, Oxford University Press, 1992, 
           108, 281-292
    """
    pass

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
                 selection='tournament', crossover='simple', pc=0.2, 
                 mutation='standard', pm=0.05):
       
        # Initiate class
        self.func = func
        self.fargs = fargs
        self.maxiter = maxiter
        self.bounds = bounds
        self.pc = pc
        self.pm = pm
        
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
            self._selection = getattr(self, selection.lower())
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
    def tournament(self):
        """
        Tournament between two chromosomes chosen at random from the current 
        pool. The winner, the one with the best fitness value, is selected for
        crossover in order to produce the next generation of chromosome.
        """
        # Chose two challengers in the pool
        challenger1 = np.random.randint(0, self.popsize)
        challenger2 = challenger1
        while challenger1 == challenger2:
            challenger2 = np.random.randint(0, self.popsize)
        # Tournament
        if self.fitness[challenger1] < self.fitness[challenger2]:
            selected = self.current[challenger1]
        else:
            selected = self.current[challenger2]
        return selected
    
    def _roulette_selection(self):
        raise NotImplementedError("This function is not implemented yet.")
    
    # ------------------------------------------------------------------------
    # >> CROSSOVER STRATEGIES
    # ------------------------------------------------------------------------
    def _simple_crossover(self, parent1, parent2):
        """
        Combine the genetic information of two chromosomes (parents) to 
        generate the offsprings (childrens).
        """
        if np.random.random_sample() < self.pc:
            # Determine the crossover point
            icross = np.random.randint(0, len(parent1))
            # Crossover
            offspring1 =  parent1[:icross]+parent2[icross:]
            offspring2 =  parent2[:icross]+parent1[icross:]
        else:
            # No crossover
            offspring1 = parent1
            offspring2 = parent2
        
        return offspring1, offspring2
    
    def _multiple_crossover(self):
        raise NotImplementedError("This function is not implemented yet.")
    
    # ------------------------------------------------------------------------
    # >> MUTATION STRATEGIES
    # ------------------------------------------------------------------------
    def _standard_mutation(self, offspring):
        """
        Flip an arbitrary bit in the chromosome genes. 

        """
        # Split chromosome in genes
        genes = offspring.split(':')
        new_genes = []
        # Loop over genes
        for igene in range(len(genes)):
            # Split gene in bits
            bits = list(genes[igene])
            # Loop over bits
            for ibit in range(len(bits)): 
                # Test for mutation
                if np.random.random() < self.pm:
                    if bits[ibit] == '0':
                        bits[ibit] = '1'
                    else:
                        bits[ibit] = '0'
            # Reassemble gene
            new_genes.append(''.join(bits))
        # Reassemble chromosome
        offspring = ':'.join(new_genes)
        
        return offspring
    
    def _linear_mutation(self):
        raise NotImplementedError("This function is not implemented yet.")
    
    def _exponential_mutation(self):
        raise NotImplementedError("This function is not implemented yet.")
    
    # ------------------------------------------------------------------------
    # >> SOLVE
    # ------------------------------------------------------------------------
    def solve(self):
        # Initiate OptimizeResult
        result = OptimizeResult()
        # Generate pool
        self._initialize_pool()
        # Evaluate chromosomes using the external function
        self._evaluate_pool()
        # Loop over iteration
        for iteration in range(self.maxiter):
            # New generation
            new_generation = []
            for i in range(self.pop.size/2):
                # Selection
                parent1 = self.tournament()
                parent2 = self.tournament()
                # Crossover
                offspring1, offspring2 = self._simple_crossover(parent1, parent2)
                # Mutation
                offspring1 = self._standard_mutation(offspring1)
                offspring2 = self._standard_mutation(offspring2)
                # New generation
                new_generation.append(offspring1)
                new_generation.append(offspring2)
            # New generation becomes the current generation
            self.current = new_generation[:]
            # Evaluate chromosome using the external function
            self._evaluate_pool()
        
        # Get the best solution
        ibest = np.argmin(self.misfit)
        result.x = self.current[ibest]
        result.success = True
        
        return result