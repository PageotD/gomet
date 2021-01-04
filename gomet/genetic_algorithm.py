import math
import numpy as np
from common import Candidate
from scipy.optimize import OptimizeResult, Bounds
from scipy.optimize.optimize import _status_message
from operator import attrgetter

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
    _crossover_modes = ['single', 'multiple']
    _mutation_modes  = ['standard', 'linear', 'exponential']

    # Selection strategies
    _selStrategies = {
        'tournament': '_selTournament',
        'roulette': '_selRoulette'
        }
    
    # Crossover strategies
    _xovStrategies = {
        'single': '_xovSingle',
        'multiple': '_xovMultiple'
        }
    
    # Mutation strategies
    _mutStrategies = {
        'uniform': '_mutUniform',
        'linear': '_mutLinear',
        'exponential': '_mutExponent'
        }
    
    # ------------------------------------------------------------------------
    # >> INITIALIZE
    # ------------------------------------------------------------------------
    def __init__(self, func, bounds, fargs=(), popsize=100, maxiter=100, 
                 selection='tournament', crossover='single', pc=0.2, 
                 mutation='uniform', pm=0.05, coding='binary'):
       
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
        if selection in self._selStrategies:
            self._selection = getattr(self, self._selStrategies[selection])
        else:
            raise ValueError("Please select a valid selection strategy")
        
        # Check crossover mode
        if crossover in self._xovStrategies:
            self._crossover = getattr(self, self._xovStrategies[crossover])
        else:
            raise ValueError("Please select a valid crossover strategy")
            
        # Check crossover mode
        if mutation in self._mutStrategies:
            self._mutation = getattr(self, self._mutStrategies[mutation])
        else:
            raise ValueError("Please select a valid mutation strategy")
    
        # Check coding
        if coding == 'binary':
            self.encoding = self.codInt2bin
            self.decoding = self.codBin2int
        elif coding == 'gray':
            self.encoding = self.codInt2gray
            self.decoding = self.codGray2int
            
    # ------------------------------------------------------------------------
    # >> CODING
    # ------------------------------------------------------------------------
    def nearPower2(self, n):
        """
        Get the nearest power of 2 of an integer value.

        Parameters
        ----------
        n : int
            An integer.

        Returns
        -------
        int
            nearest power of 2 of n.

        """
        # Get the log of n in base 2 and get the closest integer to n
        npower = round(math.log(n, 2))
        
        # Return the next power of 2
        return int(math.pow(2, npower))
    
    def codInt2bin(self, intval, nsample=32):
        """
        Convert an `int` in its binary string representation.

        Parameters
        ----------
        intval : int
            Integer to convert
        nsample : int, optional
            Number of samples in the interval min/max bounds. 
            The default is 32.

        Returns
        -------
        binstr : int
            Binary string representation of the input integer.

        """
        
        # Calculate the number of bits needed for the binary representation
        # given the number of samples between the min. and max. bounds.
        # The number of samples must be a power of 2. If not, the nearest power 
        # of 2 is used.
        nsample_check = self.nearPower2(nsample)
        
        # Calculate the maximum lenght of the bit-string
        # Old version : nelements = 1./np.log(2.)*np.log(float(nsamples))
        nbits = len(np.binary_repr(nsample_check-1))
    
        # Convert the gene in binary format
        binstr = np.binary_repr(intval, width=nbits)

        return binstr
    
    def codBin2int(self, binstr):
        """
        Convert binary string representation in its integer value.

        Parameters
        ----------
        binstr : str
            Binary string representation of an integer.

        Returns
        -------
        intval : int
            Corresponding integer value.

        """
        # Convert the gene in integer
        intval = np.int(binstr, 2)
        
        return intval
    
    def codBin2gray(self, binstr):
        """
        Convert a binary string representation in gray code.

        Parameters
        ----------
        binstr : int
            Binary string representation.

        Returns
        -------
        grayCode : int
            Gray code representation of the input binary string.

        """
        grayCode = binstr[0]
        
        for ibit in range(1, len(binstr)):
            if binstr[ibit-1] == binstr[ibit]:
                grayCode = grayCode+'0'
            else:
                grayCode = grayCode+'1'

        return grayCode
    
    def codGray2bin(self, grayCode):
        """
        Convert a gray code in binary string.

        Parameters
        ----------
        grayCode : int
            Gray code representation.

        Returns
        -------
        binstr : int
            Binary string representation of the input gray code.

        """
        binstr = grayCode[0]
        
        for ibit in range(1, len(grayCode)):
            if grayCode[ibit] == '0':
                binstr = binstr+binstr[ibit-1]
            else:
                if binstr[ibit-1] == '0':
                    binstr = binstr+'1'
                else:
                    binstr = binstr+'0'

        return binstr
    
    def codInt2gray(self, intval, nsample=32):
        """
        Convert an `int` in its gray code representation.

        Parameters
        ----------
        intval : int
            Integer to convert
        nsample : int, optional
            Number of samples in the interval min/max bounds. 
            The default is 32.

        Returns
        -------
        binstr : int
            Gray code representation of the input integer.

        """
        
        binstr = self.codInt2bin(intval, nsample=nsample)
        return self.codBin2gray(binstr)
    
    def codGray2int(self, grayCode):
        """
        Convert binary string representation in its integer value.

        Parameters
        ----------
        grayCode : str
            Gray code binary string representation of an integer.

        Returns
        -------
        intval : int
            Corresponding integer value.

        """
        binstr = self.codGray2bin(grayCode)
        return self.codBin2int(binstr)
    
    # ------------------------------------------------------------------------
    # >> POPULATION
    # ------------------------------------------------------------------------
    def chrInitialize(self):
        """
        Create a chromosome (candidate solution).

        Returns
        -------
        Candidate solution object.

        """
        
        # Initialise candidate solution
        solCandidate = Candidate(chromosome=[], fitness=None)
        
        # Get the number of genes (parameters) from bounds
        if isinstance(self.bounds, list):
            ngenes = len(self.bounds)
        else:
            ngenes = 1
            
        # Loop over number of genes
        for igene in range(ngenes):
            # Get the corresponding bounds and number of samples
            if isinstance(self.bounds, list):
                (bmin, bmax, nsample) = self.bounds[igene]
            else:
                (bmin, bmax, nsample) = self.bounds
            # Draw an integer at random in [0, nsamples]
            rgene = np.random.randint(0, high=nsample)
            # Convert in binary format with the appropriate lenght
            bgene = self.encoding(rgene, nsample=nsample)
            solCandidate.chromosome.append(bgene)
            
        return solCandidate
    
    def chrDecode(self, solCandidate):
        """
        

        Parameters
        ----------
        solCandidate : Candidate object
            Input candidate solution containing the chromosome (attribute) to
            decode.

        Returns
        -------
        List of parameter values (one parameter value per gene).

        """
        
        # Initialize parameter list
        paramList = []

        # Get the number of genes (parameters) from bounds
        if isinstance(self.bounds, list):
            ngenes = len(self.bounds)
        else:
            ngenes = 1
            
        # Loop over number of genes
        for igene in range(ngenes):
            # Get the corresponding bounds and number of samples
            if isinstance(self.bounds, list):
                (bmin, bmax, nsample) = self.bounds[igene]
            else:
                (bmin, bmax, nsample) = self.bounds
            # Decode gene
            bgene = solCandidate.chromosome[igene]
            # Convert in integer
            rgene = self.decoding(bgene)
            # Convert in parameter value
            paramList.append(bmin+float(rgene/(nsample-1))*(bmax-bmin))

        return paramList
        
    def popInitialize(self):
        """
        Initialize the population of chromosomes by selecting candidate 
        solutions at random.

        Returns
        -------
        None.

        """
        
        # Initialize population list
        self.population = []
    
        # Loop over chromosomes in population
        for ichromo in range(self.popsize):
            # Add chromosome to the current pool
            self.population.append(self.chrInitialize())
            
    def popEvaluate(self):
        """
        Evaluate the current population.

        Returns
        -------
        None.

        """
        # Loop over population
        for ipop in range(len(self.population)):
            # Convert chromosome in real parameter value list
            paramValues = self.chrDecode(self.population[ipop])
            # Evaluate chromosomes using the external function
            if not self.fargs:
                self.population[ipop].fitness = self.func(paramValues)
            else:
                self.population[ipop].fitness = self.func(paramValues, self.fargs)
            
    # ------------------------------------------------------------------------
    # >> SELECTION STRATEGIES
    # ------------------------------------------------------------------------
    def selProportionate(self):
        # Get the fitness of each chromosome
        fitness = [obj.fitness for obj in self.population]
        
        # Sum the fitness values
        fitnessSum = sum(fitness)
        
        # Calculate probabilities for each chromosome
        prob = []
        prob.append(fitness[0]/fitnessSum)
        for ifit in range(1,len(fitness)):
            prob.append(prob[ifit-1]+fitness[ifit]/fitnessSum)
        
        # Initialize newpopulation
        newpopulation = []
        
        # Trial
        for ipop in range(len(self.population)):
            trial = np.random.random_sample()
            idx = 0
            while trial > prob[idx]:
                idx += 1
            newpopulation.append(Candidate(chromosome=self.population[idx].chromosome, fitness=None))

        chromosome = [obj.chromosome for obj in self.population]
        print(chromosome)
        return newpopulation
    
    # ------------------------------------------------------------------------
    # >> CROSSOVER STRATEGIES
    # ------------------------------------------------------------------------
    
    # ------------------------------------------------------------------------
    # >> MUTATION STRATEGIES
    # ------------------------------------------------------------------------
    
    # ------------------------------------------------------------------------
    # >> SELECTION STRATEGIES
    # ------------------------------------------------------------------------
    
    def _selTournament(self, k=5):
        """
        Tournament between k chromosomes chosen at random from the current 
        pool. The winner, the one with the best fitness value, is selected for
        crossover in order to produce the next generation of chromosome.
        """
        
        # Generate the new parent population
        parentpop = []
        for indv in range(self.popsize):
            # Choose k challengers in the pool
            challengers = np.random.choice(self.pool, size=k)
            # Select winner
            winner = max(challengers, key=attrgetter('fitness'))
            # Add the winner to the new population
            parentpop.append(winner)
        
        return parentpop
    
    def _selRoulette(self):
        raise NotImplementedError("This function is not implemented yet.")
    
    # ------------------------------------------------------------------------
    # >> CROSSOVER STRATEGIES
    # ------------------------------------------------------------------------
    def _xovSingle(self, parent1, parent2):
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
    
    def _xovMultiple(self):
        raise NotImplementedError("This function is not implemented yet.")
    
    # ------------------------------------------------------------------------
    # >> MUTATION STRATEGIES
    # ------------------------------------------------------------------------
    def _mutUniform(self, offspring):
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
    
    def _mutLinear(self):
        raise NotImplementedError("This function is not implemented yet.")
    
    def _mutExponent(self):
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
            new_generation = self._selection()
            for i in range(self.pop.size/2):
                # Selection
                parent1 = self.tournament()
                parent2 = self.tournament()
                # Crossover
                offspring1, offspring2 = self._crossover(parent1, parent2)
                # Mutation
                offspring1 = self._mutation(offspring1)
                offspring2 = self._mutation(offspring2)
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