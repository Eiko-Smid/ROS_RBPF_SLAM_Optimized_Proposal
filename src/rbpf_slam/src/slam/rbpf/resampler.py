#!/usr/bin/env python3

from typing import List, Tuple
import numpy as np
import random


class Resampler:
    '''
    Resampler class for particle filter resampling. Contains general resampling methods that can be used for
    any particle filter implementation.
    '''
    @staticmethod
    def compute_neff(weights: List[float]) -> float:
        '''
        Get's the particles weights and computes the effective number of particles (neff).
        '''
        # Tranfer to numpy arr
        weights = np.array(weights)

        # Sanity check
        if weights.ndim != 1 or weights.size == 0:
            raise ValueError("Weights must be a 1D array.")
        
        # Compute neff
        return 1 / np.sum(weights**2)


    @staticmethod
    def do_resampling(weights: List[float], min_neff: float=10.0) -> bool:
        '''
        Get's the particles weights and the minimum effective number of particles (neff) and checks if resampling is needed.
        '''
        neff = Resampler.compute_neff(weights)
        if neff < min_neff:
            return True
        else:
            return False
        

    @staticmethod
    def low_variance_sampler(weights) -> List[int]:
        '''
        Implementation of stochastic universal sampling (low variance resampling) for particle filters. 
        '''
        number_of_weights= len(weights)
        acc_weight= weights[0]
        indices= []
        particle_index= 0
        
        # Pick particles according to weight.
        random_number= random.uniform(0.0, 1/number_of_weights)
        
        for j in range(number_of_weights):
            u= random_number + j * (1/number_of_weights)

            while(u > acc_weight):
                particle_index+= 1
                acc_weight+= weights[particle_index]
            
            indices.append(particle_index)
        
        # return new_particles.copy()
        return indices
    


def init_weights(n= 10):
    list_weights = []
    for i in range(n):
        list_weights.append(1/n)

    return list_weights


def main():
    min_neff = 10

    # Init weights
    weights = init_weights(n=10)
    
    neff = Resampler.compute_neff(weights=weights)

    print(neff)


if __name__ == "__main__":
    main()