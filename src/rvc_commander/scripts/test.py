#!/usr/bin/env python3

import numpy as np
from math import inf



class ICPStopCondition:
    '''
    Class that checks if ICP should stop based on multiple criteria:
    - Max iterations
    - Relative improvement
    - No improvement limit
    - Absolute error threshold
    - Transformation magnitude threshold
    '''
    def __init__(
        self,
        max_iterations: int = 10,
        epsilon_rel: float = 1e-3,
        no_improvement_limit: int = 2,
        min_error: float = 1.0,
        epsilon_transform: float = 1e-4,
        min_dtrans: float = 1e-4,
        min_drot: float = 1e-1

    ):
        # Init params
        self.max_iterations = max_iterations
        self.epsilon_rel = epsilon_rel
        self.no_improvement_limit = no_improvement_limit
        self.min_error = min_error
        self.epsilon_transform = epsilon_transform
        self.min_dtrans = min_dtrans
        self.min_drot = min_drot

        # internal state
        self.prev_error = inf
        self.no_improvement_counter = 0
        self.iteration = 0
        self.stop_reason = None


    def get_stop_reason(self):
        '''
        Returns the reason why ICP stopped.'''
        return self.stop_reason
    

    def reset(self):
        '''
        Resets the stop condition state for a new ICP run.
        '''
        self.prev_error = inf
        self.no_improvement_counter = 0
        self.iteration = 0


    def should_stop(self, current_error: float, dtransformation) -> bool:
        """
        Checks if ICP should stop based on the current error and transformation change.

        Parameters:
        ----------
            current_error: The current error metric (e.g., mean distance between correspondences).
            dtransformation: The change in transformation (translation and rotation) from the last iteration.
        
        Returns:
        ----------
            bool: True if ICP should stop, False otherwise.
        """
        self.iteration += 1
        no_improvement = False 

        # Stop cause: Max iterations
        if self.iteration >= self.max_iterations:
            self.stop_reason = "Max iterations reached"
            return True

        # Stop cause: Absolute error threshold
        if current_error < self.min_error:
            self.stop_reason = "Absolute error threshold reached"
            return True        

        # Check transformation magnitude
        if dtransformation is not None:
            
            if not np.all(np.isfinite(dtransformation)):
                self.stop_reason = "Non-finite transformation detected"
                return True
            
            # Compute translation magnitude
            dtrans_norm = np.linalg.norm(dtransformation[:2])
            drot_abs = abs(dtransformation[2])
            
            if dtrans_norm < self.min_dtrans and drot_abs < self.min_drot:
                self.stop_reason = "Transformation magnitude below threshold"
                return True

        if self.iteration > 1:
            # Compute Relative improvement
            if not np.isfinite(self.prev_error):
                rel_improvement = float("inf")
            else:
                rel_improvement = abs(self.prev_error - current_error) / max(self.prev_error, 1e-12)

            # Track improvement
            if rel_improvement < self.epsilon_rel:
                no_improvement = True

            # Divergence check 
            if current_error > self.prev_error:
                no_improvement = True

            # Update improvement counter
            if no_improvement:
                self.no_improvement_counter += 1
            else:
                self.no_improvement_counter = 0

            # Check if improvement 
            if self.no_improvement_counter >= self.no_improvement_limit:
                self.stop_reason = "No improvement limit reached"
                return True
        
        # update state
        self.prev_error = current_error

        return False


def test_icp_stop():
    # Define parameters
    max_iterations = 10
    epsilon_rel = 1e-3
    no_improvement_limit = 2
    min_error = 1.0
    epsilon_transform = 1e-4

    # Init icp stop condition
    icp_stop = ICPStopCondition(
        max_iterations=max_iterations,
        epsilon_rel=epsilon_rel,
        no_improvement_limit=no_improvement_limit,
        min_error=min_error,
        epsilon_transform=epsilon_transform
    )

    # Simulate ICP iterations with decreasing error
    errors = [10.0, 5.0, 2.0, 1.5, 1.2, 1.1, 1.05, 1.02, 1.01, 1.005, 0.999]
    dtransformation = np.array([0.01, 0.01, 0.01])  # Simulated transformation change
    for error in errors:
        if icp_stop.should_stop(error, dtransformation):
            print(f"Stopping at error: {error}")
            break
        else:
            print(f"Continuing at error: {error}")
    
    # Reset stop condition for next test
    icp_stop.reset()
    print("Reset done. Starting new test.")
    

def test_2():
    var = inf
    print(var)


def main():
    test_icp_stop()
    # test_2()


if __name__ == "__main__":
    main()