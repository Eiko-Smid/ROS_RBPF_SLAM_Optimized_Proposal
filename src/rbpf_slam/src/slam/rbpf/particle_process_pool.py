import multiprocessing as mp
from multiprocessing.pool import Pool

import logging

from typing import Optional, List, Dict

from .motion_model import MotionModel
from .measurement_model import MeasurementModel
from .proposal import ProposalEstimator


logger = logging.getLogger(__name__)

# Init worker objects
_WORKER_MOTION_MODEL = None
_WORKER_MEAS_MODEL = None
_WORKER_PROPOSAL = None


def _init_worker(
        motion_model: MotionModel,
        measurement_model: MeasurementModel,
        proposal_estimator: ProposalEstimator,
) -> None:
    """
    Initializes the worker process with the motion, measurement models and proposal.
    This function is called when a new worker process is created.
    """
    global _WORKER_MOTION_MODEL
    global _WORKER_MEAS_MODEL
    global _WORKER_PROPOSAL

    _WORKER_MOTION_MODEL = motion_model
    _WORKER_MEAS_MODEL = measurement_model
    _WORKER_PROPOSAL = proposal_estimator



class ParticleProcessPool:
    def __init__(
            self,
            n_workers: int,
            initializer: Optional[callable] = None,
            initargs: Optional[tuple] = None,
    ):
        # Init n workers
        max_n_workers = mp.cpu_count()
        if n_workers is not None and n_workers > 0 and n_workers <= max_n_workers:
            self.n_workers = int(n_workers)
            logger.info(f"Using {self.n_workers} workers for the process pool.")            
        else:
            self.n_workers = max_n_workers
            logger.warning(f"Invalid n_workers specified ({n_workers}). Using maximum available workers: {self.n_workers}.")

        # Define process pool
        self._pool: Optional[Pool] = None
        self._initializer = initializer
        self._initargs = initargs


    def start(self) -> None:
        '''
        Creates the process pool
        '''
        if self._pool is not None:
            return
        
        context = mp.get_context("spawn")
        self._pool = context.Pool(
            processes=self.n_workers,
            initializer=self._initializer,
            initargs=self._initargs,
        )


    def map(self, worker_func, tasks, chunksize=1):
        '''
        Maps the worker function to the tasks using the process pool.
        '''
        if self._pool is None:
            raise RuntimeError("Pool has not been started yet.")

        return self._pool.map(
            func=worker_func,
            iterable=tasks,
            chunksize=chunksize,
        )

    
    def close(self)-> None: 
        '''
        Closes the process pool and waits for the worker processes to finish.
        '''
        if self._pool is None:
            return
        
        self._pool.close()
        self._pool.join()
        self._pool = None


    def terminate(self) -> None:
        '''
        Terminates the process pool. When using terminate, the worker processes are killed immediately without
        completing their current tasks nor their shutdown behavior.
        '''
        if self._pool is None:
            return
        
        self._pool.terminate()
        self._pool.join()
        self._pool = None

    
    def __enter__(self):
        '''
        Starts the process pool when entering the context.
        '''
        self.start()
        return self
    

    def __exit__(self, exc_type, exc_value, traceback):
        '''
        Closes the process pool when exiting the context.
        '''
        self.close()



def main():
    ppp = ParticleProcessPool()

    print(f"N workers: {ppp.n_workers}")


if __name__ == "__main__":
    main()