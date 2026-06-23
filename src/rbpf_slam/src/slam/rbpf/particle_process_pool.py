import multiprocessing as mp
from multiprocessing.pool import Pool

from typing import Optional, List, Dict


class ParticleProcessPool:
    def __init__(self, n_workers: Optional[int] = None):
        # Init n workers
        if n_workers is not None and n_workers > 0:
            self.n_workers = int(n_workers)
        else:
            self.n_workers = mp.cpu_count()

        # Define process pool
        self._pool: Optional[Pool] = None


    def start(self) -> None:
        '''
        Creates the process pool
        '''
        if self._pool is not None:
            return
        
        context = mp.get_context("spawn")
        self._pool = context.Pool(processes=self.n_workers)


    def map(self, worker_func, tasks):
        '''
        Maps the worker function to the tasks using the process pool.
        '''
        if self._pool is None:
            raise RuntimeError("Pool has not been started yet.")

        return self._pool.map(
            func=worker_func,
            iterable=tasks
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