from concurrent.futures import ThreadPoolExecutor
from typing import Optional


class ParticleThreadPool:
    def __init__(self, n_workers: int):
        if n_workers < 1:
            raise ValueError("n_workers must be at least 1")

        self.n_workers = int(n_workers)
        self._executor: Optional[ThreadPoolExecutor] = None

    def start(self) -> None:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self.n_workers
            )

    def map(self, worker_func, tasks):
        if self._executor is None:
            raise RuntimeError("Thread pool has not been started.")

        return list(
            self._executor.map(
                worker_func,
                tasks,
            )
        )

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()