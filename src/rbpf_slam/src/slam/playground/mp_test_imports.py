#!/usr/bin/env python3

import multiprocessing as mp
import os
import sys


def worker() -> None:
    print("Worker PID:", os.getpid())
    print("Worker sys.path:")
    print("\n".join(sys.path))

    import rvc_commander
    print("rvc_commander:", rvc_commander.__file__)

    from rvc_commander.msg import Measurement
    print("Measurement:", Measurement)


def main() -> None:
    context = mp.get_context("spawn")
    process = context.Process(target=worker)
    process.start()
    process.join()

    print("Exit code:", process.exitcode)


if __name__ == "__main__":
    main()