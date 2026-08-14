from dataclasses import dataclass, field
from multiprocessing import get_context
from typing import Any, Dict, List
import os
import time


# ---------------------------------------------------------------------
# Simple example classes
# ---------------------------------------------------------------------

@dataclass
class Particle:
    particle_id: int
    value: float
    weight: float


class MotionModel:
    def __init__(self, movement: float):
        self.movement = movement

    def predict(self, value: float) -> float:
        return value + self.movement


class MeasurementModel:
    def __init__(self, target: float):
        self.target = target

    def likelihood(self, value: float) -> Dict[str, float]:
        error = abs(self.target - value)

        # Simplified likelihood
        likelihood = 1.0 / (1.0 + error)

        return {
            "likelihood": likelihood,
            "measurement_calls": 1,
            "absolute_error": error,
        }


class ProposalEstimator:
    def estimate(self, predicted_value: float) -> float:
        # Simplified proposal computation
        return predicted_value + 0.5


# ---------------------------------------------------------------------
# Task and result objects
# ---------------------------------------------------------------------

@dataclass
class ParticleUpdateTask:
    particle_index: int
    particle: Particle
    motion_model: MotionModel
    measurement_model: MeasurementModel


@dataclass
class ParticleUpdateResult:
    particle_index: int
    particle: Particle
    likelihood: float
    worker_pid: int

    timings: Dict[str, float] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------
# Worker function
# ---------------------------------------------------------------------

def update_particle_worker(
    task: ParticleUpdateTask,
) -> ParticleUpdateResult:
    """
    Executes one particle update.

    This function runs inside a worker process when pool.map() is used.
    It does not modify the parent object.
    """

    # One fresh proposal object for this task
    proposal = ProposalEstimator()

    # Local particle reference.
    # Inside a worker process this is a deserialized copy.
    particle = task.particle

    # Delay worker
    time.sleep(0.4)

    timings = {}

    # Motion model
    start = time.perf_counter()

    predicted_value = task.motion_model.predict(
        particle.value
    )

    timings["motion_model_s"] = time.perf_counter() - start

    # Proposal
    start = time.perf_counter()

    new_value = proposal.estimate(
        predicted_value
    )

    timings["proposal_s"] = time.perf_counter() - start

    # Measurement model
    start = time.perf_counter()

    measurement_result = task.measurement_model.likelihood(
        new_value
    )

    timings["measurement_model_s"] = (
        time.perf_counter() - start
    )

    likelihood = measurement_result["likelihood"]

    # Create updated particle
    new_particle = Particle(
        particle_id=particle.particle_id,
        value=new_value,
        weight=particle.weight,
    )

    diagnostics = {
        "measurement_calls": measurement_result[
            "measurement_calls"
        ],
        "absolute_error": measurement_result[
            "absolute_error"
        ],
    }

    return ParticleUpdateResult(
        particle_index=task.particle_index,
        particle=new_particle,
        likelihood=likelihood,
        worker_pid=os.getpid(),
        timings=timings,
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------
# Task creation
# ---------------------------------------------------------------------

def create_tasks(
    particles: List[Particle],
    motion_model: MotionModel,
    measurement_model: MeasurementModel,
) -> List[ParticleUpdateTask]:

    tasks = []

    for particle_index, particle in enumerate(particles):
        task = ParticleUpdateTask(
            particle_index=particle_index,
            particle=particle,
            motion_model=motion_model,
            measurement_model=measurement_model,
        )

        tasks.append(task)

    return tasks


# ---------------------------------------------------------------------
# Parent-side result processing
# ---------------------------------------------------------------------

def process_results(
    results: List[ParticleUpdateResult],
) -> List[Particle]:

    # pool.map() preserves task order.
    # Still, particle_index is useful for validation and debugging.
    for expected_index, result in enumerate(results):
        if result.particle_index != expected_index:
            raise RuntimeError(
                "Unexpected particle result order"
            )

    updated_particles = [
        result.particle
        for result in results
    ]

    # Aggregate diagnostics returned by every task
    total_measurement_calls = sum(
        result.diagnostics.get("measurement_calls", 0)
        for result in results
    )

    mean_absolute_error = sum(
        result.diagnostics.get("absolute_error", 0.0)
        for result in results
    ) / len(results)

    mean_proposal_time = sum(
        result.timings.get("proposal_s", 0.0)
        for result in results
    ) / len(results)

    print("\nAggregated in parent process:")
    print(
        f"Total measurement calls: "
        f"{total_measurement_calls}"
    )
    print(
        f"Mean absolute error: "
        f"{mean_absolute_error:.3f}"
    )
    print(
        f"Mean proposal time: "
        f"{mean_proposal_time:.9f} s"
    )

    return updated_particles


# ---------------------------------------------------------------------
# Sequential version
# ---------------------------------------------------------------------

def run_sequential(
    tasks: List[ParticleUpdateTask],
) -> List[ParticleUpdateResult]:

    results = []

    for task in tasks:
        result = update_particle_worker(task)
        results.append(result)

    return results


# ---------------------------------------------------------------------
# Multiprocessing version
# ---------------------------------------------------------------------

def run_parallel(
    tasks: List[ParticleUpdateTask],
    n_workers: int,
) -> List[ParticleUpdateResult]:

    # Explicit multiprocessing context
    context = get_context("spawn")

    with context.Pool(processes=n_workers) as pool:
        results = pool.map(
            update_particle_worker,
            tasks,
        )

    # The pool is closed here because of the "with" block.
    return results


# ---------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------

def main():
    # Initialize particles
    particles = [
        Particle(
            particle_id=i,
            value=float(i),
            weight=0.25,
        )
        for i in range(4)
    ]

    # Instantiate models
    motion_model = MotionModel(
        movement=1.0
    )

    measurement_model = MeasurementModel(
        target=4.0
    )
    
    # Define the tasks for each particle update
    tasks = create_tasks(
        particles=particles,
        motion_model=motion_model,
        measurement_model=measurement_model,
    )

    print("Running multiprocessing version")

    results = run_parallel(
        tasks=tasks,
        n_workers=4,
    )

    # Print results
    for result in results:
        print(
            f"Particle index: {result.particle_index}, "
            f"new value: {result.particle.value}, "
            f"likelihood: {result.likelihood:.3f}, "
            f"worker PID: {result.worker_pid}"
        )

    # Process results
    updated_particles = process_results(results)

    print("\nUpdated particles in parent:")

    for particle in updated_particles:
        print(particle)


if __name__ == "__main__":
    main()
