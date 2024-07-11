import os
import time
import csv
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from quBLP.problemtemplate import FacilityLocationProblem as FLP
from quBLP.problemtemplate import GraphColoringProblem as GCP
from quBLP.problemtemplate import KPartitionProblem as KPP
from quBLP.models import CircuitOption, OptimizerOption

optimizer_option = OptimizerOption(
    params_optimization_method='COBYLA',
    max_iter=150
)

problems = [
    FLP(1, 2, [[10, 2]], [2, 2]), 
    FLP(2, 2, [[10, 2],[1, 20]],[1, 1]),
    FLP(2, 3, [[10, 2, 3],[1, 20, 3]],[1, 1, 1]),
    FLP(3, 3, [[10, 2, 3],[10, 2, 3],[10, 2, 3]],[1, 1, 1]),
    FLP(3, 4, [[10, 2, 3, 1],[10, 2, 3, 1],[10, 2, 3, 1],[10, 2, 3, 1]],[1, 1, 1, 1]),
    GCP(3, [[0, 1]]), 
    GCP(3, [[0, 1], [1, 2]]),
    GCP(4, [[0, 1]]),
    GCP(4, [[0, 1], [1, 2]]),
    GCP(4, [[0, 1], [1, 2], [2, 3]]),
    KPP(3, [2, 1], [[[0, 1], 1], [[1, 2], 1]]),
    KPP(4, [3, 1], [[[0, 1], 1], [[1, 2], 1], [[2, 3], 1]]),
    KPP(5, [4, 1], [[[0, 1], 1], [[1, 2], 1], [[2, 3], 1], [[3, 4], 1]]),
    KPP(5, [3, 2], [[[0, 1], 1], [[1, 2], 1], [[2, 3], 1], [[3, 4], 1]]),
    KPP(5, [3, 1, 1], [[[0, 1], 1], [[1, 2], 1], [[2, 3], 1], [[3, 4], 1]]),
]

mcx_modes = ['constant', 'linear']
if_use_decompose = [True, False]
feedback = ['transpile_time', 'depth', 'culled_depth']
layers = range(1, 2)

headers = ["pid", 'num_layers', "use_decompose"] + feedback

def process_layer(prb, use_decompose, num_layers, feedback):
    prb.set_algorithm_optimization_method('commute', 400)
    circuit_option = CircuitOption(
        num_layers=num_layers,
        need_draw=False,
        use_decompose=use_decompose,
        circuit_type='qiskit',
        mcx_mode='constant',
        backend='AerSimulator',
        feedback=feedback,
    )
    result = prb.optimize(optimizer_option, circuit_option)
    return result

if __name__ == '__main__':
    set_timeout = 60 * 60 * 24 # Set timeout duration
    num_complete = 0
    script_path = os.path.abspath(__file__)
    new_path = script_path.replace('experiment', 'data')[:-3]
    print(new_path)
    with open(f'{new_path}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # Write headers once

        with ProcessPoolExecutor() as executor:
            futures = []
            for use_decompose in if_use_decompose:
                for pid, prb in enumerate(problems):
                    for num_layers in layers:
                        future = executor.submit(process_layer, prb, use_decompose, num_layers, feedback)
                        futures.append((future, pid, use_decompose, num_layers))

            start_time = time.perf_counter()
            for future, pid, use_decompose, num_layers in futures:
                current_time = time.perf_counter()
                remaining_time = max(set_timeout - (current_time - start_time), 0)
                diff = []
                try:
                    result = future.result(timeout=remaining_time)
                    for dict_term in feedback:
                        diff.append(result[dict_term])
                    print(f"Task for problem {pid}, use_decompose {use_decompose}, num_layers {num_layers} executed successfully.")
                except MemoryError:
                    for dict_term in feedback:
                        diff.append('memory_error')
                    print(f"Task for problem {pid}, use_decompose {use_decompose}, num_layers {num_layers} encountered a MemoryError.")
                except TimeoutError:
                    for dict_term in feedback:
                        diff.append('timeout')
                    print(f"Task for problem {pid}, use_decompose {use_decompose}, num_layers {num_layers} timed out.")
                finally:
                    row = [pid, num_layers, use_decompose] + diff
                    writer.writerow(row)  # Write row immediately
                    num_complete += 1
                    if num_complete == 2 * len(problems):
                        print(f'Data has been written to {new_path}.csv')