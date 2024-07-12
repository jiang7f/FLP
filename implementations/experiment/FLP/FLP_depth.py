import os
import time
import csv
import signal
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
    # GCP(3, [[0, 1]]), 
    # GCP(3, [[0, 1], [1, 2]]),
    # GCP(4, [[0, 1]]),
    # GCP(4, [[0, 1], [1, 2]]),
    # GCP(4, [[0, 1], [1, 2], [2, 3]]),
    # KPP(3, [2, 1], [[[0, 1], 1], [[1, 2], 1]]),
    # KPP(4, [3, 1], [[[0, 1], 1], [[1, 2], 1], [[2, 3], 1]]),
    # KPP(5, [4, 1], [[[0, 1], 1], [[1, 2], 1], [[2, 3], 1], [[3, 4], 1]]),
    # KPP(5, [3, 2], [[[0, 1], 1], [[1, 2], 1], [[2, 3], 1], [[3, 4], 1]]),
    # KPP(5, [3, 1, 1], [[[0, 1], 1], [[1, 2], 1], [[2, 3], 1], [[3, 4], 1]]),
]
methods = ['penalty', 'cyclic', 'commute', 'HEA']
feedback = ['depth', 'culled_depth']
mcx_modes = ['constant', 'linear']
layers = range(1, 2)

headers = ["demands", 'facilities', 'layers', "variables", 'constraints', 'mcx_mode', 'method'] + feedback

def process_layer(prb, num_layers, mcx_mode, feedback, method):
    prb.set_algorithm_optimization_method(method, 400)
    circuit_option = CircuitOption(
        num_layers=num_layers,
        need_draw=False,
        use_decompose=True,
        circuit_type='qiskit',
        mcx_mode=mcx_mode,
        backend='AerSimulator',
        feedback=feedback,
    )
    result = prb.optimize(optimizer_option, circuit_option)
    return result

if __name__ == '__main__':
    set_timeout = 60 * 10 # Set timeout duration
    num_complete = 0
    script_path = os.path.abspath(__file__)
    new_path = script_path.replace('experiment', 'data')[:-3]
    print(new_path)
    with open(f'{new_path}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # Write headers once

        with ProcessPoolExecutor() as executor:
            futures = []
            for mcx_mode in mcx_modes:
                for pid, prb in enumerate(problems):
                    for idx, method in enumerate(methods):
                        for layer in layers:
                            future = executor.submit(process_layer, prb, layer, mcx_mode, feedback, method)
                        futures.append((future, prb, pid, layer, mcx_mode, feedback, method))

            start_time = time.perf_counter()
            for future, prb, pid, layer, mcx_mode, feedback, method in futures:
                current_time = time.perf_counter()
                remaining_time = max(set_timeout - (current_time - start_time), 0)
                diff = []
                try:
                    result = future.result(timeout=remaining_time)
                    for dict_term in feedback:
                        diff.append(result[dict_term])
                    print(f"Task for problem {pid} {method} executed successfully.")
                except MemoryError:
                    print(f"Task for problem {pid} {method} encountered a MemoryError.")
                    for dict_term in feedback:
                        diff.append('memory_error')
                except TimeoutError:
                    print(f"Task for problem {pid} {method} timed out.")
                    for dict_term in feedback:
                        diff.append('timeout')
                finally:
                    row = [prb.num_demands, prb.num_facilities, layer, prb.num_variables, len(prb.linear_constraints), mcx_mode, method] + diff
                    writer.writerow(row)  # Write row immediately
                    num_complete += 1
                    if num_complete == len(futures):
                        print(f'Data has been written to {new_path}.csv')
                        for process in executor._processes.values():
                            os.kill(process.pid, signal.SIGTERM)