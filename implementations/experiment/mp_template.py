import os
import time
import csv
import signal
import random
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from quBLP.problemtemplate import FacilityLocationProblem as FLP
from quBLP.problemtemplate import GraphColoringProblem as GCP
from quBLP.problemtemplate import KPartitionProblem as KPP
from quBLP.models import CircuitOption, OptimizerOption
from quBLP.analysis import generater

random.seed(7)

script_path = os.path.abspath(__file__)
new_path = script_path.replace('experiment', 'data')[:-3]

optimizer_option = OptimizerOption(
    params_optimization_method='COBYLA',
    max_iter=150
)
flp_problems, flp_configs = generater.generate_flp(10, [(1, 2), (2, 2), (2, 3), (3, 4)], 1, 20)
gcp_problems, gcp_configs = generater.generate_gcp(10, [(3, 2), (4, 1), (4, 2), (4, 3)])
kpp_problems, kpp_configs = generater.generate_kpp(10, [(4, [2, 2], 3), (6, [2, 2, 2], 5), (8, [2, 2, 4], 7), (9, [3, 3, 3], 8)], 1, 20)

problems_pkg = flp_problems + gcp_problems + kpp_problems # 问题是按规模打包的
problems = [prb for problems in problems_pkg for prb in problems]

# configs = flp_configs + gcp_configs + kpp_configs

# with open(f"{new_path}_configs.txt", "w") as file:
#     for item in configs:
#         file.write(str(item) + '\n')
# exit()

layers = range(1, 5)
methods = ['penalty', 'penalty', 'cyclic', 'HEA']
evaluation_metrics = ['ARG', 'in_constraints_probs', 'best_solution_probs']
headers = ['pid', 'layers', "variables", 'constraints', 'method'] + evaluation_metrics

def process_layer(prb, num_layers, method):
    prb.set_algorithm_optimization_method(method, 400)
    circuit_option = CircuitOption(
        num_layers=num_layers,
        need_draw=False,
        use_decompose=True,
        circuit_type='qiskit',
        mcx_mode='constant',
        backend='ddsim' if method == 'commute' else 'AerSimulator', # 'FakeQuebec' # 'AerSimulator'
    )
    ARG, in_constraints_probs, best_solution_probs = prb.optimize(optimizer_option, circuit_option)
    return [ARG, in_constraints_probs, best_solution_probs]

if __name__ == '__main__':
    all_start_time = time.perf_counter()
    set_timeout = 60 * 60 * 24 * 7 # Set timeout duration
    num_complete = 0
    print(new_path)
    with open(f'{new_path}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # Write headers once

        # cpu_num_processes = os.cpu_count() // 2
        num_processes = 1
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = []
            for layer in layers: 
                for method in methods:
                    for pid, prb in enumerate(problems):
                        print(f'{pid}, {layer}, {method} build')
                        future = executor.submit(process_layer, prb, layer, method)
                        futures.append((future, prb, pid, layer, method))

            start_time = time.perf_counter()
            for future, prb, pid, layer, method in futures:
                current_time = time.perf_counter()
                remaining_time = max(set_timeout - (current_time - start_time), 0)
                diff = []
                try:
                    metrics = future.result(timeout=remaining_time)
                    diff.extend(metrics)
                    print(f"Task for problem {pid} L={layer} {method} executed successfully.")
                except MemoryError:
                    print(f"Task for problem {pid} L={layer} {method} encountered a MemoryError.")
                    for dict_term in evaluation_metrics:
                        diff.append('memory_error')
                except TimeoutError:
                    print(f"Task for problem {pid} L={layer} {method} timed out.")
                    for dict_term in evaluation_metrics:
                        diff.append('timeout')
                except Exception as e:
                    print(f"An error occurred: {e}")
                finally:
                    row = [pid, layer, prb.num_variables, len(prb.linear_constraints), method] + diff
                    writer.writerow(row)  # Write row immediately
                    num_complete += 1
                    if num_complete == len(futures):
                        print(f'Data has been written to {new_path}.csv')
                        for process in executor._processes.values():
                            os.kill(process.pid, signal.SIGTERM)
    print(time.perf_counter()- all_start_time)