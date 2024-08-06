import os
import time
import csv
import signal
import random
import itertools
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from quBLP.problemtemplate import FacilityLocationProblem as FLP
from quBLP.problemtemplate import GraphColoringProblem as GCP
from quBLP.problemtemplate import KPartitionProblem as KPP
from quBLP.models import CircuitOption, OptimizerOption
from quBLP.analysis import generater
import numpy as np


np.random.seed(0x7f)

script_path = os.path.abspath(__file__)
new_path = script_path.replace('experiment', 'data')[:-3]

optimizer_option = OptimizerOption(
    params_optimization_method='COBYLA',
    max_iter=150
)
flp_problems_pkg, flp_configs_pkg = generater.generate_flp(0, [(1, 2), (2, 3), (3, 3), (3, 4)], 1, 20)
gcp_problems_pkg, gcp_configs_pkg = generater.generate_gcp(4, [(3, 1), (3, 2), (4, 2), (4, 3)])
kpp_problems_pkg, kpp_configs_pkg = generater.generate_kpp(0, [(4, 2, 3), (6, 3, 5), (8, 3, 7), (9, 3, 8)], 1, 20)

problems_pkg = list(itertools.chain(enumerate(flp_problems_pkg), enumerate(gcp_problems_pkg), enumerate(kpp_problems_pkg)))

configs_pkg = flp_configs_pkg + gcp_configs_pkg + kpp_configs_pkg
with open(f"{new_path}.config", "w") as file:
    for pkid, configs in enumerate(configs_pkg):
        for pbid, problem in enumerate(configs):
            file.write(f'{pkid}-{pbid}: {problem}\n')

layers = range(7, 8)
methods = ['cyclic']
evaluation_metrics = ['ARG', 'in_constraints_probs', 'best_solution_probs', 'iteration_count']
headers = ['pkid', 'pbid', 'layers', "variables", 'constraints', 'method'] + evaluation_metrics

def process_layer(prb, num_layers, method):
    prb.set_algorithm_optimization_method(method, 15)
    circuit_option = CircuitOption(
        num_layers=num_layers,
        need_draw=False,
        use_decompose=True,
        circuit_type='qiskit',
        mcx_mode='constant',
        backend='ddsim' if method == 'commute' else 'AerSimulator-GPU', # 'FakeQuebec' # 'AerSimulator'
    )
    ARG, in_constraints_probs, best_solution_probs, iteration_count = prb.optimize(optimizer_option, circuit_option)
    return [ARG, in_constraints_probs, best_solution_probs, iteration_count]

if __name__ == '__main__':
    all_start_time = time.perf_counter()
    set_timeout = 60 * 60 * 24 * 3 # Set timeout duration
    num_complete = 0
    print(new_path)
    with open(f'{new_path}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # Write headers once

        num_processes_cpu = os.cpu_count()
        # pkid-pbid: 问题包序-包内序号
        for pkid, (diff_level, problems) in enumerate(problems_pkg):
            for method in methods:
                if method == 'commute':
                    num_processes = num_processes_cpu // 2
                else:
                    num_processes = 2**(4 - diff_level)
                with ProcessPoolExecutor(max_workers=num_processes) as executor:
                    futures = []
                    for layer in layers: 
                        for pbid, prb in enumerate(problems):
                            print(f'{pkid}-{pbid}, {layer}, {method} build')
                            future = executor.submit(process_layer, prb, layer, method)
                            futures.append((future, prb, pkid, pbid, layer, method))

                    start_time = time.perf_counter()
                    for future, prb, pkid, pbid, layer, method in futures:
                        current_time = time.perf_counter()
                        remaining_time = max(set_timeout - (current_time - start_time), 0)
                        diff = []
                        try:
                            metrics = future.result(timeout=remaining_time)
                            diff.extend(metrics)
                            print(f"Task for problem {pkid}-{pbid} L={layer} {method} executed successfully.")
                        except MemoryError:
                            print(f"Task for problem {pkid}-{pbid} L={layer} {method} encountered a MemoryError.")
                            for dict_term in evaluation_metrics:
                                diff.append('memory_error')
                        except TimeoutError:
                            print(f"Task for problem {pkid}-{pbid} L={layer} {method} timed out.")
                            for dict_term in evaluation_metrics:
                                diff.append('timeout')
                        except Exception as e:
                            print(f"An error occurred: {e}")
                        finally:
                            row = [pkid, pbid, layer, prb.num_variables, len(prb.linear_constraints), method] + diff
                            writer.writerow(row)  # Write row immediately
                            num_complete += 1
                            if num_complete == len(futures):
                                print(f'problem_pkg_{pkid} has finished')
                                for process in executor._processes.values():
                                    os.kill(process.pid, signal.SIGTERM)
    print(f'Data has been written to {new_path}.csv')
    print(time.perf_counter()- all_start_time)