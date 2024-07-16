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

random.seed(0x7fff)

script_path = os.path.abspath(__file__)
new_path = script_path.replace('experiment', 'data')[:-3]

optimizer_option = OptimizerOption(
    params_optimization_method='COBYLA',
    max_iter=150
)

kpp_problems_pkg, kpp_configs_pkg = generater.generate_kpp(1, [(9, [3, 3, 3], 8), (8, [2, 2, 4], 7), (7, [2, 2, 3], 6), (6, [2, 2, 2], 5), (5, [1, 2, 2], 4), (6, [3, 3], 3), (3, [1, 1, 1], 2)], 1, 20)
# kpp_problems_pkg, kpp_configs_pkg = generater.generate_kpp(1, [(3, [1, 1, 1], 2), (6, [3, 3], 3), (5, [1, 2, 2], 4), (6, [2, 2, 2], 5), (7, [2, 2, 3], 6), (8, [2, 2, 4], 7), (9, [3, 3, 3], 8)], 1, 20)
problems_pkg = kpp_problems_pkg

configs_pkg = kpp_configs_pkg
with open(f"{new_path}_configs.txt", "w") as file:
    for pkid, configs in enumerate(configs_pkg):
        for pbid, problem in enumerate(configs):
            file.write(f'{pkid}-{pbid}: {problem}\n')

layers = range(1, 2)
# mcx_modes = ['constant', 'linear']
if_use_decompose = [True, False]
feedback = ['transpile_time', 'depth', 'culled_depth', 'rss_usage']

headers = ["pkid", 'pbid', 'layers', "use_decompose"] + feedback

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
    set_timeout = 60 * 60 * 1 # Set timeout duration
    num_complete = 0
    script_path = os.path.abspath(__file__)
    new_path = script_path.replace('experiment', 'data')[:-3]
    print(new_path)

    with open(f'{new_path}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # Write headers once

        # num_processes_cpu = os.cpu_count()
        # num_processes = num_processes_cpu // 2
        with ProcessPoolExecutor(max_workers=7) as executor:
            futures = []
            for use_decompose in if_use_decompose:
                for pkid, problems in enumerate(problems_pkg):
                    for pbid, problem in enumerate(problems):
                        for num_layers in layers:
                            future = executor.submit(process_layer, problem, use_decompose, num_layers, feedback)
                            futures.append((future, pkid, pbid, use_decompose, num_layers))

            start_time = time.perf_counter()
            for future, pkid, pbid, use_decompose, num_layers in futures:
                current_time = time.perf_counter()
                remaining_time = max(set_timeout - (current_time - start_time), 0)
                diff = []
                try:
                    result = future.result(timeout=remaining_time)
                    for dict_term in feedback:
                        diff.append(result[dict_term])
                    print(f"Task for problem {pkid}-{pbid}, use_decompose {use_decompose}, num_layers {num_layers} executed successfully.")
                except MemoryError:
                    for dict_term in feedback:
                        diff.append('memory_error')
                    print(f"Task for problem {pkid}-{pbid}, use_decompose {use_decompose}, num_layers {num_layers} encountered a MemoryError.")
                except TimeoutError:
                    for dict_term in feedback:
                        diff.append('timeout')
                    print(f"Task for problem {pkid}-{pbid}, use_decompose {use_decompose}, num_layers {num_layers} timed out.")
                finally:
                    row = [pkid, pbid, num_layers, use_decompose] + diff
                    writer.writerow(row)  # Write row immediately
                    num_complete += 1
                    if num_complete == len(futures):
                        print(f'Data has been written to {new_path}.csv')
                        for process in executor._processes.values():
                            os.kill(process.pid, signal.SIGTERM)