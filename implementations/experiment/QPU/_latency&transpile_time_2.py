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

random.seed(0x7ff)

script_path = os.path.abspath(__file__)
new_path = script_path.replace('experiment', 'data')[:-3]

optimizer_option = OptimizerOption(
    params_optimization_method='COBYLA',
    max_iter=150
)

flp_problems_pkg, flp_configs_pkg = generater.generate_flp(1, [(1, 2)], 1, 20)
gcp_problems_pkg, gcp_configs_pkg = generater.generate_gcp(0, [(3, 2)])
kpp_problems_pkg, kpp_configs_pkg = generater.generate_kpp(0, [(6, 3, 5)], 1, 20)

problems_pkg = flp_problems_pkg + gcp_problems_pkg + kpp_problems_pkg

configs_pkg = flp_configs_pkg + gcp_configs_pkg + kpp_configs_pkg
with open(f"{new_path}.config", "w") as file:
    for pkid, configs in enumerate(configs_pkg):
        for problem in configs:
            file.write(f'{pkid}: {problem}\n')

# mcx_modes = ['constant', 'linear']
feedback = ['transpile_time', 'latency']
methods = ['penalty', 'cyclic', 'commute', 'HEA']
backends = ['FakeKyiv', 'FakeTorino', 'FakeBrisbane']

headers = ["pkid", 'layers', 'method', 'backend'] + feedback

def process_layer(prb, method, backend):
    prb.set_algorithm_optimization_method(method, 400)
    circuit_option = CircuitOption(
        num_layers=1 if method == 'commute' else 7,
        need_draw=False,
        use_decompose=True,
        circuit_type='qiskit',
        mcx_mode='linear',
        backend=backend,
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

        num_processes_cpu = os.cpu_count()
        with ProcessPoolExecutor(max_workers=num_processes_cpu) as executor:
            futures = []
            for pkid, problems in enumerate(problems_pkg):
                for problem in problems:
                    for backend in backends:
                        for method in methods:
                            future = executor.submit(process_layer, problem, method, backend)
                            futures.append((future, pkid, method, backend))

            start_time = time.perf_counter()
            for future, pkid, method, backend in futures:
                num_layers = 1 if method == 'commute' else 7
                current_time = time.perf_counter()
                remaining_time = max(set_timeout - (current_time - start_time), 0)
                diff = []
                try:
                    result = future.result(timeout=remaining_time)
                    for dict_term in feedback:
                        value = result[dict_term]
                        if method == 'commute' and dict_term == 'transpile_time':
                            value = sum(value)
                        diff.append(value)
                    print(f"Task for problem {pkid}, {backend} {method} num_layers {num_layers} executed successfully.")
                except MemoryError:
                    for dict_term in feedback:
                        diff.append('memory_error')
                    print(f"Task for problem {pkid}, {backend} {method} num_layers {num_layers} encountered a MemoryError.")
                except TimeoutError:
                    for dict_term in feedback:
                        diff.append('timeout')
                    print(f"Task for problem {pkid}, {backend} {method} num_layers {num_layers} timed out.")
                finally:
                    row = [pkid, num_layers, method, backend] + diff
                    writer.writerow(row)  # Write row immediately
                    num_complete += 1
                    if num_complete == len(futures):
                        print(f'Data has been written to {new_path}.csv')
                        for process in executor._processes.values():
                            os.kill(process.pid, signal.SIGTERM)