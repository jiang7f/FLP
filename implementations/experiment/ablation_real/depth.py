import os
import time
import csv
import signal
import random
import numpy as np
import itertools
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from quBLP.problemtemplate import FacilityLocationProblem as FLP
from quBLP.problemtemplate import GraphColoringProblem as GCP
from quBLP.problemtemplate import KPartitionProblem as KPP
from quBLP.models import CircuitOption, OptimizerOption, ConstrainedBinaryOptimization
from quBLP.analysis import generater

random.seed(0x7ff)
np.random.seed(0xdb)

script_path = os.path.abspath(__file__)
new_path = script_path.replace('experiment', 'data')[:-3]

flp_problems_pkg, flp_configs_pkg = generater.generate_flp(1, [(1, 2)], 1, 20)
gcp_problems_pkg, gcp_configs_pkg = generater.generate_gcp(1, [(3, 1)])
kpp_problems_pkg, kpp_configs_pkg = generater.generate_kpp(1, [(4, 2, 3)], 1, 20)
# flp_problems_pkg, flp_configs_pkg = generater.generate_flp(1, [(2, 3)], 1, 20)
# gcp_problems_pkg, gcp_configs_pkg = generater.generate_gcp(1, [(3, 2)])
# kpp_problems_pkg, kpp_configs_pkg = generater.generate_kpp(1, [(6, 3, 5)], 1, 20)

problems_pkg = flp_problems_pkg + gcp_problems_pkg + kpp_problems_pkg

configs_pkg = flp_configs_pkg + gcp_configs_pkg + kpp_configs_pkg
with open(f"{new_path}.config", "w") as file:
    for pkid, configs in enumerate(configs_pkg):
        for problem in configs:
            file.write(f'{pkid}: {problem}\n')

backends = ['FakeKyiv', 'FakeTorino', 'FakeBrisbane']
feedback = ['depth', 'culled_depth', 'transpile_time']
strategys = [[1, 2], [1, 1], [1, 0], [0, 2], [0, 1], [0, 0]]
# strategys = [[1, 2], [1, 1], [1, 0]]
headers = ['pkid', 'backend', 'strategy'] + feedback
file_name = __file__.split("/")[-1].split(".")[0]

def process_layer(prb : ConstrainedBinaryOptimization, backend, strategy):
    prb.set_algorithm_optimization_method('commute', 400)
    circuit_option = CircuitOption(
        num_layers=1,
        need_draw=False,
        use_decompose=False,
        circuit_type='qiskit',
        mcx_mode='linear',
        backend=backend,
        feedback=feedback,
    )
    optimizer_option = OptimizerOption(
        params_optimization_method='COBYLA',
        max_iter=150,
        # use_local_params=True,
        # opt_id= '_'.join([str(x) for x in [file_name, pkid]]),
    )
    if strategy[0]:
        circuit_option.use_decompose = True
    if strategy[1]:
        result = prb.dichotomy_optimize(optimizer_option, circuit_option, strategy[1])
    else:
        result = prb.optimize(optimizer_option, circuit_option)
    return result

if __name__ == '__main__':
    set_timeout = 60 * 60 * 2 # Set timeout duration
    num_complete = 0
    script_path = os.path.abspath(__file__)
    new_path = script_path.replace('experiment', 'data')[:-3]
    print(new_path)

    with open(f'{new_path}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # Write headers once

        num_processes_cpu = os.cpu_count()
        with ProcessPoolExecutor(max_workers=num_processes_cpu//2) as executor:
            futures = []
            for pkid, problems in enumerate(problems_pkg):
                for problem in problems:
                    for backend in backends:
                        for strategy in strategys:
                            future = executor.submit(process_layer, problem, backend, strategy)
                            futures.append((future, pkid, problem, backend, strategy))

            start_time = time.perf_counter()
            for future, pkid, problem, backend, strategy in futures:
                num_layers = 1
                current_time = time.perf_counter()
                remaining_time = max(set_timeout - (current_time - start_time), 0)
                diff = []
                try:
                    result = future.result(timeout=remaining_time)
                    for dict_term in feedback:
                        diff.append(result[dict_term])
                    print(f"Task for problem { pkid, backend, strategy} executed successfully.")
                except MemoryError:
                    for dict_term in feedback:
                        diff.append('memory_error')
                    print(f"Task for problem {pkid, backend, strategy} encountered a MemoryError.")
                except TimeoutError:
                    for dict_term in feedback:
                        diff.append('timeout')
                    print(f"Task for problem {pkid, backend, strategy} timed out.")
                finally:
                    row = [pkid, backend, strategy] + diff
                    writer.writerow(row)  # Write row immediately
                    num_complete += 1
                    if num_complete == len(futures):
                        print(f'Data has been written to {new_path}.csv')
                        for process in executor._processes.values():
                            os.kill(process.pid, signal.SIGTERM)