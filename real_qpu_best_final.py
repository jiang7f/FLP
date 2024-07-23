import os
import time
import csv
import signal
import random
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from quBLP.models import CircuitOption, OptimizerOption
from quBLP.analysis import generater
from multiprocessing import Manager, Lock
from quBLP.solvers.cloud_execute.cloud_manager import CloudManager
from quBLP.solvers.cloud_execute import get_IBM_service
import traceback

random.seed(0x7ff)

script_path = os.path.abspath(__file__)
new_path = script_path.replace('experiment', 'data')[:-3]
# print(__name__)
flp_problems_pkg, flp_configs_pkg = generater.generate_flp(3, [(1, 2)], 1, 20)
gcp_problems_pkg, gcp_configs_pkg = generater.generate_gcp(3, [(3, 1)])
kpp_problems_pkg, kpp_configs_pkg = generater.generate_kpp(3, [(4, 2, 3)], 1, 20)

problems_pkg = [[flp_problems_pkg[0][1]], [gcp_problems_pkg[0][1]], [kpp_problems_pkg[0][1]]]
configs_pkg = [[flp_configs_pkg[0][1]], [gcp_configs_pkg[0][1]], [kpp_configs_pkg[0][1]]]

with open(f"{new_path}.config", "w") as file:
    for pkid, configs in enumerate(configs_pkg):
        for problem in configs:
            file.write(f'{pkid}: {problem}\n')
# exit()

methods = ['penalty', 'cyclic', 'commute', 'HEA']
backends = ['ibm_fez']
evaluation_metrics = ['ARG', 'in_constraints_probs', 'best_solution_probs', 'iteration_count']
shotss = [1024]
headers = ["pkid", 'pbid', 'layers', 'method', 'backend', 'shots'] + evaluation_metrics

use_free = False

num_problems = sum([len(problem) for problem in problems_pkg])
num_methods = len(methods)

file_name = __file__.split("\\")[-1].split(".")[0]
# exit()
def process_layer( pkid, pbid, prb, method, backend, shots, shared_cloud_manager):
    try:
        prb.set_algorithm_optimization_method(method, 400)
        circuit_option = CircuitOption(
            num_layers=1 if method == 'commute' else 7,
            need_draw=False,
            use_decompose=True,
            circuit_type='qiskit',
            mcx_mode='constant',
            backend=backend,
            shots=shots,
            use_IBM_service_mode='group',
            use_free_IBM_service=use_free,
            cloud_manager=shared_cloud_manager,
            # use_fake_IBM_service=True
        )
        optimizer_option = OptimizerOption(
            params_optimization_method='COBYLA',
            max_iter=50,
            use_local_params = True,
            opt_id= '_'.join([str(x) for x in [file_name, pkid, pbid, method, backend, shots]]),
        )
        result = prb.optimize(optimizer_option, circuit_option)
        return result
    except Exception as e:
        error_message = traceback.format_exc()
        print(f"Error in {method} on {backend}: {e, error_message}")
        return ['error'] * len(evaluation_metrics)

if __name__ == '__main__':
    set_timeout = 60 * 60 * 24 * 7
    num_complete = 0
    script_path = os.path.abspath(__file__)
    new_path = script_path.replace('experiment', 'data')[:-3]
    print(new_path)

    with open(f'{new_path}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)

        num_cpu = os.cpu_count()
        with Manager() as manager:
            job_dic = manager.dict()
            job_category = [tuple((backend, shots)) for backend in backends for shots in shotss]
            for key in job_category:
                job_dic[key] = manager.Queue()
            results = manager.dict()
            one_job_lens=manager.Value('i', num_problems * num_methods)
            shared_cloud_manager = CloudManager(
                job_dic,
                results,
                one_job_lens=one_job_lens,
                sleep_interval=10,
                use_free=use_free,
            )

            with ProcessPoolExecutor(max_workers=num_cpu - 2) as executor:
                for key in job_category:
                    print(f"{key} manager build")
                    executor.submit(shared_cloud_manager.process_task, key)
                futures = []
                for backend in backends:
                    for shots in shotss:
                        for pkid, problems in enumerate(problems_pkg):
                            for pbid, problem in enumerate(problems):
                                for method in methods:
                                    print(f'{pkid, method, backend, shots} build')
                                    future = executor.submit(process_layer,  pkid, pbid, problem, method, backend, shots, shared_cloud_manager)
                                    futures.append((future, pkid, pbid, method, backend, shots))

                start_time = time.perf_counter()
                for future, pkid, pbid, method, backend, shots in futures:
                    num_layers = 1 if method == 'commute' else 7
                    current_time = time.perf_counter()
                    remaining_time = max(set_timeout - (current_time - start_time), 0)
                    diff = []
                    try:
                        metrics = future.result(timeout=remaining_time)
                        diff.extend(metrics)
                        print(f"Task for problem {pkid, method, backend, shots} num_layers {num_layers} executed successfully.")
                    except MemoryError:
                        diff = ['memory_error'] * len(evaluation_metrics)
                        print(f"Task for problem {pkid, method, backend, shots} num_layers {num_layers} encountered a MemoryError.")
                    except TimeoutError:
                        diff = ['timeout'] * len(evaluation_metrics)
                        print(f"Task for problem {pkid, method, backend, shots} num_layers {num_layers} timed out.")
                    except Exception as e:
                        diff = ['error'] * len(evaluation_metrics)
                        error_message = traceback.format_exc()
                        print(f"Task for problem {pkid, method, backend, shots} num_layers {num_layers} encountered an error: {error_message}")
                    finally:
                        row = [pkid, pbid, num_layers, method, backend, shots] + diff
                        writer.writerow(row)
                        num_complete += 1
                        if num_complete == len(futures):
                            print(f'Data has been written to {new_path}.csv')
