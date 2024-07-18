should_print = True
import random
from quBLP.problemtemplate import FacilityLocationProblem as FLP
from quBLP.problemtemplate import GraphColoringProblem as GCP
from quBLP.problemtemplate import KPartitionProblem as KPP
from quBLP.models import CircuitOption, OptimizerOption
from quBLP.analysis import generater

random.seed(0x7f2f)
optimizer_option = OptimizerOption(
    params_optimization_method='COBYLA',
    max_iter=1
)
circuit_option = CircuitOption(
    num_layers=1,
    need_draw=False,
    use_decompose=True,
    mcx_mode='constant',
    circuit_type='qiskit',
    backend='AerSimulator',  # 'FakeQuebec' # 'AerSimulator'
    # backend='AerSimulator',  # 'FakeQuebec' # 'AerSimulator'
    # feedback=['depth', 'culled_depth', 'latency', 'width'],
    # feedback=['depth', 'culled_depth', 'transpile_time', 'rss_usage'],
)
methods = ['penalty', 'cyclic', 'commute', 'HEA']
raw_depth = [[] for _ in range(len(methods))]
depth_without_one_qubit_gate = [[] for _ in range(len(methods))]
# latency = [[] for _ in range(len(methods))]
flp_problems, flp_configs = generater.generate_flp(0, [(1, 2), (2, 3), (3, 3), (3, 4)], 1, 20)
gcp_problems, gcp_configs = generater.generate_gcp(0, [(3, 1), (3, 2), (4, 2), (4, 3)])
# kpp_problems, kpp_configs = generater.generate_kpp(1, [(4, [2, 2], 3), (6, [2, 2, 2], 5), (8, [2, 2, 4], 7), (9, [3, 3, 3], 8)], 1, 20)
# kpp_problems, kpp_configs = generater.generate_kpp(1, [(4, [2, 2], 3), (6, [2, 2, 2], 5), (8, [2, 2, 4], 7), (9, [3, 3, 3], 8)], 1, 20)
kpp_problems, kpp_configs = generater.generate_kpp(1, [(4, 2, 3), (6, 3, 5), (8, 3, 7), (9, 3, 8)], 1, 20)

problems_pkg = flp_problems + gcp_problems + kpp_problems
problems = [prb for problems in problems_pkg for prb in problems]

# all_configs = [flp_configs, gcp_configs, kpp_configs]
# problem_types = ["FLP", "GCP", "KPP"]

# for problem_type, configs in zip(problem_types, all_configs):
#     print(f"{problem_type} Configurations:")
#     for config in configs:
#         print(*config)
#     print()
prb = problems[0]
# print(prb.get_best_cost())
# print(flp.get_solution_bitstr())
# exit()
prb.set_algorithm_optimization_method('commute', 400)
# print(prb.optimize(optimizer_option, circuit_option))

print(prb.dichotomy_optimize(optimizer_option, circuit_option))
# print(kpp_configs)