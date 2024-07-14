should_print = True
from quBLP.problemtemplate import FacilityLocationProblem as FLP
from quBLP.problemtemplate import GraphColoringProblem as GCP
from quBLP.problemtemplate import KPartitionProblem as KPP
from quBLP.models import CircuitOption, OptimizerOption
from quBLP.analysis import generater
optimizer_option = OptimizerOption(
    params_optimization_method='COBYLA',
    max_iter=150
)
circuit_option = CircuitOption(
    num_layers=5,
    need_draw=False,
    use_decompose=True,
    mcx_mode='linear',
    circuit_type='qiskit',
    backend='AerSimulator',  # 'FakeQuebec' # 'AerSimulator'
    # feedback=['depth', 'culled_depth', 'latency', 'width'],
    feedback=['transpile_time', 'run_time'],
)
methods = ['penalty', 'cyclic', 'commute', 'HEA']
raw_depth = [[] for _ in range(len(methods))]
depth_without_one_qubit_gate = [[] for _ in range(len(methods))]
# latency = [[] for _ in range(len(methods))]
flp_problems, flp_configs = generater.generate_flp(10, [(1, 2), (2, 2), (2, 3), (3, 4)], 1, 20)
gcp_problems, gcp_configs = generater.generate_gcp(10, [(3, 2), (4, 1), (4, 2), (4, 3)])
kpp_problems, kpp_configs = generater.generate_kpp(10, [(4, [2, 2], 3), (6, [2, 2, 2], 5), (8, [2, 2, 4], 7), (9, [3, 3, 3], 8)], 1, 20)

problems = flp_problems + gcp_problems + kpp_problems

# problems = [FLP(1, 2, [[2, 4]],[3, 2]), 
#             FLP(2, 2, [[10, 2],[1, 20]],[1, 1]),
#             FLP(2, 3, [[10, 2, 3],[1, 20, 3]],[1, 1, 1]),
#             FLP(3, 3, [[10, 2, 3],[10, 2, 3],[10, 2, 3]],[1, 1, 1]),
#             FLP(3, 4, [[10, 2, 3, 1],[10, 2, 3, 1],[10, 2, 3, 1],[10, 2, 3, 1]],[1, 1, 1, 1]),
#             FLP(4, 4, [[10, 2, 3, 1],[10, 2, 3, 1],[10, 2, 3, 1],[10, 2, 3, 1]],[1, 1, 1, 1]),
#             GCP(4, [[0, 1], [1, 2], [2, 3]]),
#             KPP(3, [2, 1], [[[0, 1], 1], [[1, 2], 1]]),
#             KPP(4, [2, 2],  [((2, 3), 17), ((1, 3), 9)]),
#             KPP(10, [3, 3, 4], [((3, 7), 6), ((1, 7), 4), ((1, 3), 11), ((0, 6), 20)]),
#             FLP(5, 5, [[1, 1, 1, 1, 1],[1, 1, 1, 1, 1],[1, 1, 1, 1, 1],[1, 1, 1, 1, 1],[1, 1, 1, 1, 1]],[1, 1, 1, 1, 1])]

print(problems[1].driver_bitstr)
# all_configs = [flp_configs, gcp_configs, kpp_configs]
# problem_types = ["FLP", "GCP", "KPP"]

# for problem_type, configs in zip(problem_types, all_configs):
#     print(f"{problem_type} Configurations:")
#     for config in configs:
#         print(*config)
#     print()
exit()
flp = problems[100]
# print(flp.get_best_cost())
# # print(flp.get_solution_bitstr())
# exit()
flp.set_algorithm_optimization_method('penalty', 300)
print(flp.optimize(optimizer_option, circuit_option))