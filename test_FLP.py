from quBLP.problemtemplate import FacilityLocationProblem as FLP
from quBLP.models import CircuitOption, OptimizerOption


optimizer_option = OptimizerOption(
    params_optimization_method='COBYLA',
    max_iter=150
)
circuit_option = CircuitOption(
    num_layers=4,
    need_draw=False,
    use_decompose=True,
    circuit_type='qiskit',
    mcx_mode='constant',
    use_debug=True,
    backend='FakeQuebec'  # 'FakeQuebec' # 'AerSimulator'
)
# flp.optimize(optimizer_option, circuit_option)
methods = ['penalty', 'cyclic', 'commute', 'HEA']
depth_without_one_qubit_gate = [[] for _ in range(len(methods))]
latency = [[] for _ in range(len(methods))]
problems = [FLP(2,1,[[10, 2]],[2, 2]), 
            FLP(2,2,[[10, 2],[1, 20]],[1, 1]),
            FLP(3,2,[[10, 2, 3],[1, 20, 3]],[1, 1, 1]),
            FLP(3,3,[[10, 2, 3],[10, 2, 3],[10, 2, 3]],[1, 1, 1])]
# FLP(5,3,[[3,3,3,3,3],[3,3,3,3,3],[3,3,3,3,3]],[2,2,2,2,2])
# flp = FLP(2,2,[[2, 1],[2, 1]],[1, 2])
flp = problems[1]
for idx, method in enumerate(methods):
  flp.set_algorithm_optimization_method(method, 400)
  for num_layers in range(1, 6):
    circuit_option.num_layers = num_layers
    data = flp.optimize(optimizer_option, circuit_option)
    depth_without_one_qubit_gate[idx].append(data[0])
    latency[idx].append(data[1])
print(f'问题规模:{flp.m} * {flp.n}')
print('depth')
print('层数 1,  2,  3,  4,  5')
for idx, method in enumerate(methods):
  print(f'method: {method}')
  print(depth_without_one_qubit_gate[idx])
print('latency')
for idx, method in enumerate(methods):
  print(f'method: {method}')
  print(latency[idx])
# print(flp.find_state_probability([0, 1, 0, 1, 0, 0]))
# print(flp.find_state_probability([0, 1, 0, 1, 0, 1, 0, 0, 0, 0]))