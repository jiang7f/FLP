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
    mcx_mode='linear',
    backend='FakeQuebec',  # 'FakeQuebec' # 'AerSimulator'
    feedback=['depth', 'culled_depth', 'latency'],
)
methods = ['penalty', 'cyclic', 'commute', 'HEA']
raw_depth = [[] for _ in range(len(methods))]
depth_without_one_qubit_gate = [[] for _ in range(len(methods))]
# latency = [[] for _ in range(len(methods))]
problems = [FLP(2,1,[[10, 2]],[2, 2]), 
            FLP(2,2,[[10, 2],[1, 20]],[1, 1]),
            FLP(3,2,[[10, 2, 3],[1, 20, 3]],[1, 1, 1]),
            FLP(3,3,[[10, 2, 3],[10, 2, 3],[10, 2, 3]],[1, 1, 1]),
            FLP(5,5,[[1, 1, 1, 1, 1],[1, 1, 1, 1, 1],[1, 1, 1, 1, 1],[1, 1, 1, 1, 1],[1, 1, 1, 1, 1]],[1, 1, 1, 1, 1])]
flp = problems[3]
data = [[] for _ in range(len(methods))]
print(f'问题规模:{flp.m} * {flp.n}')
print('层数 1,  2,  3,  4,  5')
feedback=['depth', 'culled_depth', 'latency']
layers = range(1, 5)
for idx, method in enumerate(methods):
  flp.set_algorithm_optimization_method(method, 400)
  for num_layers in layers:
    circuit_option.num_layers = num_layers
    data[idx].append(flp.optimize(optimizer_option, circuit_option))

for dict_term in feedback:
  print(dict_term)
  for idx, method in enumerate(methods):
      print(f'method: {method:<8} {[data[idx][l - 1][dict_term] for l in layers]}')
