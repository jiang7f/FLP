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
    backend='AerSimulator',  # 'FakeQuebec' # 'AerSimulator'
    # feedback=['depth', 'culled_depth', 'latency', 'width'],
)
methods = ['penalty', 'cyclic', 'commute', 'HEA']
raw_depth = [[] for _ in range(len(methods))]
depth_without_one_qubit_gate = [[] for _ in range(len(methods))]
# latency = [[] for _ in range(len(methods))]
problems = [FLP(1,2,[[10, 2]],[2, 2]), 
            FLP(2,2,[[10, 2],[1, 20]],[1, 1]),
            FLP(2,3,[[10, 2, 3],[1, 20, 3]],[1, 1, 1]),
            FLP(3,3,[[10, 2, 3],[10, 2, 3],[10, 2, 3]],[1, 1, 1]),
            FLP(3,4,[[10, 2, 3, 1],[10, 2, 3, 1],[10, 2, 3, 1],[10, 2, 3, 1]],[1, 1, 1, 1]),
            FLP(5,5,[[1, 1, 1, 1, 1],[1, 1, 1, 1, 1],[1, 1, 1, 1, 1],[1, 1, 1, 1, 1],[1, 1, 1, 1, 1]],[1, 1, 1, 1, 1])]
flp = problems[0]
flp.set_algorithm_optimization_method('commute', 400)
ARG, in_constraints_probs, best_solution_probs = flp.optimize(optimizer_option, circuit_option)
print(ARG, in_constraints_probs, best_solution_probs)