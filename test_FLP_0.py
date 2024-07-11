from quBLP.problemtemplate import FacilityLocationProblem as FLP
from quBLP.models import CircuitOption, OptimizerOption
should_print = True
optimizer_option = OptimizerOption(
    params_optimization_method='Adam',
    max_iter=150
)
circuit_option = CircuitOption(
    num_layers=4,
    need_draw=False,
    use_decompose=True,
    circuit_type='qiskit',
    mcx_mode='constant',
    log_depth= False,
    backend='ddsim',  # 'FakeQuebec' # 'AerSimulator'
    # feedback=['transpile_time','run_time'],
)
methods = ['penalty', 'cyclic', 'commute', 'HEA']
raw_depth = [[] for _ in range(len(methods))]
depth_without_one_qubit_gate = [[] for _ in range(len(methods))]
# latency = [[] for _ in range(len(methods))]
problems = [FLP(2,1,[[10, 2]],[2, 2]), 
            FLP(2,2,[[10, 2],[1, 20]],[1, 1]),
            FLP(3,2,[[10, 2, 3],[1, 20, 3]],[1, 1, 1]),
            FLP(3,3,[[10, 2, 3],[10, 2, 3],[10, 2, 3]],[1, 1, 1]),
            FLP(4,3,[[10, 2, 3, 1],[10, 2, 3, 1],[10, 2, 3, 1],[10, 2, 3, 1]],[1, 1, 1, 1]),
            FLP(5,4,[[1, 1, 1, 1, 1],[1, 1, 1, 1, 1],[1, 1, 1, 1, 1],[1, 1, 1, 1, 1]],[1, 1, 1, 1, 1])]
flp = problems[4]
print(flp.collapse_state, flp.probs)
print(flp.optimize(optimizer_option, circuit_option))