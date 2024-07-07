import csv
from quBLP.problemtemplate import KPartitionProblem as KPP
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
    # feedback=['depth', 'culled_depth', 'latency'],
)
methods = ['penalty', 'cyclic', 'commute', 'HEA']
feedback = ['depth', 'culled_depth']

problems = [KPP(3,[2, 1],[[0, 1], [1, 2]]), 
            KPP(4,[3, 1],[[0, 1], [1, 2], [2, 3]]),
            KPP(5,[4, 1],[[0, 1], [1, 2], [2, 3], [3, 4]]),
            KPP(5,[3, 2], [[0, 1], [1, 2], [2, 3], [3, 4]]),
            KPP(5,[3, 1, 1], [[0, 1], [1, 2], [2, 3], [3, 4]])]
kpp = problems[4]
print(kpp.optimize(optimizer_option, circuit_option))