from quBLP.problemtemplate import GraphColoringProblem as GCP
from quBLP.models import CircuitOption, OptimizerOption
problems = [
    GCP(3, [[0, 1]]), 
    GCP(3, [[0, 1], [1, 2]]),
    GCP(4, [[0, 1]]),
    GCP(4, [[0, 1], [1, 2]]),
    GCP(4, [[0, 1], [1, 2], [2, 3]])
]
gcp = problems[0]
# 这个层数和迭代次数要稍微多一点.
# print(gcp.solution)
gcp.set_optimization_direction('min')
gcp.set_algorithm_optimization_method('commute', 400)
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
    backend='AerSimulator',  # 'FakeQuebec' # 'AerSimulator'
    feedback=None,
)
print(gcp.optimize(optimizer_option, circuit_option))