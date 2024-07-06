from quBLP.problemtemplate import GraphColoringProblem as GCP
from quBLP.models import CircuitOption, OptimizerOption
# gcp = GCP(3,[[0,1]])
gcp = GCP(3,[[0, 1], [0, 2]], False)
# gcp = GCP(4,[[0, 1], [1, 2], [2, 3], [3, 0]], False)
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
gcp.optimize(optimizer_option, circuit_option)
print(gcp.find_state_probability([0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0]))
print(gcp.find_state_probability([0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0]))
print(gcp.find_state_probability([1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0]))
print(gcp.find_state_probability([0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1]))