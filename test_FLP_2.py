from quBLP.problemtemplate import FacilityLocationProblem as FLP
from quBLP.models import CircuitOption, OptimizerOption

flp = FLP(2,1,[[10, 2]],[2, 2])
flp.set_algorithm_optimization_method('commute', 400)
optimizer_option = OptimizerOption(
    params_optimization_method='COBYLA',
    max_iter=150
)
circuit_option = CircuitOption(
    num_layers=1,
    need_draw=True,
    use_decompose=True,
    circuit_type='qiskit',
    mcx_mode='constant',
    backend='AerSimulator',  # 'FakeQuebec' # 'AerSimulator'
    feedback=None,
)
data = flp.optimize(optimizer_option, circuit_option)
