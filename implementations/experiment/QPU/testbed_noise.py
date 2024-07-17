should_print = True
import random
from quBLP.problemtemplate import FacilityLocationProblem as FLP
from quBLP.problemtemplate import GraphColoringProblem as GCP
from quBLP.problemtemplate import KPartitionProblem as KPP
from quBLP.models import CircuitOption, OptimizerOption
from quBLP.analysis import generater

random.seed(0x7fff)
optimizer_option = OptimizerOption(
    params_optimization_method='COBYLA',
    max_iter=150
)
circuit_option = CircuitOption(
    num_layers=1,
    need_draw=False,
    use_decompose=True,
    mcx_mode='constant',
    circuit_type='qiskit',
    backend='FakeKyiv',  # FakeKyiv, FakeTorino, FakeBrisbane
    # feedback=['depth', 'culled_depth', 'latency', 'width'],
    feedback=['latency', 'transpile_time'],
)
methods = ['penalty', 'cyclic', 'commute', 'HEA']
raw_depth = [[] for _ in range(len(methods))]
depth_without_one_qubit_gate = [[] for _ in range(len(methods))]
# latency = [[] for _ in range(len(methods))]
flp_problems, flp_configs = generater.generate_flp(1, [(1, 2)], 1, 20)
gcp_problems, gcp_configs = generater.generate_gcp(1, [(3, 1)])
kpp_problems, kpp_configs = generater.generate_kpp(1, [(4, 2, 3)], 1, 20)

problems_pkg = flp_problems + gcp_problems + kpp_problems
problems = [prb for problems in problems_pkg for prb in problems]
prb = problems[0]
prb.set_algorithm_optimization_method('penalty', 300)
print(prb.optimize(optimizer_option, circuit_option))
