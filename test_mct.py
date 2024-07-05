from quBLP.problemtemplate import MaxCutProblem as MCP
from quBLP.models import CircuitOption, OptimizerOption
mcp = MCP(4,[[0,1], [0,2], [0,3]], False)

mcp.set_algorithm_optimization_method('penalty', 30)
optimizer_option = OptimizerOption(
    params_optimization_method='COBYLA',
    max_iter=300
)
circuit_option = CircuitOption(
    num_layers=4,
    need_draw=False,
    use_decompose=True,
    circuit_type='qiskit',
    mcx_mode='constant',
    use_debug=False,
    backend='AerSimulator'  # 'FakeQuebec' # 'AerSimulator'
)
mcp.optimize(optimizer_option, circuit_option)
# print(mcp.collapse_state)
# print(mcp.probs)
labels = mcp.collapse_state
values = mcp.probs

import matplotlib.pyplot as plt
plt.bar([str(x) for x in labels], values)
plt.title('Probability Distribution')
plt.xlabel('State')
plt.ylabel('Probability')
plt.xticks(rotation=45, ha='right')
plt.show()