from quBLP.problemtemplate import MaxCutProblem as MCP
mcp = MCP(4,[[0,1], [1,2], [2,3]], False)

mcp.set_algorithm_optimization_method('penalty', 30)
mcp.optimize(params_optimization_method='COBYLA', max_iter=300,num_layers=2, need_draw=True, use_Ho_gate_list=True, circuit_type='qiskit')

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