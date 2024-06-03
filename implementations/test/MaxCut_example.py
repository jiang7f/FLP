from quBLP.problemtemplate import MaxCutProblem as MCP
mcp = MCP(4,[[0,1], [1,2], [2,3], [3, 0]], False)

mcp.set_optimization_method_type('penalty', 30)
mcp.optimize(max_iter=300,num_layers=2)
print(mcp.collapse_state)
print(mcp.probs)
labels = mcp.collapse_state
values = mcp.probs

import matplotlib.pyplot as plt
plt.bar([str(x) for x in labels], values)
plt.title('Probability Distribution')
plt.xlabel('State')
plt.ylabel('Probability')
plt.xticks(rotation=45, ha='right')
plt.show()