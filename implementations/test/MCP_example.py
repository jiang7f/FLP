from quBLP.problemtemplate import MaximumCliqueProblem as MCP
mcp = MCP(5,[[0, 1], [3, 4]], False)
# print(mcp.solution) 
mcp.set_algorithm_optimization_method('penalty', 40)
mcp.optimize(max_iter=150,num_layers=2, need_draw=True)