from quBLP.problemtemplate import MaximumCliqueProblem as MCP
mcp = MCP(5,[[0, 1], [3, 4]], False)
# print(mcp.solution) 
# mcp.set_algorithm_optimization_method('penalty', 40)
mcp.optimize(params_optimization_method='COBYLA',
             max_iter=150,
             num_layers=10,
             need_draw=True,
             use_decompose=True,
             circuit_type='pennylane',
             mcx_mode='constant',
             debug=False)
# print(mcp.find_state_probability([0, 1, 0, 1, 0, 1, 0, 0, 0, 0]))
# print(mcp.find_state_probability([1, 0, 1, 0, 0, 0]))