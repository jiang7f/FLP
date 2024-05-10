from quBLP.problemtemplate import MaximumCliqueProblem as MCP
mcp = MCP(5,[[0, 1], [3, 4]], False)
mcp.optimize(30, 0.1, 2)
# print(mcp.solution)