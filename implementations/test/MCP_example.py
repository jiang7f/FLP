from quBLP.problemtemplate import MaximumCliqueProblem as MCP
mcp = MCP(4,[[0, 1]], False)
mcp.optimize()
print(mcp.solution)