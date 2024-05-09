from quBLP.problemtemplate import SetCoverProblem as SCP
scp = SCP(4,[[0, 1, 2], [1, 2], [0, 3]], False)
# pending [[0, 1], [1, 2], [0, 2]] 这组求出来的基础解系还得转换一下 
# scp = SCP(3,[[0, 1], [1, 2], [0, 2]], False)
scp.optimize(100, 0.1, 3)
# print(scp.solution)