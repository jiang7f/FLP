from quBLP.problemtemplate import SetCoverProblem as SCP
scp = SCP(4,[[0, 1, 2], [1, 2], [0, 3]], False)
# pending [[0, 1], [1, 2], [0, 2]] 这组求出来的基础解系还得转换一下 
# scp.set_optimization_method_type('penalty', 40)

scp.optimize(params_optimization_method='COBYLA', max_iter=150,num_layers=4, need_draw=True)