from quBLP.problemtemplate import GraphColoringProblem as GCP
# gcp = GCP(2,[[0, 1]], False)
# gcp = GCP( 2,[], False)
gcp = GCP(3,[[0, 1]], False)
# gcp = GCP(4,[[0, 1], [1, 2], [2, 3], [3, 0]], False)
gcp.set_optimization_direction('min')
# 这个层数和迭代次数要稍微多一点.
# gcp.set_optimization_method_type('HEA', 40)
gcp.optimize(params_optimization_method='COBYLA', max_iter=300, num_layers=5, need_draw=True)
# print(gcp.solution)