from quBLP.problemtemplate import GraphColoringProblem as GCP
# gcp = GCP(3,[[0,1]])
gcp = GCP(3,[[0, 1], [0, 2]], False)
# gcp = GCP(4,[[0, 1], [1, 2], [2, 3], [3, 0]], False)
# 这个层数和迭代次数要稍微多一点.
# print(gcp.solution)
gcp.set_optimization_direction('min')
gcp.set_algorithm_optimization_method('commute', 400)
gcp.optimize(params_optimization_method='COBYLA',
             max_iter=150,
             num_layers=2,
             need_draw=False,
             use_decompose=True,
             circuit_type='qiskit',
             mcx_mode='constant',
             use_debug=False,
             backend='FakeQuebec',   #'FakeQuebec' #'AerSimulator'
            )

print(gcp.find_state_probability([0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0]))
print(gcp.find_state_probability([0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0]))
print(gcp.find_state_probability([1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0]))
print(gcp.find_state_probability([0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1]))