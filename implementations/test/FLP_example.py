from quBLP.problemtemplate import FacilityLocationProblem as FLP
# flp = FLP(2,2,[[2,10], [10,2]],[2,2])
flp = FLP(2, 2,[[2, 1], [2, 1]], [1, 2])
# flp = FLP(6, 4,[[3, 3, 2, 2, 2, 2], [3, 3, 2, 2, 2, 2], [3, 3, 2, 2, 2, 2], [3, 3, 2, 2, 2, 2]], [2, 2, 2, 2, 2, 2])
# flp = FLP(3, 2,[[2, 1, 3], [2, 1, 3], [2, 1, 3]], [1, 2, 3])
# print(flp.fast_solve_driver_bitstr())
flp.set_optimization_direction('min')
# flp.set_algorithm_optimization_method('penalty', 400)
flp.optimize(params_optimization_method='COBYLA',
             max_iter=150,
             num_layers=1,
             need_draw=False,
             use_decompose=True,
             circuit_type='qiskit',
             mcx_mode='constant',
             
             use_debug=True)
print(flp.find_state_probability([0, 1, 0, 1, 0, 1, 0, 0, 0, 0]))
print(flp.find_state_probability([1, 0, 1, 0, 0, 0]))