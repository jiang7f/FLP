from quBLP.problemtemplate import FacilityLocationProblem as FLP
flp = FLP(2,1,[[10, 2]],[2, 2])
# flp = FLP(5,3,[[3,3,3,3,3],[3,3,3,3,3],[3,3,3,3,3]],[2,2,2,2,2])
# flp = FLP(2,2,[[10, 2],[1, 20]],[1, 1])
# flp = FLP(2,2,[[2, 1],[2, 1]],[1, 2])
# print(flp.fast_solve_driver_bitstr())
flp.set_optimization_direction('min')
flp.set_algorithm_optimization_method('commute', 400)
flp.optimize(params_optimization_method='COBYLA',
             max_iter=150,
             num_layers=4,
             need_draw=False,
             use_decompose=False,
             circuit_type='qiskit',
             mcx_mode='constant',
             use_debug=False,
             backend='AerSimulator',   #'FakeQuebec' # 'AerSimulator'
            )

print(flp.find_state_probability([0, 1, 0, 1, 0, 0]))
# print(flp.find_state_probability([0, 1, 0, 1, 0, 1, 0, 0, 0, 0]))