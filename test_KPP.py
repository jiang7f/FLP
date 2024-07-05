from quBLP.problemtemplate import KPartitionProblem as KPP
kpp = KPP(3, [1, 2], [[[0, 1], 1], [[0, 2], 2],[[2,1],10]])

kpp.set_optimization_direction('max')
kpp.set_algorithm_optimization_method('commute', 400)
kpp.optimize(params_optimization_method='COBYLA',
             max_iter=150,
             num_layers=2,  
             need_draw=False,
             use_decompose=True,
             circuit_type='qiskit',  
             mcx_mode='constant',
             use_debug=True,
             backend='AerSimulator',   #'FakeQuebec' # 'AerSimulator'
            )

print(kpp.find_state_probability([1, 0, 1, 0, 0, 1, 0, 1]))
print(kpp.find_state_probability([0, 1, 0, 1, 1, 0, 1, 0]))