from quBLP.problemtemplate import FacilityLocationProblem as FLP
flp = FLP(2,2,[[2,-10], [10,2]],[2,2])
# flp = FLP(2,2,[[3,3],[2,2]],[2,2])
# flp = FLP(2,2,[[2,1],[2,1]],[1,2])
# print(flp.fast_solve_driver_bitstr())
flp.set_optimization_direction('min')
# flp.set_algorithm_optimization_method('cyclic', 40)
flp.optimize(max_iter=150,num_layers=5, need_draw=True, use_Ho_gate_list=False, use_decompose=True)

print(flp.find_state_probability([1, 0, 1, 0, 1, 0, 0, 0, 0, 0]))
print(flp.find_state_probability([0, 1, 0, 1, 0, 1, 0, 0, 0, 0]))