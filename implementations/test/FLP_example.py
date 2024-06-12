from quBLP.problemtemplate import FacilityLocationProblem as FLP
flp = FLP(2,2,[[3,3],[2,2]],[2,2], False)
# flp = FLP(2,2,[[3,3],[2,2]],[2,2])
# flp = FLP(2,2,[[2,1],[2,1]],[1,2])
flp.set_optimization_direction('min')
flp.set_optimization_method_type('cyclic', 40)
flp.optimize(max_iter=150,num_layers=2, need_draw=True)
print(flp.find_state_probability([1, 0, 1, 0, 1, 0, 0, 0, 0, 0]))
print(flp.find_state_probability([0, 1, 0, 1, 0, 1, 0, 0, 0, 0]))