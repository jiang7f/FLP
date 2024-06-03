from quBLP.problemtemplate import FacilityLocationProblem as FLP
flp = FLP(2,2,[[3,3],[2,2]],[2,2], False)
# flp = FLP(2,2,[[3,3],[2,2]],[2,2])
# flp = FLP(2,2,[[2,1],[2,1]],[1,2])
# flp.set_optimization_method_type('penalty', 40)
flp.optimize(max_iter=300,num_layers=2)
print(flp.find_state_probability([1, 0, 1, 0, 1, 0, 0, 0, 0, 0]))