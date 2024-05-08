from quBLP.problemtemplate import FacilityLocationProblem as FLP
flp = FLP(2,2,[[3,3],[2,2]],[2,2], False)
# flp = FLP(2,2,[[3,3],[2,2]],[2,2])
# flp = FLP(2,2,[[2,1],[2,1]],[1,2])
flp.optimize()
print(flp.solution)