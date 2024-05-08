from quBLP.problemtemplate import FacilityLocationProblem as FLP
problem = FLP(2,2,[[3,3],[2,2]],[2,2])
problem.optimize()
print(problem.solution)