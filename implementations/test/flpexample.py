from quBLP.problemtemplate.FLP import FLProblem
problem = FLProblem(2,2,[[3,3],[2,2]],[2,2])
problem.optimize()
print(problem.solution)