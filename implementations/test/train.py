from quBLP import ConstrainedBinaryOptimization
from pennylane import numpy as np
problem = ConstrainedBinaryOptimization()
problem.add_binary_variables('x', range(5))
problem.add_constraint("x1+x2-x0-x3==1")
problem._add_linear_constraint([1,1,1,0,1,2])
# problem._add_linear_constraint([1,-1,1,-1,0,1])
# problem._add_linear_constraint([0,1,-1,1,-1,1])
def objective(x):
    return np.sum(x)
problem.add_objective(objective)
problem.optimize()
