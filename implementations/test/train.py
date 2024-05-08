from quBLP import ConstrainedBinaryOptimization
from pennylane import numpy as np
problem = ConstrainedBinaryOptimization()
problem.add_binary_variables('x', [5])
problem.add_constraint("x_1 + x_2 - x_0 - x_3 == 1")
problem._add_linear_constraint([1,1,1,0,1,2])
# problem._add_linear_constraint([1,-1,1,-1,0,1])
# problem._add_linear_constraint([0,1,-1,1,-1,1])
def objective(x):
    return np.sum(x)
problem.add_objective(objective)
problem.optimize()
