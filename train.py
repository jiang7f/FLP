from quBLP import BinaryConstraintOptimization
from pennylane import numpy as np
problem = BinaryConstraintOptimization()
problem.add_binary_variables('x',5)
problem.add_constraint("x1+x2-x0-x3==1")
problem._add_linear_constraint([1,1,1,0,1,2])
# problem._add_linear_constraint([1,-1,1,-1,0,1])
# problem._add_linear_constraint([0,1,-1,1,-1,1])
def objective(x):
    return np.sum(x)
problem.add_objective(objective)
problem.optimize()
