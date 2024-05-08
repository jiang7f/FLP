from quBLP import ConstrainedBinaryOptimization
from pennylane import numpy as np
problem = ConstrainedBinaryOptimization()
problem.add_binary_variables('x', [5])
problem.add_constraint("x_0 + x_1 + x_2 + x_3 - x_4 == 3")
print(problem.get)
def objective(x):
    return x[3]*3 +sum(x)
problem.add_objective(objective)
problem.optimize()
