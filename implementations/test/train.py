from quBLP import ConstrainedBinaryOptimization
from pennylane import numpy as np
problem = ConstrainedBinaryOptimization()
problem.add_binary_variables('x', [5])
problem.add_constraint("x_0 + x_1 + x_2 + x_3 - x_4 == 3")
def objective(x):
    return x[3]*10 +sum(x)
problem.set_algorithm_optimization_method('penalty', 40)
problem.add_objective(objective)
problem.optimize(100, 0.1, 5)
