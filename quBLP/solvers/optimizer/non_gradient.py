from quBLP.utils import iprint
from scipy.optimize import minimize
import numpy as np
from ...models import OptimizerOption

def train_non_gradient(optimizer_option: OptimizerOption):

    def callback(params):
        nonlocal iteration_count
        iteration_count += 1
        if iteration_count % 10 == 0:
            iprint(f"Iteration {iteration_count}, Result: {optimizer_option.circuit_cost_function(params)}")

    def train_with_scipy(params, cost_function, max_iter):
        result = minimize(cost_function, params, method='COBYLA', options={'maxiter': max_iter}, callback=callback)
        return result.x, iteration_count
    
    iteration_count = 0 
    params = 2*np.pi*np.random.uniform(0, 1, optimizer_option.num_params)
    return train_with_scipy(params, optimizer_option.circuit_cost_function, optimizer_option.max_iter)