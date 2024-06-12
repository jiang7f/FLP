from scipy.optimize import minimize
import numpy as np

def train_non_gradient(num_params, cost_function, max_iter):

    def callback(params):
        nonlocal iteration_count
        iteration_count += 1
        if iteration_count % 10 == 0:
            print(f"Iteration {iteration_count}, Result: {cost_function(params)}")

    def train_with_scipy(params, cost_function, max_iter):
        result = minimize(cost_function, params, method='COBYLA', options={'maxiter': max_iter}, callback=callback)
        return result.x
    
    iteration_count = 0 
    params = 2*np.pi*np.random.uniform(0, 1, num_params)
    return train_with_scipy(params, cost_function, max_iter)