from quBLP.utils.gadget import iprint, read_last_row
from scipy.optimize import minimize
import numpy as np
from ...models import OptimizerOption
import csv
import os

current_dir = os.path.dirname(__file__)

def train_non_gradient(optimizer_option: OptimizerOption):
    opt_id = optimizer_option.opt_id
    filename = os.path.join(current_dir, f'params_storage/{opt_id}.csv')
    
    iteration_count = 0
    
    def callback(params):
        nonlocal iteration_count
        iteration_count += 1
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            res = [iteration_count] + params.tolist()
            writer.writerow(res) 
        if iteration_count % 10 == 0:
            iprint(f"Iteration {iteration_count}, Result: {optimizer_option.circuit_cost_function(params)}")

    def train_with_scipy(params, cost_function, max_iter):
        remaining_iter = max_iter - iteration_count
        result = minimize(cost_function, params, method='COBYLA', options={'maxiter': remaining_iter}, callback=callback)
        return result.x, iteration_count

    if os.path.exists(filename):
        lastrow = read_last_row(filename)
        iteration_count = int(lastrow[0])
        params = [float(x) for x in lastrow[1:]]
    else:
        iteration_count = 0 
        params = 2*np.pi*np.random.uniform(0, 1, optimizer_option.num_params)
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['iteration_count'] + [f'param_{p}' for p in range(len(params))])
    
    return train_with_scipy(params, optimizer_option.circuit_cost_function, optimizer_option.max_iter)
