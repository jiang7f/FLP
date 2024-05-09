import pennylane as qml
from pennylane import numpy as np
from tqdm import tqdm
def gradient_by_param_shift_pauli(params, cost_function):
    num_params = len(params)
    gradients = np.zeros(num_params)
    shift = np.pi / 2
    for i in range(num_params):
        shifted_params = params.copy()
        shifted_params[i] += shift
        forward = cost_function(shifted_params)
        shifted_params[i] -= 2 * shift
        backward = cost_function(shifted_params)
        gradients[i] = 0.5 * (forward - backward)
    return gradients

def gradient_by_param_shift(params, cost_function):
    num_params = len(params)
    gradients = np.zeros(num_params)
    shift = 0.01
    for i in range(num_params):
        shifted_params = params.copy()
        shifted_params[i] += shift
        forward = cost_function(shifted_params)
        shifted_params[i] -= 2 * shift
        backward = cost_function(shifted_params)
        gradients[i] = (forward - backward)/(2*shift)
    return gradients

def adam_optimizer(params, cost_function, num_iter, learning_rate):
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    m = np.zeros(len(params))
    v = np.zeros(len(params))
    best_params = params
    best_cost = cost_function(params)
    with tqdm(total=num_iter) as pbar:
        for i in range(num_iter):
            gradients = gradient_by_param_shift(params, cost_function)
            m = beta1 * m + (1 - beta1) * gradients
            v = beta2 * v + (1 - beta2) * gradients ** 2
            m_hat = m / (1 - beta1 ** (i + 1))
            v_hat = v / (1 - beta2 ** (i + 1))
            params -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)
            cost = cost_function(params)
            pbar.set_postfix(cost=cost)
            pbar.update(1)
            if cost < best_cost:
                best_cost = cost
                best_params = params
            # print(f'---- {best_params}') #-
    return best_params

def train_gradient(num_params, cost_function, num_iter, learning_rate):
    params = 2*np.pi*np.random.uniform(0, 1, num_params, requires_grad=True)
    return adam_optimizer(params, cost_function, num_iter, learning_rate)