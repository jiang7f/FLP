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

def adam_optimizer(params, cost_function, max_iter, learning_rate, beta1, beta2, num_consecutive_iter = 5, early_stopping_threshold=0.0001):
    eps = 1e-8
    m = np.zeros(len(params))
    v = np.zeros(len(params))
    best_params = params
    best_cost = cost_function(params)
    prev_cost = best_cost  # 跟踪先前成本
    consecutive_no_improvement = 0  # 没有明显提升的连续迭代次数
    costs_list = []
    with tqdm(total=max_iter) as pbar:
        for i in range(max_iter):
            gradients = gradient_by_param_shift(params, cost_function)
            m = beta1 * m + (1 - beta1) * gradients
            v = beta2 * v + (1 - beta2) * gradients ** 2
            m_hat = m / (1 - beta1 ** (i + 1))
            v_hat = v / (1 - beta2 ** (i + 1))
            params -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)
            cost = cost_function(params)
            costs_list.append(float(cost))
            pbar.set_postfix(cost=cost)
            pbar.update(1)
            if cost < best_cost:
                best_cost = cost
                best_params = params
            if abs(prev_cost - cost) < early_stopping_threshold:
                consecutive_no_improvement += 1
                if consecutive_no_improvement >= num_consecutive_iter:  # consecutive iterations
                    print("Early stopping: Loss change is below threshold.")
                    break
            else:
                consecutive_no_improvement = 0
                prev_cost = cost
    #+
    print('====')
    print(costs_list)
    import matplotlib.pyplot as plt 
    plt.figure(figsize=(10, 5))  
    plt.plot(range(len(costs_list)), costs_list, marker='o')  
    plt.xlabel('Iteration')  
    plt.ylabel('Cost')  
    plt.grid(True)  
    plt.show()
    #+
    return best_params

def train_gradient(num_params, cost_function, max_iter, learning_rate, beta1, beta2):
    params = 2*np.pi*np.random.uniform(0, 1, num_params, requires_grad=True)
    return adam_optimizer(params, cost_function, max_iter, learning_rate, beta1, beta2)