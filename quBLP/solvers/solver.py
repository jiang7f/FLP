from .circuits import PennylaneCircuit
from .optimizer import train_gradient, train_non_gradient
import numpy as np
def solve(params_optimization_method, max_iter, learning_rate, variables, num_layers, objective, feasiable_state, optimization_method, optimization_direction, need_draw):
    """solver for the problem
    Args:
        variables (List[str]): variables of the problem
        objective (callable): the objective function
        feasiable_state (List): the feasible state
        optimization_method (List): [type, other_params_list]
    """
    print(f'params_optimization_method: {params_optimization_method}') #+
    num_qubits = len(variables)
    circuit = PennylaneCircuit(num_qubits, num_layers, objective, feasiable_state, optimization_method, optimization_direction)
    if optimization_method[0] == 'HEA':
        num_params = num_layers * num_qubits * 3
    else:
        num_params = num_layers * 2
    cost_func = circuit.create_circuit()
    if need_draw:
        circuit.draw_circuit()
    # 测试一组预设参数的结果
    collapse_state, probs = circuit.inference([0.5]*num_params)
    test_maxprobidex = np.argmax(probs)
    print(f'test_max_prob: {probs[test_maxprobidex]:.2%}, test_max_prob_state: {collapse_state[test_maxprobidex]}') #-
    # 进行参数优化
    print(circuit.inference_circuit) #-
    if params_optimization_method == 'adam':
        best_params = train_gradient(num_params,cost_func,max_iter,learning_rate)
    elif params_optimization_method == 'COBYLA':
        best_params = train_non_gradient(num_params, cost_func, max_iter)
    collapse_state, probs = circuit.inference(best_params)
    print(f"best_params: {best_params}") #-
    return collapse_state, probs


    
