from .circuits import PennylaneCircuit, QiskitCircuit
from .optimizer import train_gradient, train_non_gradient
from ..models import OptimizerOption, CircuitOption
import numpy as np
def solve(optimizer_option: OptimizerOption, circuit_option: CircuitOption):
    print(f'algorithm_optimization_method: {circuit_option.algorithm_optimization_method}') #+
    if circuit_option.circuit_type == 'pennylane':
        circuit = PennylaneCircuit(circuit_option)
    elif circuit_option.circuit_type == 'qiskit':
        circuit = QiskitCircuit(circuit_option)

    if circuit_option.algorithm_optimization_method == 'HEA':
        num_params = circuit_option.num_layers * circuit_option.num_qubits * 3
    else:
        num_params = circuit_option.num_layers * 2

    optimizer_option.num_params = num_params
    circuit.create_circuit()
    optimizer_option.cost_function = circuit.get_costfunc()
    print(optimizer_option.cost_function)
    if circuit_option.need_draw:
        circuit.draw_circuit()
    # 测试一组预设参数的结果
    collapse_state, probs = circuit.inference([0.5] * num_params)
    test_maxprobidex = np.argmax(probs)
    print(f'test_max_prob: {probs[test_maxprobidex]:.2%}, test_max_prob_state: {collapse_state[test_maxprobidex]}') #-
    # 进行参数优化
    if optimizer_option.params_optimization_method == 'Adam':
        best_params = train_gradient(optimizer_option)
    elif optimizer_option.params_optimization_method == 'COBYLA':
        best_params = train_non_gradient(optimizer_option)
    collapse_state, probs = circuit.inference(best_params)
    print(f"best_params: {best_params}") #-
    return collapse_state, probs


    
