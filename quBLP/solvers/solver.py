from .circuits import pennylaneCircuit
from .optimizer import train_gradient

def solve(max_iter, learning_rate, variables, num_layers, objective, feasiable_state, optimization_method):
    """solver for the problem
    Args:
        variables (List[str]): variables of the problem
        objective (callable): the objective function
        feasiable_state (List): the feasible state
        optimization_method (List): [type, other_params_list]
    """
    num_qubits = len(variables)
    circuit = pennylaneCircuit(num_qubits, num_layers, objective, feasiable_state, optimization_method)
    ## test
    if optimization_method[0] == 'penalty':
        num_params = num_layers * 2
    elif optimization_method[0] == 'commute':
        num_params = num_layers
    cost_func = circuit.create_circuit()
    test_res = circuit.inference([0.5]*num_params)
    print(f'test_res: {test_res}') #-
    print(circuit.inference_circuit) #-
    best_params = train_gradient(num_params,cost_func,max_iter,learning_rate)
    print(f"best_params: {best_params}") #-
    best_solution = circuit.inference(best_params)
    cost = objective(best_solution)
    print(f"best solution: {best_solution}, cost: {cost}") #-
    return best_solution,cost


    
