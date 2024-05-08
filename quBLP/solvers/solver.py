from .circuits import pennylaneCircuit
from .optimizer import train_gradient

def solve(variables,objective,driver_bitstrs,feasiable_state,max_iter=50,learning_rate=0.1,num_layers=3):
    """solver for the problem
    Args:
        variables (List[str]): variables of the problem
        objective (callable): the objective function
        driver_bitstrs (List): the bit strings for driver hamiltonian
        feasiable_state (List): the feasible state
    """
    num_qubits = len(variables)
    circuit = pennylaneCircuit(num_qubits,num_layers=num_layers)
    ## test
    cost_func = circuit.build_unitary_circuit(feasiable_state,driver_bitstrs,objective)
    test_res = circuit.inference([0.1]*num_layers)
    print(f'test_res: {test_res}')
    print(circuit.inference_circuit)
    best_params = train_gradient(num_layers,cost_func,max_iter,learning_rate)
    best_solution = circuit.inference(best_params)
    cost = objective(best_solution)
    print(f"best solution: {best_solution}, cost: {cost}")
    return best_solution,cost


    
