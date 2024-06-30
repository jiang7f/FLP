def mcx_nancillary_linear_depth(circuit,control_qubits, target_qubit, ancillary_qubits):
    """
    This function implements the multi-controlled-X gate using the Toffoli gate.
    """
    if len(control_qubits) == 0:
        circuit.x(target_qubit)
    elif len(control_qubits) == 1:
        circuit.cx(control_qubits[0], target_qubit)
    elif len(control_qubits) == 2:
        circuit.ccx(control_qubits[0], control_qubits[1], target_qubit)
    else:
        circuit.ccx(control_qubits[0], control_qubits[1], ancillary_qubits[0])
        for i in range(len(control_qubits)-2):
            circuit.ccx(ancillary_qubits[i], control_qubits[i+2], ancillary_qubits[i+1])
        circuit.ccx(ancillary_qubits[len(control_qubits)-2], control_qubits[-1], target_qubit)
        # reverse the process
        for i in list(range(len(control_qubits)-2))[::-1]:
            circuit.ccx(ancillary_qubits[i], control_qubits[i+2], ancillary_qubits[i+1])
        circuit.ccx(control_qubits[0], control_qubits[1], ancillary_qubits[0])



def mcx_nancillary_log_depth(circuit,control_qubits, target_qubit, ancillary_qubits):
    """
    This function implements the multi-controlled-X gate using the Toffoli gate.
    """
    if len(control_qubits) == 0:
        circuit.x(target_qubit)
    elif len(control_qubits) == 1:
        circuit.cx(control_qubits[0], target_qubit)
        return 0
    elif len(control_qubits) == 2: 
        circuit.ccx(control_qubits[0], control_qubits[1], target_qubit)
        return 1
    else:
        circuit.ccx(control_qubits[0], control_qubits[1], ancillary_qubits[0])
        res =  mcx_nancillary_log_depth(circuit, control_qubits[2:]+[ancillary_qubits[0]], target_qubit, ancillary_qubits[1:])
        circuit.ccx(control_qubits[0], control_qubits[1], ancillary_qubits[0])
        return res

        
if __name__ == '__main__':
    from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister,transpile
    log_depth_list = []
    linear_depth_list = []
    qubit_list = list(range(4,100,5))
    for i in qubit_list:
        circuit1 = QuantumCircuit(i+1+i-2)
        mcx_nancillary_log_depth(circuit1, list(range(i)), i, list(range(i+1,i+1+i-2)))
        transpile_cir1 = transpile(circuit1, basis_gates=['ecr', 'id', 'rz','sx','x'], optimization_level=3)
        log_depth_list.append(transpile_cir1.depth())
        print(i, transpile_cir1.num_nonlocal_gates())
        circuit2 = QuantumCircuit(i+1+i-1)
        mcx_nancillary_linear_depth(circuit2, list(range(i)), i, list(range(i+1,i+1+i-1)))
        transpile_cir2 = transpile(circuit2, basis_gates=['ecr', 'id', 'rz','sx','x'], optimization_level=3)
        linear_depth_list.append(transpile_cir2.depth())
        
    print(qubit_list)
    print(log_depth_list)
    print(linear_depth_list)
    ## plot the results
    import matplotlib.pyplot as plt
    plt.plot(qubit_list, log_depth_list, label='log depth')
    plt.plot(qubit_list, linear_depth_list, label='linear depth')
    plt.xlabel('number of qubits')
    plt.ylabel('circuit depth')
    plt.legend()
    # plt.show()
    plt.savefig('depth_vs_qubit.png')

    
