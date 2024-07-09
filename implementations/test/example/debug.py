import pennylane as qml
from pennylane import numpy as np

# Define the phase gate


# Define the number of qubits and control qubits
num_qubits = 4
control_qubits = [0, 1, 2]
target_qubit = 3

# Create a circuit
dev = qml.device('default.qubit', wires=num_qubits)

@qml.qnode(dev)
def circuit(t):
    phase_shift = np.array([[1, 0], [0, np.exp(1j *2* np.pi )]])
    print(qml.math.shape(phase_shift))
    # phase_shift_new = np.array([[1, 0], [0, np.exp(1j * np.pi * t)]])
    # Apply the triple-controlled phase gate
    # U = np.array([[1,0],[0,np.exp(-1j*2*np.pi*t)]])
    qml.ControlledQutritUnitary(phase_shift, control_wires=control_qubits, wires= target_qubit)
    # qml.Barrier()
    # qml.PauliX(0)
    # U = np.array([[1,0],[0,np.exp(1j*2*np.pi*t)]])
    # qml.ControlledQutritUnitary(U, control_wires=control_qubits, wires= target_qubit)
    # qml.PauliX(0)
    # phase_shift_inv = np.array([[1, 0], [0, np.exp(-1j * 2*np.pi * t)]])
    # qml.ControlledQubitUnitary(phase_shift, control_wires=control_qubits, wires=target_qubit)
    # qml.PauliX(0)
    # qml.ControlledQubitUnitary(phase_shift_inv, control_wires=control_qubits, wires=target_qubit)
    return qml.state()

# Execute the circuit
print(circuit(0.1))