from qiskit import QuantumCircuit
import pennylane as qml
from pennylane import numpy as np
def iter_apply(bitstring,i,num_qubits,qubit_indexes):
    flip = bitstring[0]==0
    if flip:
        bitstring = [not(j) for j in bitstring]
    if bitstring[1]:
        qml.PauliX(qubit_indexes[num_qubits-i-1])
    qml.CNOT(wires=[qubit_indexes[num_qubits - i],qubit_indexes[num_qubits-(i+1)]])
    next_bitstring = bitstring[1:]
    if len(next_bitstring)>1:
        return iter_apply(next_bitstring,i+1,num_qubits,qubit_indexes)
    else:
        return None

def apply_convert(bitstring,qubit_indexes):
    num_qubits = len(bitstring)
    qml.PauliX(qubit_indexes[num_qubits - 1])
    qml.Hadamard(qubit_indexes[num_qubits - 1])
    iter_apply(bitstring,1,num_qubits,qubit_indexes)

def reversed_iter_apply(bitstring,i,num_qubits,qubit_indexes):
    flip = bitstring[0]==0
    if flip:
        bitstring = [not(j) for j in bitstring]
    if len(bitstring)>1:
        reversed_iter_apply(bitstring[1:],i+1,num_qubits,qubit_indexes)
    else:
        return None
    qml.CNOT(wires=[qubit_indexes[num_qubits - i],qubit_indexes[num_qubits-(i+1)]])
    if bitstring[1]:
        qml.PauliX(qubit_indexes[num_qubits-i-1])
    return None

def reverse_apply_convert(bitstring,qubit_indexes):
    ## 1111
    num_qubits = len(bitstring)
    reversed_iter_apply(bitstring,1,num_qubits,qubit_indexes)
    qml.Hadamard(qubit_indexes[num_qubits - 1])
    qml.PauliX(qubit_indexes[num_qubits - 1])


def get_driver_component(num_qubits:int,t:float,bitstring:list,qubit_indexes:list):
    """ the drving hamitonian simulation for the bitstring like [0,1,0]

    Args:
        num_qubits (int):  the number of total qubits
        t (float): the paramter  t
        bitstring (list):  the bitstring of v or w
        qubit_indexes (list): the indexes of the qubits which is applied.
    """
    bitstring = [(i+1)//2 for i in bitstring]
    reverse_apply_convert(bitstring,qubit_indexes)
    qml.Barrier()
    ## multi control phase gate
    phase_shift = np.array([[1,0],[0,np.exp(-1j*t)]])
    qml.ControlledQubitUnitary(phase_shift, control_wires=qubit_indexes[1:], wires=qubit_indexes[0])
    qml.Barrier()
    qml.PauliX(qubit_indexes[-1])
    phase_shift_inv = np.array([[1,0],[0,np.exp(1j*t)]])
    qml.ControlledQubitUnitary(phase_shift_inv, control_wires=qubit_indexes[1:], wires=qubit_indexes[0])
    qml.PauliX(qubit_indexes[-1])
    qml.Barrier()
    # qc.mcp(-2*np.pi * t, list(range(num_qubits-1)),num_qubits-1)
    apply_convert(bitstring,qubit_indexes)

