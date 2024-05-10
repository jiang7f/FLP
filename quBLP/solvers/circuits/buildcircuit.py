import pennylane as qml
from qiskit import QuantumCircuit
import numpy as np
import numpy.linalg as ls
from scipy.linalg import expm
from .penneylanedecompose import get_driver_component as get_driver_component_pennylane

def plus_minus_gate_sequence_to_unitary(s):
    # 把非0元素映射成01
    filtered_arr = [0 if x == -1 else 1 for x in s if x != 0]
    binary_str = ''.join(map(str, filtered_arr))
    # 取到二进制表示的数
    map_num = int(binary_str, 2)
    length = len(s)
    scale = 2**length
    matrix = np.zeros((scale, scale))
    matrix[map_num, scale - 1 - map_num] = 1
    map_num = scale - 1 - map_num
    matrix[map_num, scale - 1 - map_num] = 1
    return matrix

class pennylaneCircuit:
    def __init__(self, num_qubits,num_layers):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
    
    def build_unitary_circuit(self, feasiable_state, hd_bitsList, objective_function, Hp=None):
        return self._create_circuit(feasiable_state, hd_bitsList, objective_function, Hp, is_decompose=False)

    def decompose_circuit(self, feasiable_state, hd_bitsList, objective_function, Hp=None):
        return self._create_circuit(feasiable_state, hd_bitsList, objective_function, Hp, is_decompose=True)

    def _create_circuit(self,feasiable_state,hd_bitsList,objective_function,Hp=None, is_decompose=False):
        num_qubits = self.num_qubits
        dev = qml.device("default.qubit", wires=num_qubits)
        @qml.qnode(dev)
        def circuit(params):
            for i in np.nonzero(feasiable_state)[0]:
                qml.PauliX(i)
            depth  = self.num_layers
            if Hp:
                hpparams = params[:depth]
                hdparams = params[depth:]
                assert len(hdparams) == depth
            else:
                hdparams = params
                assert len(hdparams) == depth
            for dp in range(depth):
                # apply_problem_hamitonian_pennylane(num_qubits,hpparams[dp])
                for bitstrings in range(len(hd_bitsList)):
                    hd_bits = hd_bitsList[bitstrings]
                    nonzero_indices = np.nonzero(hd_bits)[0]
                    nonzerobits = hd_bits[nonzero_indices]
                    if is_decompose:
                        get_driver_component_pennylane(num_qubits, hdparams[dp], nonzerobits, nonzero_indices)
                    else:
                        qml.QubitUnitary(expm(-1j * hdparams[dp] * plus_minus_gate_sequence_to_unitary(nonzerobits)), wires=nonzero_indices)
            return qml.probs(wires=range(num_qubits))
        self.inference_circuit = circuit
        def costfunc(params):
            bitstrs = circuit(params)
            bitstrsindex = np.nonzero(bitstrs)[0]
            probs = bitstrs[bitstrsindex]
            variablevalues = [[int(j) for j in list(bin(i)[2:].zfill(self.num_qubits))] for i in bitstrsindex]
            costs = 0
            for value, prob in zip(variablevalues, probs):
                costs += objective_function(value) * prob
            return costs
        return costfunc

    def inference(self,params):
        bitstrs = self.inference_circuit(params)
        bitstrsindex = np.nonzero(bitstrs)[0]
        probs = bitstrs[bitstrsindex]
        maxprobidex = np.argmax(probs)
        collapse_variable_values = [[int(j) for j in list(bin(i)[2:].zfill(self.num_qubits))] for i in bitstrsindex]
        print(f'max_prob: {probs[maxprobidex]:.2%}') #-
        print(f'collapse_variable_values: {collapse_variable_values}') #-
        return collapse_variable_values[maxprobidex]

    



    

   
    
    
class QiskitCircuit:
    def __init__(self, n, p):
        self.n = n
        self.p = p
        self.num_qubits = n * p
    def gnrt_hdi(u):
        hds = []
        for urow in u:
            hds.append(plus_minus_gate_sequence_to_unitary(urow))
        return hds
    def build_circuit(self,ini_states,hpparms,hdparms,gate_hds,nonzero_indices,Hp=None):
        # Initialize the circuit
        num_qubits = self.num_qubits
        qc = QuantumCircuit(num_qubits, num_qubits)
        from scipy.linalg import expm
        beta = hpparms
        gamma = hdparms
        depth = len(gamma)
        for i in ini_states:
            qc.x(i)
        if len(beta) ==0:
            for dp in range(depth):
                for gate_hdi, ind in zip(gate_hds, nonzero_indices):
                    qc.unitary(expm(-1j * gamma[dp] * gate_hdi), (num_qubits - 1 - i for i in ind))
        for dp in range(depth):
            qc.unitary(expm(-1j * beta[dp] * Hp), range(num_qubits))
            for gate_hdi, ind in zip(gate_hds, nonzero_indices):
                qc.unitary(expm(-1j * gamma[dp] * gate_hdi), (num_qubits - 1 - i for i in ind))
        qc.measure_all()
        return qc
    
    def decompose_circuit(self, qc):
        qc = qc.decompose()
        return qc
