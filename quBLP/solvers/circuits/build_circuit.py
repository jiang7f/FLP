import pennylane as qml
from qiskit import QuantumCircuit
import numpy as np
import numpy.linalg as ls
from scipy.linalg import expm
from .penneylane_decompose import get_driver_component as get_driver_component_pennylane

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

# num_qubits, num_layers, objective, feasiable_state, optimization_method
class pennylaneCircuit:
    def __init__(self, num_qubits, num_layers, objective, feasiable_state, optimization_method, optimization_direction):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.objective = objective
        self.feasiable_state = feasiable_state
        self.optimization_method = optimization_method
        assert optimization_direction in ['min', 'max']
        self.Ho_dir =  1 if optimization_direction == 'max' else -1

    def create_circuit(self, is_decompose=False):
        num_qubits = self.num_qubits
        dev = qml.device("default.qubit", wires=num_qubits)
        def add_in_target(num_qubits, target_qubit, gate):
            H = np.eye(2 ** (target_qubit))
            H = np.kron(H, gate)
            H = np.kron(H, np.eye(2 ** (num_qubits - 1 - target_qubit)))
            return H
        @qml.qnode(dev)
        def circuit(params):
            num_qubits = self.num_qubits
            num_layers = self.num_layers
            optimization_method = self.optimization_method[0] 
            Ho_vector, Ho_matrix = self.optimization_method[1]
            Ho_params = params[:num_layers]
            Hd_params = params[num_layers:]
            assert len(Hd_params) == num_layers
            if optimization_method == 'penalty':
                pnt_lbd = self.optimization_method[2]
                constraints = self.optimization_method[3]
                for i in range(num_qubits):
                    # if self.op_dir == 'min':
                    #     qml.PauliX(i)
                    qml.Hadamard(i)
                    pass
                for layer in range(num_layers):
                    if is_decompose:
                        pass
                    else:
                        #$ 目标函数
                        Ho = np.zeros((2**num_qubits, 2**num_qubits))
                        for index, Ho_vi in enumerate(Ho_vector):
                            Ho += Ho_vi * add_in_target(num_qubits, index, (np.eye(2) - np.array([[1, 0],[0, -1]]))/2)
                        for index, Ho_mi in enumerate(Ho_matrix):
                            Ho += Ho_mi
                        # 惩罚项 Hamiltonian penalty
                        for penalty_mi in constraints:
                            H_pnt = np.zeros((2**num_qubits, 2**num_qubits))
                            for index, penalty_vi in enumerate(penalty_mi[:-1]):
                                H_pnt += penalty_vi * add_in_target(num_qubits, index, (np.eye(2) - np.array([[1, 0],[0, -1]]))/2)
                            H_pnt -= penalty_mi[-1] * np.eye(2**num_qubits)
                            Ho += pnt_lbd * H_pnt @ H_pnt
                        # Ho取负，对应绝热演化能级问题，但因为训练参数，可能没区别
                        qml.QubitUnitary(expm(-1j * Ho_params[layer] * self.Ho_dir * Ho), wires=range(num_qubits))
                    # Rx 驱动哈密顿量
                    for i in range(num_qubits):
                        qml.RX(Hd_params[layer],i)
            elif optimization_method == 'commute':
                Hd_bitsList = self.optimization_method[2]
                for i in np.nonzero(self.feasiable_state)[0]:
                    qml.PauliX(i)
                for layer in range(num_layers):
                    #$ 目标函数
                    Ho = np.zeros((2**num_qubits, 2**num_qubits))
                    for index, Ho_vi in enumerate(Ho_vector):
                        Ho += Ho_vi * add_in_target(num_qubits, index, (np.eye(2) - np.array([[1, 0],[0, -1]]))/2)
                    for index, Ho_mi in enumerate(Ho_matrix):
                        Ho += Ho_mi
                    qml.QubitUnitary(expm(-1j * Ho_params[layer] * self.Ho_dir * Ho), wires=range(num_qubits))
                    # 惩罚约束
                    for bitstrings in range(len(Hd_bitsList)):
                        hd_bits = Hd_bitsList[bitstrings]
                        nonzero_indices = np.nonzero(hd_bits)[0]
                        nonzerobits = hd_bits[nonzero_indices]
                        if is_decompose:
                            pass
                            # get_driver_component_pennylane(num_qubits, Hd_params[layer], nonzerobits, nonzero_indices)
                        else:
                            qml.QubitUnitary(expm(-1j * Hd_params[layer] * plus_minus_gate_sequence_to_unitary(nonzerobits)), wires=nonzero_indices)
            return qml.probs(wires=range(num_qubits))
        self.inference_circuit = circuit

        def costfunc(params):
            bitstrs = circuit(params)
            bitstrsindex = np.nonzero(bitstrs)[0]
            probs = bitstrs[bitstrsindex]
            variablevalues = [[int(j) for j in list(bin(i)[2:].zfill(self.num_qubits))] for i in bitstrsindex]
            costs = 0
            for value, prob in zip(variablevalues, probs):
                costs += self.objective(value) * prob
            return costs
        return costfunc
    
    def draw_circuit(self) -> None:
        from pennylane.drawer import draw
        circuit_drawer = draw(self.inference_circuit)
        print(circuit_drawer(np.zeros(self.num_layers * 2)))

    def inference(self,params):
        qml_probs = self.inference_circuit(params)
        bitstrsindex = np.nonzero(qml_probs)[0]
        probs = qml_probs[bitstrsindex]
        collapse_state = [[int(j) for j in list(bin(i)[2:].zfill(self.num_qubits))] for i in bitstrsindex]
        return collapse_state, probs




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
            for layer in range(depth):
                for gate_hdi, ind in zip(gate_hds, nonzero_indices):
                    qc.unitary(expm(-1j * gamma[layer] * gate_hdi), (num_qubits - 1 - i for i in ind))
        for layer in range(depth):
            qc.unitary(expm(-1j * beta[layer] * Hp), range(num_qubits))
            for gate_hdi, ind in zip(gate_hds, nonzero_indices):
                qc.unitary(expm(-1j * gamma[layer] * gate_hdi), (num_qubits - 1 - i for i in ind))
        qc.measure_all()
        return qc
    
    def decompose_circuit(self, qc):
        qc = qc.decompose()
        return qc
