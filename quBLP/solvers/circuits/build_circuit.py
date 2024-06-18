import pennylane as qml
from qiskit import QuantumCircuit
from ...utils.quantum_lib import *
import numpy as np
import numpy.linalg as ls
from scipy.linalg import expm
from .penneylane_decompose import get_driver_component as get_driver_component_pennylane
from ...models import CircuitOption

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

class PennylaneCircuit:
    def __init__(self, circuit_option: CircuitOption):
        self.circuit_option = circuit_option
        assert circuit_option.optimization_direction in ['min', 'max']
        self.Ho_dir =  1 if circuit_option.optimization_direction == 'max' else -1
        if circuit_option.algorithm_optimization_method == 'cyclic':
            self.cyclic_qubit_set = {item for sublist in self.circuit_option.constraints_for_cyclic for item in np.nonzero(sublist[:-1])[0]}
            self.others_qubit_set = set(range(circuit_option.num_qubits)) - self.cyclic_qubit_set
        self.num_qubits = circuit_option.num_qubits
        self.num_layers = circuit_option.num_layers

    # 先给出不同电路的函数，变量初始化在此之后
    def create_circuit(self):
        is_decompose = self.circuit_option.is_decompose
        by_Ho_gate_list = self.circuit_option.by_Ho_gate_list
        Ho_gate_list = self.circuit_option.Ho_gate_list
        Ho_vector = self.circuit_option.linear_objective_vector
        Ho_matrix = self.circuit_option.nonlinear_objective_matrix
        self.inference_circuit = None
        num_qubits = self.num_qubits
        num_layers = self.num_layers
        algorithm_optimization_method = self.circuit_option.algorithm_optimization_method
        dev = qml.device("default.qubit", wires=num_qubits)
        def add_in_target(num_qubits, target_qubit, gate):
            H = np.eye(2 ** (target_qubit))
            H = np.kron(H, gate)
            H = np.kron(H, np.eye(2 ** (num_qubits - 1 - target_qubit)))
            return H
        def gnrt_cyclic_Hd(n):
            Hd = np.zeros((2**n, 2**n)).astype(np.complex128)
            for i in range(n):
                j = (i + 1) % n
                Hd += (add_in_target(n, i, gate_x) @ add_in_target(n, j, gate_x) + add_in_target(n, i, gate_y) @ add_in_target(n, j, gate_y))
            return -Hd 
        
        @qml.qnode(dev)
        def circuit_penalty(params):
            Ho_params = params[:num_layers]
            Hd_params = params[num_layers:]
            assert len(Hd_params) == num_layers
            pnt_lbd = self.circuit_option.penalty_lambda
            constraints = self.circuit_option.constraints
            for i in range(num_qubits):
                # if self.op_dir == 'min':
                #     qml.PauliX(i)
                qml.Hadamard(i)
            for layer in range(num_layers):
                if is_decompose == True:
                    pass
                elif is_decompose == False:
                    if by_Ho_gate_list == True:
                        pass
                    elif by_Ho_gate_list == False:
                        #$ 目标函数
                        Ho = np.zeros((2**num_qubits, 2**num_qubits))
                        for index, Ho_vi in enumerate(Ho_vector):
                            Ho += Ho_vi * add_in_target(num_qubits, index, (gate_I - gate_z)/2)
                        for index, Ho_mi in enumerate(Ho_matrix):
                            Ho += Ho_mi
                        # 惩罚项 Hamiltonian penalty
                        for penalty_mi in constraints:
                            H_pnt = np.zeros((2**num_qubits, 2**num_qubits))
                            for index, penalty_vi in enumerate(penalty_mi[:-1]):
                                H_pnt += penalty_vi * add_in_target(num_qubits, index, (gate_I - gate_z)/2)
                            H_pnt -= penalty_mi[-1] * np.eye(2**num_qubits)
                            Ho += pnt_lbd * H_pnt @ H_pnt
                        # Ho取负，对应绝热演化能级问题，但因为训练参数，可能没区别
                        qml.QubitUnitary(expm(-1j * Ho_params[layer] * self.Ho_dir * Ho), wires=range(num_qubits))
                # Rx 驱动哈密顿量
                for i in range(num_qubits):
                    qml.RX(Hd_params[layer],i)
            return qml.probs(wires=range(num_qubits))
        
        @qml.qnode(dev)
        def circuit_cyclic(params):
            Ho_params = params[:num_layers]
            Hd_params = params[num_layers:]
            assert len(Hd_params) == num_layers
            pnt_lbd = self.circuit_option.penalty_lambda
            constraints_for_cyclic = self.circuit_option.constraints_for_cyclic
            constraints_for_others = self.circuit_option.constraints_for_others
            # 找到需要用cyclic的量子位设定可行解，其余照常用H门
            cyclic_qubit_set = self.cyclic_qubit_set
            others_qubit_set = self.others_qubit_set
            for i in set(np.nonzero(self.circuit_option.feasiable_state)[0]).intersection(cyclic_qubit_set):
                qml.PauliX(i)
            for i in others_qubit_set:
                qml.Hadamard(i)                    
            for layer in range(num_layers):
                if is_decompose:
                    pass
                else:
                    if by_Ho_gate_list == True:
                        pass
                    elif by_Ho_gate_list == False:
                        #$ 目标函数
                        Ho = np.zeros((2**num_qubits, 2**num_qubits))
                        for index, Ho_vi in enumerate(Ho_vector):
                            Ho += Ho_vi * add_in_target(num_qubits, index, (gate_I - gate_z)/2)
                        for index, Ho_mi in enumerate(Ho_matrix):
                            Ho += Ho_mi
                        # constraints_for_others 惩罚项
                        for penalty_mi in constraints_for_others:
                            H_pnt = np.zeros((2**num_qubits, 2**num_qubits))
                            for index, penalty_vi in enumerate(penalty_mi[:-1]):
                                H_pnt += penalty_vi * add_in_target(num_qubits, index, (gate_I - gate_z)/2)
                            H_pnt -= penalty_mi[-1] * np.eye(2**num_qubits)
                            Ho += pnt_lbd * H_pnt @ H_pnt
                        # Ho取负，对应绝热演化能级问题，但因为训练参数，可能没区别
                        qml.QubitUnitary(expm(-1j * Ho_params[layer] * self.Ho_dir * Ho), wires=range(num_qubits))
                # constraints_for_cyclic --> 驱动哈密顿量
                for cyclic_mi in constraints_for_cyclic:
                    nzlist = np.nonzero(cyclic_mi[:-1])[0]
                    cyclic_Hd_i = gnrt_cyclic_Hd(len(nzlist))
                    qml.QubitUnitary(expm(-1j * Hd_params[layer] * cyclic_Hd_i), wires=nzlist)
                for i in others_qubit_set:
                    qml.RX(Hd_params[layer],i)
            return qml.probs(wires=range(num_qubits))
        
        @qml.qnode(dev)
        def circuit_commute(params):
            Ho_params = params[:num_layers]
            Hd_params = params[num_layers:]
            assert len(Hd_params) == num_layers
            Hd_bits_list = self.circuit_option.Hd_bits_list
            for i in np.nonzero(self.circuit_option.feasiable_state)[0]:
                qml.PauliX(i)
            for layer in range(num_layers):
                if by_Ho_gate_list == True:
                    pass
                elif by_Ho_gate_list == False:
                    #$ 目标函数
                    Ho = np.zeros((2**num_qubits, 2**num_qubits))
                    for index, Ho_vi in enumerate(Ho_vector):
                        Ho += Ho_vi * add_in_target(num_qubits, index, (gate_I - gate_z)/2)
                    for index, Ho_mi in enumerate(Ho_matrix):
                        Ho += Ho_mi
                    qml.QubitUnitary(expm(-1j * Ho_params[layer] * self.Ho_dir * Ho), wires=range(num_qubits))
                # 惩罚约束
                for bitstrings in range(len(Hd_bits_list)):
                    hd_bits = Hd_bits_list[bitstrings]
                    nonzero_indices = np.nonzero(hd_bits)[0]
                    nonzerobits = hd_bits[nonzero_indices]
                    if is_decompose:
                        pass
                        # get_driver_component_pennylane(num_qubits, Hd_params[layer], nonzerobits, nonzero_indices)
                    else:
                        qml.QubitUnitary(expm(-1j * Hd_params[layer] * plus_minus_gate_sequence_to_unitary(nonzerobits)), wires=nonzero_indices)
            return qml.probs(wires=range(num_qubits))
        
        @qml.qnode(dev)
        def circuit_HEA(params):
            params = np.array(params).reshape((num_layers, num_qubits, 3))
            r1_params = params[:, :, 0]
            r2_params = params[:, :, 1]
            r3_params = params[:, :, 2]
            assert len(r3_params) == num_layers
            for layer in range(num_layers):
                for i in range(num_qubits):
                    qml.RZ(r1_params[layer][i], wires=i)
                    qml.RY(r2_params[layer][i], wires=i)
                    qml.RZ(r3_params[layer][i], wires=i)
                for i in range(num_qubits):
                    qml.CNOT(wires=[i, (i + 1) % num_qubits])
            return qml.probs(wires=range(num_qubits))
        
        circuit_map = {
            'penalty': circuit_penalty,
            'cyclic': circuit_cyclic,
            'commute': circuit_commute,
            'HEA': circuit_HEA
        }
        self.inference_circuit = circuit_map.get(algorithm_optimization_method)

        def costfunc(params):
            bitstrs = self.inference_circuit(params)
            bitstrsindex = np.nonzero(bitstrs)[0]
            probs = bitstrs[bitstrsindex]
            variablevalues = [[int(j) for j in list(bin(i)[2:].zfill(self.num_qubits))] for i in bitstrsindex]
            costs = 0
            for value, prob in zip(variablevalues, probs):
                costs += self.circuit_option.objective(value) * prob
            return costs
        return costfunc
    
    def draw_circuit(self) -> None:
        from pennylane.drawer import draw
        circuit_drawer = draw(self.inference_circuit)
        if self.circuit_option.algorithm_optimization_method == 'HEA':
            print(circuit_drawer(np.zeros(self.num_layers * self.num_qubits * 3)))
        else:
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
