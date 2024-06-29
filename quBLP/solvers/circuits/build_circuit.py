import pennylane as qml
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime.fake_provider import FakeKyoto,FakeKyiv,FakeSherbrooke,FakeQuebec,FakeAlmadenV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from ...utils.quantum_lib import *
from qiskit_aer import AerSimulator
import numpy as np
import numpy.linalg as ls
from collections.abc import Iterable
from scipy.linalg import expm
from .pennylane_decompose import driver_component as driver_component_pennylane
from .qiskit_decompose import driver_component as driver_component_qiskit
from ...models import CircuitOption
from qiskit_ibm_runtime import SamplerV2 as Sampler

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

    def inference(self, params):
        qml_probs = self.inference_circuit(params)
        bitstrsindex = np.nonzero(qml_probs)[0]
        probs = qml_probs[bitstrsindex]
        collapse_state = [[int(j) for j in list(bin(i)[2:].zfill(self.num_qubits))] for i in bitstrsindex]
        return collapse_state, probs
    
    # 先给出不同电路的函数，变量初始化在此之后
    def create_circuit(self):
        use_decompose = self.circuit_option.use_decompose
        use_Ho_gate_list = self.circuit_option.use_Ho_gate_list
        Ho_gate_list = self.circuit_option.Ho_gate_list
        Ho_vector = self.circuit_option.linear_objective_vector
        Ho_matrix = self.circuit_option.nonlinear_objective_matrix
        self.inference_circuit = None
        num_qubits = self.num_qubits
        num_layers = self.num_layers
        algorithm_optimization_method = self.circuit_option.algorithm_optimization_method
        
        dev = qml.device("default.qubit", wires=num_qubits + 1)
        
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
                
                if use_Ho_gate_list == True:
                    for i_list, theta in Ho_gate_list:
                        for i in range(len(i_list) - 1):
                            qml.CNOT(wires=[i_list[i], i_list[i + 1]])
                        qml.RZ(theta, wires=i_list[-1])
                        for i in range(len(i_list) - 2, -1, -1):
                            qml.CNOT(wires=[i_list[i], i_list[i + 1]])
                elif use_Ho_gate_list == False:
                    Ho = np.zeros((2**num_qubits, 2**num_qubits))
                    for index, Ho_vi in enumerate(Ho_vector):
                        Ho += Ho_vi * add_in_target(num_qubits, index, (gate_I - gate_z)/2)
                    for index, Ho_mi in enumerate(Ho_matrix):
                        Ho += Ho_mi

                # 惩罚项 Hamiltonian penalty
                if use_decompose:
                    for penalty_mi in constraints:
                        coeff = (np.sum(penalty_mi)-3*penalty_mi[-1])/2
                        for i in range(num_qubits):
                            qml.RZ(coeff*penalty_mi[i]*Ho_params[layer], i)
                        for i in range(num_qubits-1):
                            for j in range(i+1, num_qubits):
                                if penalty_mi[i] != 0 and penalty_mi[j] != 0:
                                    qml.CNOT(wires=[i, j])
                                    coeff = (penalty_mi[i]* penalty_mi[j])/2
                                    qml.RZ(coeff*Ho_params[layer], j)
                                    qml.CNOT(wires=[i, j])
                else:
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
                if use_Ho_gate_list == True:
                    for i_list, theta in Ho_gate_list:
                        for i in range(len(i_list) - 1):
                            qml.CNOT(wires=[i_list[i], i_list[i + 1]])
                        qml.RZ(theta, wires=i_list[-1])
                        for i in range(len(i_list) - 2, -1, -1):
                            qml.CNOT(wires=[i_list[i], i_list[i + 1]])
                elif use_Ho_gate_list == False:
                    Ho = np.zeros((2**num_qubits, 2**num_qubits))
                    for index, Ho_vi in enumerate(Ho_vector):
                        Ho += Ho_vi * add_in_target(num_qubits, index, (gate_I - gate_z)/2)
                    for index, Ho_mi in enumerate(Ho_matrix):
                        Ho += Ho_mi
                if use_decompose:
                    for penalty_mi in constraints_for_others:
                        coeff = (np.sum(penalty_mi)-3*penalty_mi[-1])/2
                        for i in range(num_qubits):
                            qml.RZ(coeff*Ho_params[layer], i)
                        for i in range(num_qubits-1):
                            for j in range(i+1, num_qubits):
                                if penalty_mi[i] != 0 and penalty_mi[j] != 0:
                                    qml.CNOT(wires=[i, j])
                                    coeff = (penalty_mi[i]* penalty_mi[j])/2
                                    qml.RZ(coeff*Ho_params[layer], j)
                                    qml.CNOT(wires=[i, j])
                else:
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
                if use_Ho_gate_list == True:
                    for i_list, theta in Ho_gate_list:
                        for i in range(len(i_list) - 1):
                            qml.CNOT(wires=[i_list[i], i_list[i + 1]])
                        qml.RZ(theta, wires=i_list[-1])
                        for i in range(len(i_list) - 2, -1, -1):
                            qml.CNOT(wires=[i_list[i], i_list[i + 1]])
                elif use_Ho_gate_list == False:
                    #$ 目标函数
                    Ho = np.zeros((2**num_qubits, 2**num_qubits))
                    for index, Ho_vi in enumerate(Ho_vector):
                        Ho += Ho_vi * add_in_target(num_qubits, index, (gate_I - gate_z)/2)
                    for index, Ho_mi in enumerate(Ho_matrix):
                        Ho += Ho_mi
                    qml.QubitUnitary(expm(-1j * Ho_params[layer] * self.Ho_dir * Ho), wires=range(num_qubits))
                # 惩罚约束
                for bit_strings in range(len(Hd_bits_list)):
                    hd_bits = Hd_bits_list[bit_strings]
                    nonzero_indices = np.nonzero(hd_bits)[0]
                    nonzerobits = hd_bits[nonzero_indices]
                    hdi_string = [0 if x == -1 else 1 for x in hd_bits if x != 0]
                    if use_decompose:
                        driver_component_pennylane(nonzero_indices, [num_qubits] ,hdi_string, Hd_params[layer])
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

    def get_costfunc(self):
        def costfunc(params):
            collapse_state, probs = self.inference(params)
            costs = 0
            for value, prob in zip(collapse_state, probs):
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




class QiskitCircuit:
    def __init__(self, circuit_option: CircuitOption):
        self.circuit_option = circuit_option
        assert circuit_option.optimization_direction in ['min', 'max']
        self.Ho_dir =  1 if circuit_option.optimization_direction == 'max' else -1
        if circuit_option.algorithm_optimization_method == 'cyclic':
            self.cyclic_qubit_set = {item for sublist in self.circuit_option.constraints_for_cyclic for item in np.nonzero(sublist[:-1])[0]}
            self.others_qubit_set = set(range(circuit_option.num_qubits)) - self.cyclic_qubit_set
        self.num_qubits = circuit_option.num_qubits
        self.num_layers = circuit_option.num_layers

    def inference(self, params, shots=1024):
        # backend = FakeQuebec()
        # backend = FakeAlmadenV2()
        backend = AerSimulator()
        qc = self.inference_circuit(params)
        # pm = generate_preset_pass_manager(backend=backend, optimization_level=2)
        pm = generate_preset_pass_manager(optimization_level=2,basis_gates=['ecr', 'id', 'rz', 'sx','x'])
        transpiled_qc = pm.run(qc)
        if self.circuit_option.debug:
            print("transpiled_qc.depth()", transpiled_qc.depth())
            print("transpiled_qc.width()", transpiled_qc.width())
            print("transpiled_qc.two_qubit_gate()", transpiled_qc.num_nonlocal_gates())
        # print("transpiled_qc.duration()", transpiled_qc.duration())
        options = {"simulator": {"seed_simulator": 42}}
        sampler = Sampler(backend=backend, options=options)
        result = sampler.run([transpiled_qc],shots=shots).result()
        pub_result = result[0]
        counts = pub_result.data.c.get_counts()
        collapse_state = [[int(char) for char in state] for state in counts.keys()]
        total_count = sum(counts.values())
        probs = [count / total_count for count in counts.values()]
        return collapse_state, probs
    
    def create_circuit(self) -> None:
        use_decompose = self.circuit_option.use_decompose
        use_Ho_gate_list = self.circuit_option.use_Ho_gate_list
        Ho_gate_list = self.circuit_option.Ho_gate_list
        Ho_vector = self.circuit_option.linear_objective_vector
        Ho_matrix = self.circuit_option.nonlinear_objective_matrix
        self.inference_circuit = None
        num_qubits = self.num_qubits
        num_layers = self.num_layers
        algorithm_optimization_method = self.circuit_option.algorithm_optimization_method

        if algorithm_optimization_method == 'commute':
            if self.circuit_option.mcx_mode == 'constant':
                qc = QuantumCircuit(num_qubits + 2, num_qubits)
                ancilla = [num_qubits, num_qubits + 1]
            elif self.circuit_option.mcx_mode == 'linear':
                qc = QuantumCircuit(2*num_qubits, num_qubits)
                ancilla = list(range(num_qubits, 2*num_qubits))
            else:
                raise ValueError("mcx_mode should be 'constant' or 'linear'")
        else:
            qc = QuantumCircuit(num_qubits, num_qubits)
    
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

        def circuit_penalty(params):
            Ho_params = params[:num_layers]
            Hd_params = params[num_layers:]
            assert len(Hd_params) == num_layers
            pnt_lbd = self.circuit_option.penalty_lambda
            constraints = self.circuit_option.constraints
            for i in range(num_qubits):
                qc.h(i)
            for layer in range(num_layers):
                if use_Ho_gate_list == True:
                    for i_list, theta in Ho_gate_list:
                        for i in range(len(i_list) - 1):
                            qc.cx(i_list[i], i_list[i + 1])
                        qc.rz(theta, i_list[-1])
                        for i in range(len(i_list) - 2, -1, -1):
                            qc.cx(i_list[i], i_list[i + 1])
                elif use_Ho_gate_list == False:
                    Ho = np.zeros((2**num_qubits, 2**num_qubits))
                    for index, Ho_vi in enumerate(Ho_vector):
                        Ho += Ho_vi * add_in_target(num_qubits, index, (gate_I - gate_z)/2)
                    for index, Ho_mi in enumerate(Ho_matrix):
                        Ho += Ho_mi
                # 惩罚项 Hamiltonian penalty
                if use_decompose:
                    for penalty_mi in constraints:
                        coeff = (np.sum(penalty_mi)-3*penalty_mi[-1])/2
                        for i in range(num_qubits):
                            qc.rz(coeff*penalty_mi[i]*Ho_params[layer], i)
                        for i in range(num_qubits-1):
                            for j in range(i+1, num_qubits):
                                if penalty_mi[i] != 0 and penalty_mi[j] != 0:
                                    qc.cx(i, j)
                                    coeff = (penalty_mi[i]* penalty_mi[j])/2
                                    qc.rz(coeff*Ho_params[layer], j)
                                    qc.cx(i, j)

                else:
                    for penalty_mi in constraints:
                        H_pnt = np.zeros((2**num_qubits, 2**num_qubits))
                        for index, penalty_vi in enumerate(penalty_mi[:-1]):
                            H_pnt += penalty_vi * add_in_target(num_qubits, index, (gate_I - gate_z)/2)
                        H_pnt -= penalty_mi[-1] * np.eye(2**num_qubits)
                        Ho += pnt_lbd * H_pnt @ H_pnt
                    # Ho取负，对应绝热演化能级问题，但因为训练参数，可能没区别
                    qc.unitary(expm(-1j * Ho_params[layer] * self.Ho_dir * Ho), range(num_qubits))
                # Rx 驱动哈密顿量
                for i in range(num_qubits):
                    qc.rx(Hd_params[layer], i)
            qc.measure(range(num_qubits), range(num_qubits))
            return qc

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
                qc.x(i)
            for i in others_qubit_set:
                qc.h(i)                 
            for layer in range(num_layers):
                
                if use_Ho_gate_list == True:
                    for i_list, theta in Ho_gate_list:
                        for i in range(len(i_list) - 1):
                            qc.cx(i_list[i], i_list[i + 1])
                        qc.rz(theta, i_list[-1])
                        for i in range(len(i_list) - 2, -1, -1):
                            qc.cx(i_list[i], i_list[i + 1])
                elif use_Ho_gate_list == False:
                    Ho = np.zeros((2**num_qubits, 2**num_qubits))
                    for index, Ho_vi in enumerate(Ho_vector):
                        Ho += Ho_vi * add_in_target(num_qubits, index, (gate_I - gate_z)/2)
                    for index, Ho_mi in enumerate(Ho_matrix):
                        Ho += Ho_mi
                # constraints_for_others 惩罚项
                if use_decompose:
                    for penalty_mi in constraints_for_others:
                        coeff = (np.sum(penalty_mi)-3*penalty_mi[-1])/2
                        for i in range(num_qubits):
                            qc.rz(coeff*penalty_mi[i]*Ho_params[layer], i)
                        for i in range(num_qubits-1):
                            for j in range(i+1, num_qubits):
                                if penalty_mi[i] != 0 and penalty_mi[j] != 0:
                                    qc.cx(i, j)
                                    coeff = (penalty_mi[i]* penalty_mi[j])/2
                                    qc.rz(coeff*Ho_params[layer], j)
                                    qc.cx(i, j)
                else:
                    for penalty_mi in constraints_for_others:
                        H_pnt = np.zeros((2**num_qubits, 2**num_qubits))
                        for index, penalty_vi in enumerate(penalty_mi[:-1]):
                            H_pnt += penalty_vi * add_in_target(num_qubits, index, (gate_I - gate_z)/2)
                        H_pnt -= penalty_mi[-1] * np.eye(2**num_qubits)
                        Ho += pnt_lbd * H_pnt @ H_pnt
                    # Ho取负，对应绝热演化能级问题，但因为训练参数，可能没区别
                    qc.unitary(expm(-1j * Ho_params[layer] * self.Ho_dir * Ho), range(num_qubits))
                # constraints_for_cyclic --> 驱动哈密顿量
                if use_decompose:
                    for cyclic_mi in constraints_for_cyclic:
                        nzlist = np.nonzero(cyclic_mi[:-1])[0]
                        for i in range(len(nzlist)):
                            j = (i + 1) % len(nzlist)
                            ## gate for X_iX_j
                            qc.h(nzlist[j])
                            qc.cx(nzlist[i], nzlist[j])
                            qc.rz(Hd_params[layer],nzlist[j])
                            qc.cx(nzlist[i], nzlist[j])
                            qc.h(nzlist[j])
                            ## gate for Y_iY_j
                            qc.u(np.pi/2, np.pi/2, np.pi/2,nzlist[j])
                            qc.cx(nzlist[i], nzlist[j])
                            qc.rz(Hd_params[layer],nzlist[j])
                            qc.cx(nzlist[i], nzlist[j])
                            qc.u(np.pi/2, np.pi/2, np.pi/2,nzlist[j])

                else:
                    for cyclic_mi in constraints_for_cyclic:
                        nzlist = np.nonzero(cyclic_mi[:-1])[0]
                        cyclic_Hd_i = gnrt_cyclic_Hd(len(nzlist))
                        qc.unitary(expm(-1j * Hd_params[layer] * cyclic_Hd_i), nzlist)
                for i in others_qubit_set:
                    qc.rx(Hd_params[layer], i)
            qc.measure(range(num_qubits), range(num_qubits))
            return qc
        
        def circuit_commute(params):
            Ho_params = params[:num_layers]
            Hd_params = params[num_layers:]
            assert len(Hd_params) == num_layers
            Hd_bits_list = self.circuit_option.Hd_bits_list
            for i in np.nonzero(self.circuit_option.feasiable_state)[0]:
                qc.x(i)
            for layer in range(num_layers):
                if use_Ho_gate_list == True:
                    for i_list, theta in Ho_gate_list:
                        for i in range(len(i_list) - 1):
                            qc.cx(i_list[i], i_list[i + 1])
                        qc.rz(theta, i_list[-1])
                        for i in range(len(i_list) - 2, -1, -1):
                            qc.cx(i_list[i], i_list[i + 1])
                elif use_Ho_gate_list == False:
                    #$ 目标函数
                    Ho = np.zeros((2**num_qubits, 2**num_qubits))
                    for index, Ho_vi in enumerate(Ho_vector):
                        Ho += Ho_vi * add_in_target(num_qubits, index, (gate_I - gate_z)/2)
                    for index, Ho_mi in enumerate(Ho_matrix):
                        Ho += Ho_mi
                    qc.unitary(expm(-1j * Ho_params[layer] * self.Ho_dir * Ho), range(num_qubits))
                # 通过对易哈密顿量惩罚约束
                for bit_strings in range(len(Hd_bits_list)):
                    hd_bits = Hd_bits_list[bit_strings]
                    nonzero_indices = np.nonzero(hd_bits)[0].tolist()
                    nonzerobits = hd_bits[nonzero_indices]
                    hdi_string = [0 if x == -1 else 1 for x in hd_bits if x != 0]
                    if use_decompose:
                        driver_component_qiskit(qc, nonzero_indices, ancilla ,hdi_string, Hd_params[layer])
                    else:
                        qc.unitary(expm(-1j * Hd_params[layer] * plus_minus_gate_sequence_to_unitary(nonzerobits)), nonzero_indices)
            qc.measure(range(num_qubits), range(num_qubits))
            return qc
        
        def circuit_HEA(params):
            params = np.array(params).reshape((num_layers, num_qubits, 3))
            r1_params = params[:, :, 0]
            r2_params = params[:, :, 1]
            r3_params = params[:, :, 2]
            assert len(r3_params) == num_layers
            for layer in range(num_layers):
                for i in range(num_qubits):
                    qc.rz(r1_params[layer][i], i)
                    qc.ry(r2_params[layer][i], i)
                    qc.rz(r3_params[layer][i], i)
                for i in range(num_qubits):
                    qc.cx(i, (i + 1) % num_qubits)
            qc.measure(range(num_qubits), range(num_qubits))
            return qc
    
        circuit_map = {
            'penalty': circuit_penalty,
            'cyclic': circuit_cyclic,
            'commute': circuit_commute,
            'HEA': circuit_HEA
        }
        self.inference_circuit = circuit_map.get(algorithm_optimization_method)

    def get_costfunc(self):
        def costfunc(params):
            collapse_state, probs = self.inference(params)
            costs = 0
            for value, prob in zip(collapse_state, probs):
                costs += self.circuit_option.objective(value) * prob
            return costs

        return costfunc
    
    def draw_circuit(self) -> None:
        if self.circuit_option.algorithm_optimization_method == 'HEA':
            print(self.inference_circuit(np.zeros(self.num_layers * self.num_qubits * 3)).draw())
        else:
            print(self.inference_circuit(np.zeros(self.num_layers * 2)).draw())

