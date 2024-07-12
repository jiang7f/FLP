from quBLP.utils import iprint
import pennylane as qml
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit_ibm_runtime.fake_provider import FakeKyoto, FakeKyiv, FakeSherbrooke, FakeQuebec, FakeAlmadenV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from ...utils.quantum_lib import *
from qiskit_aer import AerSimulator
import numpy as np
import numpy.linalg as ls
from typing import List, Tuple
from collections.abc import Iterable
from scipy.linalg import expm
from .pennylane_decompose import driver_component as driver_component_pennylane
from .qiskit_decompose import driver_component as driver_component_qiskit
from ...models import CircuitOption
from qiskit_ibm_runtime import SamplerV2 as Sampler
from time import perf_counter
from itertools import combinations
from ...analysis import Feature
from ...utils import QuickFeedbackException
from qiskit.qasm2 import dump


def calculate_fidelity_by_counts(counts, target_counts) -> float:
    """
    计算两个计数分布之间的相似度
    """
    probvectordiff = []
    for key in set(counts.keys()).union(set((target_counts.keys()))):
        probvectordiff.append(abs(counts.get(key, 0) - target_counts.get(key, 0)) / max(counts.get(key, 0), target_counts.get(key, 0)))
    return 1 - sum(probvectordiff) / len(probvectordiff)
    
    
class QiskitCircuit:
    def __init__(self, circuit_option: CircuitOption):
        self.circuit_option = circuit_option
        if circuit_option.algorithm_optimization_method == 'cyclic':
            self.cyclic_qubit_set = {item for sublist in self.circuit_option.constraints_for_cyclic for item in np.nonzero(sublist[:-1])[0]}
            self.others_qubit_set = set(range(circuit_option.num_qubits)) - self.cyclic_qubit_set
        self.num_qubits = circuit_option.num_qubits
        self.num_layers = circuit_option.num_layers

        #+ will to delete
        iprint(circuit_option.backend)
        
        if circuit_option.backend == 'FakeQuebec':
            self.backend = FakeQuebec() # 新版 真机
            self.pass_manager = generate_preset_pass_manager(backend=self.backend, optimization_level=2)
        elif circuit_option.backend == 'FakeAlmadenV2':  
            self.backend = FakeAlmadenV2() # 旧版 真机
            self.pass_manager = generate_preset_pass_manager(backend=self.backend, optimization_level=2)
        # elif circuit_option.backend == 'AerSimulator':
        else:
            self.backend = AerSimulator()    # 仿真
            self.pass_manager = generate_preset_pass_manager(optimization_level=2, basis_gates=['measure', 'cx', 'id', 's','sdg', 'x','y','h','z','mcx','cz','sx','sy','t','tdg','swap','rx','ry'])      
            # self.pass_manager = generate_preset_pass_manager(optimization_level=2, basis_gates=['ecr', 'id', 'rz', 'sx', 'x'])        

    def inference(self, params, shots=10000):
        feedback = self.circuit_option.feedback
        if self.circuit_option.use_decompose:
            final_qc = self.inference_circuit.assign_parameters(params) # 已使用pass_manager编译过的理论电路 (只缺参数)
        else:
            final_qc = self.inference_circuit(params)
        with open("testqc.qasm", "w") as f:
            dump(final_qc, f)
        if feedback is None or feedback == [] or 'run_time' in feedback:
            start = perf_counter()
            options = {"simulator": {"seed_simulator": 42}}
            if self.circuit_option.backend == 'AerSimulator':
                sampler = Sampler(backend=self.backend, options=options)
                result = sampler.run([final_qc],shots=shots).result()
                pub_result = result[0]
                counts = pub_result.data.c.get_counts()
            elif self.circuit_option.backend == 'ddsim':
                from mqt import ddsim
                backend = ddsim.DDSIMProvider().get_backend("qasm_simulator")
                job = backend.run(final_qc, shots=shots)
                counts = job.result().get_counts(final_qc)
            end = perf_counter()
            self.run_time = end - start
        if feedback is not None and len(feedback) > 0:
            # iprint(final_qc.draw())
            feature = Feature(final_qc, self.backend)
            self.width = feature.width
            self.depth = feature.depth
            self.num_one_qubit_gates = feature.num_one_qubit_gates
            self.num_two_qubit_gates = feature.num_two_qubit_gates
            self.size = feature.size
            self.latency = feature.latency_all()
            self.culled_depth = feature.get_depth_without_one_qubit_gate()
            feedback_data = {feedback_term: getattr(self, feedback_term, None) for feedback_term in feedback}
            raise QuickFeedbackException(message=f"debug finished: {self.circuit_option.algorithm_optimization_method}, {self.circuit_option.backend}, use_decompose={self.circuit_option.use_decompose}",
                                         data=feedback_data)
        collapse_state = [[int(char) for char in state] for state in counts.keys()]
        total_count = sum(counts.values())
        probs = [count / total_count for count in counts.values()]
        return collapse_state, probs
    
    def create_circuit(self) -> None:
        use_decompose = self.circuit_option.use_decompose
        objective_func_term_list = self.circuit_option.objective_func_term_list
        self.inference_circuit = None
        num_qubits = self.num_qubits
        num_layers = self.num_layers
        algorithm_optimization_method = self.circuit_option.algorithm_optimization_method

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

        def Ho_decompose(qc:QuantumCircuit, objective_func_term_list, param):
            for n_power_term_list in objective_func_term_list:
                for var_idx_list, theta in n_power_term_list :
                    pow = len(var_idx_list)
                    for k in range(1, pow + 1):
                        fianl_theta = (1 / 2) ** pow * 2 * theta * (-1) ** k
                        for combo in combinations(range(pow), k):
                            for i in range(len(combo) - 1):
                                qc.cx(var_idx_list[combo[i]], var_idx_list[combo[i + 1]])
                            qc.rz(fianl_theta * param, var_idx_list[combo[-1]])
                            for i in range(len(combo) - 2, -1, -1):
                                qc.cx(var_idx_list[combo[i]], var_idx_list[combo[i + 1]])


        def penalty_decompose(qc:QuantumCircuit, penalty_mi:List, param:Parameter):
            coeff = np.sum(penalty_mi[:-1]) / 2 - penalty_mi[-1]
            for i in range(num_qubits):
                qc.rz(-2 * coeff * penalty_mi[i] * param, i)
            for i in range(num_qubits - 1):
                for j in range(i + 1, num_qubits):
                    if penalty_mi[i] != 0 and penalty_mi[j] != 0:
                        qc.cx(i, j)
                        coeff = penalty_mi[i] * penalty_mi[j]
                        qc.rz(coeff * param, j)
                        qc.cx(i, j)

        def Ho_unitary(objective_func_term_list: List[List[Tuple[List[int], float]]]):
            Ho = np.zeros((2**num_qubits, 2**num_qubits))
            for [var_idx], theta in objective_func_term_list[0]:
                Ho += theta * add_in_target(num_qubits, var_idx, (gate_I - gate_z) / 2)
            for [var_idx_1, var_idx_2], theta in objective_func_term_list[1]:
                Ho += theta * add_in_target(num_qubits, var_idx_1, (gate_I - gate_z) / 2) @ add_in_target(num_qubits, var_idx_2, (gate_I - gate_z) / 2)
            return Ho

        def penalty_unitary(penalty_mi: List):
            H_pnt = np.zeros((2**num_qubits, 2**num_qubits))
            for index, penalty_vi in enumerate(penalty_mi[:-1]):
                H_pnt += penalty_vi * add_in_target(num_qubits, index, (gate_I - gate_z)/2)
            H_pnt -= penalty_mi[-1] * np.eye(2**num_qubits)
            return H_pnt @ H_pnt

        def circuit_penalty(params=None):
            qc = QuantumCircuit(num_qubits, num_qubits)
            if use_decompose:
                Ho_params = [Parameter(f'Ho_params[{i}]') for i in range(num_layers)]
                Hd_params = [Parameter(f'Hd_params[{i}]') for i in range(num_layers)]
            else:
                Ho_params = params[:num_layers]
                Hd_params = params[num_layers:]
            assert len(Hd_params) == num_layers

            pnt_lbd = self.circuit_option.penalty_lambda
            linear_constraints = self.circuit_option.linear_constraints
            for i in range(num_qubits):
                qc.h(i)
            for layer in range(num_layers):
                if use_decompose:
                    # 目标函数
                    Ho_decompose(qc, objective_func_term_list, Ho_params[layer])
                    # 惩罚项 Hamiltonian penalty
                    for penalty_mi in linear_constraints:
                        penalty_decompose(qc, penalty_mi, Ho_params[layer])
                else:
                    Ho = Ho_unitary(objective_func_term_list)
                    # 惩罚项 Hamiltonian penalty
                    for penalty_mi in linear_constraints:
                        Ho += pnt_lbd * penalty_unitary(penalty_mi)
                    # Class Parameter 存在问题，待修改
                    # Ho取负，对应绝热演化能级问题，但因为训练参数，可能没区别
                    qc.unitary(expm(-1j * Ho_params[layer] * Ho), range(num_qubits)[::-1])
                # Rx 驱动哈密顿量
                for i in range(num_qubits):
                    qc.rx(Hd_params[layer], i)
            qc.measure(range(num_qubits), range(num_qubits)[::-1])
            if self.circuit_option.feedback is not None and 'transpile_time' in self.circuit_option.feedback:
                start = perf_counter()
            transpiled_qc = self.pass_manager.run(qc)  
            if self.circuit_option.feedback is not None and 'transpile_time' in self.circuit_option.feedback:
                end = perf_counter()
                self.transpile_time = end - start
            return transpiled_qc

        def circuit_cyclic(params=None):
            qc = QuantumCircuit(num_qubits, num_qubits)
            if use_decompose:
                Ho_params = [Parameter(f'Ho_params[{i}]') for i in range(num_layers)]
                Hd_params = [Parameter(f'Hd_params[{i}]') for i in range(num_layers)]
            else:
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
                if use_decompose:
                    # 目标函数
                    Ho_decompose(qc, objective_func_term_list, Ho_params[layer])
                    # constraints_for_others 惩罚项
                    for penalty_mi in constraints_for_others:
                        penalty_decompose(qc, penalty_mi, Ho_params[layer])
                    # constraints_for_cyclic --> 驱动哈密顿量
                    for cyclic_mi in constraints_for_cyclic:
                        nzlist = np.nonzero(cyclic_mi[:-1])[0]
                        for i in range(len(nzlist)):
                            j = (i + 1) % len(nzlist)
                            ## gate for X_iX_j
                            qc.h(nzlist[j])
                            qc.h(nzlist[i])
                            qc.cx(nzlist[i], nzlist[j])
                            qc.rz(2 * Hd_params[layer], nzlist[j])
                            qc.cx(nzlist[i], nzlist[j])
                            qc.h(nzlist[j])
                            qc.h(nzlist[i])
                            ## gate for Y_iY_j
                            qc.u(np.pi / 2, np.pi / 2, np.pi / 2, nzlist[j])
                            qc.u(np.pi / 2, np.pi / 2, np.pi / 2, nzlist[i])
                            qc.cx(nzlist[i], nzlist[j])
                            qc.rz(2 * Hd_params[layer], nzlist[j])
                            qc.cx(nzlist[i], nzlist[j])
                            qc.u(np.pi / 2, np.pi / 2, np.pi / 2, nzlist[j])
                            qc.u(np.pi / 2, np.pi / 2, np.pi / 2, nzlist[i])

                else:
                    Ho = Ho_unitary(objective_func_term_list)
                    # constraints_for_others 惩罚项
                    for penalty_mi in constraints_for_others:
                        Ho += pnt_lbd * penalty_unitary(penalty_mi)
                    # Ho取负，对应绝热演化能级问题，但因为训练参数，可能没区别
                    qc.unitary(expm(-1j * Ho_params[layer] * Ho), range(num_qubits)[::-1])
                    # constraints_for_cyclic --> 驱动哈密顿量
                    for cyclic_mi in constraints_for_cyclic:
                        nzlist = np.nonzero(cyclic_mi[:-1])[0]
                        nzlist = [int(x) for x in nzlist]
                        cyclic_Hd_i = gnrt_cyclic_Hd(len(nzlist))
                        qc.unitary(expm(-1j * Hd_params[layer] * cyclic_Hd_i), nzlist[::-1])
                for i in others_qubit_set:
                    qc.rx(Hd_params[layer], i)
            qc.measure(range(num_qubits), range(num_qubits)[::-1])
            if self.circuit_option.feedback is not None and 'transpile_time' in self.circuit_option.feedback:
                start = perf_counter()
            transpiled_qc = self.pass_manager.run(qc)
            if self.circuit_option.feedback is not None and 'transpile_time' in self.circuit_option.feedback:
                end = perf_counter()
                self.transpile_time = end - start
            return transpiled_qc
        
        def circuit_commute(params=None):
            iprint("'circuit_commute' function ran once") # wait be delete
            mcx_mode = self.circuit_option.mcx_mode
            if self.circuit_option.use_decompose == False:
                qc = QuantumCircuit(num_qubits, num_qubits)
            elif mcx_mode == 'constant':
                qc = QuantumCircuit(num_qubits + 2, num_qubits)
                ancilla = [num_qubits, num_qubits + 1]
            elif mcx_mode == 'linear':
                qc = QuantumCircuit(2 * num_qubits, num_qubits)
                ancilla = list(range(num_qubits, 2 * num_qubits))
            else:
                qc = QuantumCircuit(num_qubits + 2, num_qubits)
                ancilla = list(range(num_qubits,num_qubits + 2))
                # raise ValueError("mcx_mode should be 'constant' or 'linear'")

            if use_decompose:
                Ho_params = [Parameter(f'Ho_params[{i}]') for i in range(num_layers)]
                Hd_params = [Parameter(f'Hd_params[{i}]') for i in range(num_layers)]
            else:
                Ho_params = params[:num_layers]
                Hd_params = params[num_layers:]
            assert len(Hd_params) == num_layers

            Hd_bits_list = self.circuit_option.Hd_bits_list
            for i in np.nonzero(self.circuit_option.feasiable_state)[0]:
                qc.x(i)
            for layer in range(num_layers):
                #$ Ho
                if use_decompose:
                    # 目标函数
                    Ho_decompose(qc, objective_func_term_list, Ho_params[layer])
                else:
                    Ho = Ho_unitary(objective_func_term_list)
                    qc.unitary(expm(-1j * Ho_params[layer] * Ho), range(num_qubits)[::-1])
                # Hd 通过对易哈密顿量惩罚约束
                for bit_strings in range(len(Hd_bits_list)):
                    hd_bits = Hd_bits_list[bit_strings]
                    nonzero_indices = np.nonzero(hd_bits)[0].tolist()
                    nonzerobits = hd_bits[nonzero_indices]
                    hdi_string = [0 if x == -1 else 1 for x in hd_bits if x != 0]
                    if use_decompose:
                        driver_component_qiskit(qc, nonzero_indices, ancilla, hdi_string, Hd_params[layer], mcx_mode)
                    else:
                        qc.unitary(expm(-1j * Hd_params[layer] * plus_minus_gate_sequence_to_unitary(nonzerobits)), nonzero_indices[::-1])
            qc.measure(range(num_qubits), range(num_qubits)[::-1])
            if self.circuit_option.feedback is not None and 'transpile_time' in self.circuit_option.feedback:
                start = perf_counter()
            transpiled_qc = self.pass_manager.run(qc)
            if self.circuit_option.feedback is not None and 'transpile_time' in self.circuit_option.feedback:
                end = perf_counter()
                self.transpile_time = end - start
            return transpiled_qc
        
        def circuit_HEA(params=None):
            qc = QuantumCircuit(num_qubits, num_qubits)
            if use_decompose:
                r1_params = [[Parameter(f'r1_params[{i}_{j}]') for j in range(num_qubits)] for i in range(num_layers)]
                r2_params = [[Parameter(f'r2_params[{i}_{j}]') for j in range(num_qubits)] for i in range(num_layers)]
                r3_params = [[Parameter(f'r3_params[{i}_{j}]') for j in range(num_qubits)] for i in range(num_layers)]
            else:
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
            qc.measure(range(num_qubits), range(num_qubits)[::-1])
            if self.circuit_option.feedback is not None and 'transpile_time' in self.circuit_option.feedback:
                start = perf_counter()
            transpiled_qc = self.pass_manager.run(qc)
            if self.circuit_option.feedback is not None and 'transpile_time' in self.circuit_option.feedback:
                end = perf_counter()
                self.transpile_time = end - start
            return transpiled_qc
    
        circuit_map = {
            'penalty': circuit_penalty,
            'cyclic': circuit_cyclic,
            'commute': circuit_commute,
            'HEA': circuit_HEA
        }
        if use_decompose:
            self.inference_circuit = circuit_map.get(algorithm_optimization_method)()
        else:
            self.inference_circuit = circuit_map.get(algorithm_optimization_method)

    def get_circuit_cost_function(self):
        def circuit_cost_function(params):
            collapse_state, probs = self.inference(params)
            costs = 0
            for value, prob in zip(collapse_state, probs):
                costs += self.circuit_option.objective_func(value) * prob
            return costs
        return circuit_cost_function
    
    def draw_circuit(self) -> None:
        # 待修改
        if self.circuit_option.use_decompose:
            qc = self.inference_circuit
            if self.circuit_option.algorithm_optimization_method == 'HEA':
                iprint(qc.assign_parameters(np.zeros(self.num_layers * self.num_qubits * 3)).draw())
            else:
                iprint(qc.assign_parameters(np.zeros(self.num_layers * 2)).draw())
        else:
            if self.circuit_option.algorithm_optimization_method == 'HEA':
               params = np.zeros(self.num_layers * self.num_qubits * 3)
            else:
               params = np.zeros(self.num_layers * 2)
            iprint(self.inference_circuit(params).draw())

class PennylaneCircuit:
    def __init__(self, circuit_option: CircuitOption):
        self.circuit_option = circuit_option
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
        objective_func_term_list = self.circuit_option.objective_func_term_list
        self.inference_circuit = None
        num_qubits = self.num_qubits
        num_layers = self.num_layers
        algorithm_optimization_method = self.circuit_option.algorithm_optimization_method
        
        dev = qml.device("default.qubit", wires=num_qubits + 1)

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

        def Ho_decompose(objective_func_term_list: List[List[Tuple[List[int], float]]], param: Parameter):
            # 一次项
            for [var_idx], theta in objective_func_term_list[0]:
                qml.RZ(-theta * param, var_idx)
            # 二次项
            for [var_idx_1, var_idx_2], theta in objective_func_term_list[1]:
                qml.RZ(-theta * param / 2, var_idx_1)
                qml.RZ(-theta * param / 2, var_idx_2)
                qml.CNOT(var_idx_1, var_idx_2)
                qml.RZ(theta / 2 * param, var_idx_2)
                qml.CNOT(var_idx_1, var_idx_2)

        def penalty_decompose(penalty_mi:List, param:Parameter): 
            coeff = np.sum(penalty_mi[:-1]) / 2 - penalty_mi[-1]
            for i in range(num_qubits):
                qml.RZ(-2 * coeff * penalty_mi[i] * param, i)
            for i in range(num_qubits - 1):
                for j in range(i + 1, num_qubits):
                    if penalty_mi[i] != 0 and penalty_mi[j] != 0:
                        qml.CNOT(wires=[i, j])
                        coeff = penalty_mi[i] * penalty_mi[j]
                        qml.RZ(coeff * param, j)
                        qml.CNOT(wires=[i, j])

        def Ho_unitary(objective_func_term_list: List[List[Tuple[List[int], float]]]):
            Ho = np.zeros((2**num_qubits, 2**num_qubits))
            for [var_idx], theta in objective_func_term_list[0]:
                Ho += theta * add_in_target(num_qubits, var_idx, (gate_I - gate_z) / 2)
            for [var_idx_1, var_idx_2], theta in objective_func_term_list[1]:
                Ho += theta * add_in_target(num_qubits, var_idx_1, (gate_I - gate_z) / 2) @ add_in_target(num_qubits, var_idx_2, (gate_I - gate_z) / 2)
            return Ho

        def penalty_unitary(penalty_mi: List):
            H_pnt = np.zeros((2**num_qubits, 2**num_qubits))
            for index, penalty_vi in enumerate(penalty_mi[:-1]):
                H_pnt += penalty_vi * add_in_target(num_qubits, index, (gate_I - gate_z)/2)
            H_pnt -= penalty_mi[-1] * np.eye(2**num_qubits)
            return H_pnt @ H_pnt

        @qml.qnode(dev)
        def circuit_penalty(params):
            Ho_params = params[:num_layers]
            Hd_params = params[num_layers:]
            assert len(Hd_params) == num_layers
            pnt_lbd = self.circuit_option.penalty_lambda
            constraints = self.circuit_option.linear_constraints
            for i in range(num_qubits):
                qml.Hadamard(i)
            for layer in range(num_layers):
                if use_decompose:
                    # 目标函数
                    Ho_decompose(objective_func_term_list, Ho_params[layer])
                    # 惩罚项 Hamiltonian penalty
                    for penalty_mi in constraints:
                        penalty_decompose(penalty_mi, Ho_params[layer])
                else:
                    Ho = Ho_unitary(objective_func_term_list)
                    # 惩罚项 Hamiltonian penalty
                    for penalty_mi in constraints:
                        Ho += pnt_lbd * penalty_unitary(penalty_mi)
                    # Ho取负，对应绝热演化能级问题，但因为训练参数，可能没区别
                    qml.QubitUnitary(expm(-1j * Ho_params[layer] * Ho), wires=range(num_qubits))
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
                if use_decompose:
                    # 目标函数
                    Ho_decompose(objective_func_term_list, Ho_params[layer])
                    # constraints_for_others 惩罚项
                    for penalty_mi in constraints_for_others:
                        penalty_decompose(penalty_mi, Ho_params[layer])
                    # constraints_for_cyclic --> 驱动哈密顿量
                    for cyclic_mi in constraints_for_cyclic:
                        nzlist = np.nonzero(cyclic_mi[:-1])[0]
                        for i in range(len(nzlist)):
                            j = (i + 1) % len(nzlist)
                            ## gate for X_iX_j
                            qml.Hadamard(nzlist[j])
                            qml.CNOT(wires=[nzlist[i], nzlist[j]])
                            qml.RZ(Hd_params[layer], nzlist[j])
                            qml.CNOT(wires=[nzlist[i], nzlist[j]])
                            qml.Hadamard(nzlist[j])
                            ## gate for Y_iY_j
                            qml.U3(np.pi / 2, np.pi / 2, np.pi / 2, nzlist[j])
                            qml.CNOT(wires=[nzlist[i], nzlist[j]])
                            qml.RZ(Hd_params[layer], nzlist[j])
                            qml.CNOT(wires=[nzlist[i], nzlist[j]])
                            qml.U3(np.pi / 2, np.pi / 2, np.pi / 2, nzlist[j])
                else:
                    Ho = Ho_unitary(objective_func_term_list)
                    # constraints_for_others 惩罚项
                    for penalty_mi in constraints_for_others:
                        Ho += pnt_lbd * penalty_unitary(penalty_mi)
                    # Ho取负，对应绝热演化能级问题，但因为训练参数，可能没区别
                    qml.QubitUnitary(expm(-1j * Ho_params[layer] * Ho), wires=range(num_qubits))
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
                if use_decompose:
                    # 目标函数
                    Ho_decompose(objective_func_term_list, Ho_params[layer])
                else:
                    #$ 目标函数
                    Ho = Ho_unitary(objective_func_term_list)
                    qml.QubitUnitary(expm(-1j * Ho_params[layer] * Ho), wires=range(num_qubits))
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

    def get_circuit_cost_function(self):
        def circuit_cost_function(params):
            collapse_state, probs = self.inference(params)
            costs = 0
            for value, prob in zip(collapse_state, probs):
                costs += self.circuit_option.objective_func(value) * prob
            return costs
        return circuit_cost_function
    
    def draw_circuit(self) -> None:
        from pennylane.drawer import draw
        circuit_drawer = draw(self.inference_circuit)
        if self.circuit_option.algorithm_optimization_method == 'HEA':
            iprint(circuit_drawer(np.zeros(self.num_layers * self.num_qubits * 3)))
        else:
            iprint(circuit_drawer(np.zeros(self.num_layers * 2)))