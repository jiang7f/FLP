from quBLP.utils import iprint
from qiskit.circuit import QuantumCircuit
from qiskit.converters import dag_to_circuit, circuit_to_dag
from qiskit.dagcircuit import DAGOpNode
from qiskit import transpile
from math import prod
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeKyoto, FakeKyiv, FakeSherbrooke, FakeQuebec, FakeAlmadenV2, FakeBelem, FakeSantiago
from .Metrics import Metric
class Feature:
    """ get the feature of a quantum circuit,like fidelity, latency for every qubits
    """
    def __init__(self, circuit: QuantumCircuit, backend=None) -> None:
        """

        Args:
            circuit (QuantumCircuit): instance of circuit
            backend (_type_): the qpu backend in IBM backend object, contains the information of a qpu
        """
        if backend is None or backend.name == 'aer_simulator':
             self._circuit = circuit
        else:
            self._circuit = transpile(circuit, backend)
        self._dagcircuit = circuit_to_dag(circuit)
        self._qargs = circuit.qubits
        self._metric = Metric(backend=backend)
        self.width = self._circuit.width()
        self.depth = self._circuit.depth()
        self.one_qubit_gates = [node.op for node in self._dagcircuit._multi_graph.nodes() if isinstance(node, DAGOpNode) and len(node.qargs) == 1]
        self.two_qubit_gates = [node.op for node in self._dagcircuit._multi_graph.nodes() if isinstance(node, DAGOpNode) and len(node.qargs) == 2]
        self.num_one_qubit_gates = len(self.one_qubit_gates)
        self.num_two_qubit_gates = len(self.two_qubit_gates)
        self.size = self._circuit.size()
        self._latency_dict = None

    def latency(self, qubit):
        return self.latency_dict[qubit]

    def latency_all(self):
        return max(self.latency_dict.values())

    @property
    def latency_dict(self):
        if self._latency_dict is None:
            self._latency_dict = {q: 0 for q in self._qargs}
            for gate in self._dagcircuit.topological_op_nodes():
                q_indexes = [q._index for q in gate.qargs]
                if gate.op.name != 'measure':
                    max_latency = max([self._latency_dict[q] for q in gate.qargs]) + self._metric.latency(gate.op.name, q_indexes)
                for q in gate.qargs:
                    self._latency_dict[q] = max_latency
        return self._latency_dict
    
    def get_depth_without_one_qubit_gate(self):
        qc_empty = QuantumCircuit(*self._circuit.qregs, *self._circuit.cregs)
        for qc_data in self._circuit.data:
            if qc_data.operation.num_qubits == 2:
                import qiskit
                try:
                    qc_empty.data.append(qc_data)
                except qiskit.circuit.exceptions.CircuitError as e:
                    iprint(1)
                    pass
        # iprint(qc_empty.draw())
        return qc_empty.depth()

if __name__ == '__main__':
    # 创建一个量子电路
    qc = QuantumCircuit(4)
    qc.rz(1, 0)
    qc.rz(1, 1)
    qc.ecr(1, 0)
    qc.ecr(0, 1)
    # # 创建 Feature 实例，并传入 FakeQuebec 后端对象
    feature = Feature(qc, FakeQuebec())
    print(f'width: {feature.width}')
    print(f'depth: {feature.depth}')
    print(f'one_qubit_gates: {feature.one_qubit_gates}')
    print(f'two_qubit_gates: {feature.two_qubit_gates}')
    print(f'num_one_qubit_gates: {feature.num_one_qubit_gates}')
    print(f'num_two_qubit_gates: {feature.num_two_qubit_gates}')
    print(f'size: {feature.size}')
    print(f'latency_all(): {feature.latency_all()}')
    print(f'get_depth_without_one_qubit_gate(): {feature.get_depth_without_one_qubit_gate()}')

    # def qubit_utilizztion(self):
    #     ### caculate the latency of whole circuits
    #     gate_latency = {q: 0 for q in self._qargs}

    #     total_latency = sum(self._latency.values())
    #     if total_latency == 0:
    #         return 0.0  # 处理分母为零的情况
        
    #     for gate in self._dagcircuit.topological_op_nodes():
    #         q_indexes = [q._index for q in gate.qargs]
    #         for q in gate.qargs:
    #             gate_latency[q] += self._metric.latency(gate.op.name, q_indexes)
    #     return sum(gate_latency.values()) / total_latency

    # @property
    # def _error(self):
    #     error_dict = {q: 0 for q in self._qargs}

    #     for gate in self._dagcircuit.topological_op_nodes():
    #         q_indexes = [q._index for q in gate.qargs]
    #         # error = self._metric.gate_error(gate.op.name,q_indexes)
    #         error = 0
    #         for q in gate.qargs:
    #             error_dict[q]+= error
    #     return error_dict
    # def fidelity(self):
    #     return prod([1-e for e in self._error.values()])