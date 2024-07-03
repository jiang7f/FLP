from typing import Iterable,Union,Tuple
from qiskit_ibm_runtime.fake_provider import FakeKyoto, FakeKyiv, FakeSherbrooke, FakeQuebec, FakeAlmadenV2, FakeBelem, FakeSantiago
class Metric:
    def __init__(self,backend= FakeQuebec()) -> None:
        self.backend = backend
        print(FakeQuebec().qubit_properties('cx',(0,1)))

        print(FakeQuebec.target)
        
        exit()
        self.basis_gates = self.backend.configuration().basis_gates
        self.properties = self.backend.properties()
        pass
    def latency(self,gate:str,qubit:Union[int,Iterable[int]]):
        if gate not in self.basis_gates:
            raise ValueError("the input gate is not in basis")
        return self.properties.gate_length(gate, qubit)

    def gate_error(self,gate:str,qubit:Union[int,Iterable[int]]):
        if gate not in self.basis_gates:
            raise ValueError("the input gate is not in basis")
        return self.properties.gate_error(gate, qubit)
    def measurement_error(self,qubit:Union[int,Iterable[int]]):
        return self.properties.readout_error(qubit)
    def decoherenceTime(self,qubit:int):
        return min(self.properties.t1(qubit),self.properties.t2(qubit))

## test code
if  __name__ =='__main__':
    print('test code')
    print(Metric().latency('cx',(0,1)))
    print(Metric().gate_error('cx',(0,1)))
    print(Metric().measurement_error(2))
    print(Metric().decoherenceTime(4))