from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler, Options
from qiskit_aer.noise import NoiseModel
from copy import deepcopy
import os
_DEFAULT_ACCOUNT_CONFIG_JSON_FILE = os.path.join(
    os.path.expanduser("~"), ".qiskit", "qiskit-ibm.json"
)
_DEFAULT_ACCOUNT_NAME = "default"

class IBMCloudRun:
    def __init__(self,shots= 4000,token=None) -> None:
        if not os.path.exists(_DEFAULT_ACCOUNT_CONFIG_JSON_FILE):
            if token:
                QiskitRuntimeService.save_account(channel="ibm_quantum", token=token)
            else:
                token = input("Please input your IBM Quantum token:\n")
                QiskitRuntimeService.save_account(channel="ibm_quantum", token=token)
        
        self.service = QiskitRuntimeService(channel="ibm_quantum")
        self.options = Options()
        self.options.optimization_level = 2
        self.options.resilience_level = 0
        self.options.shots = shots
    def get_backend(self,backendname):
        return self.service.get_backend(backendname)
    def replace_token(self,token):
        self.service = QiskitRuntimeService(channel="ibm_quantum",token=token)
    def get_idle_backend(self):
        backends = sorted(self.service.backends(simulator=False, operational=True, status_msg="active"), key=lambda x: x.status().pending_jobs)
        return backends[0]
    def simulate(self,circuit,noise_device= None,evmode = "ibmq_qasm_simulator",seed=42):
        options = deepcopy(self.options)
        if noise_device:
            if noise_device == 'idle':
                fake_backend = self.get_idle_backend()
                noise_model = NoiseModel.from_backend(fake_backend)
            else:
                fake_backend = self.service.get_backend(noise_device)
                noise_model = NoiseModel.from_backend(fake_backend)
            
            options.simulator = {
                "noise_model": noise_model,
                "basis_gates": fake_backend.configuration().basis_gates,
                "coupling_map": fake_backend.configuration().coupling_map,
                "seed_simulator": seed
            }
        backend = self.service.get_backend(evmode)
        with Session(service=self.service, backend=backend) as session:
            sampler = Sampler(session=session, options=options)
            job = sampler.run(circuit)
        return self.get_prob_vector(job,circuit.num_qubits),job._job_id
    
    def systemRun(self,circuit,backendname=None):
        if backendname:
            backend = self.service.get_backend(backendname)
        else:
            backend = self.get_idle_backend()
        ## compile the circuit
        from qiskit.compiler import transpile
        circuit = transpile(circuit,backend)
        with Session(service=self.service, backend=backend) as session:
            sampler = Sampler(session=session, options=self.options)
            job = sampler.run(circuit)
        return self.get_prob_vector(job,circuit.num_qubits),job._job_id

    def get_prob_vector(self,job,num_qubits):
        result = job.result()
        import numpy as np
        res = np.zeros(2**num_qubits)
        for k in result.quasi_dists[0]:
            res[k] = result.quasi_dists[0][k]
        return res

if __name__ == "__main__":
    from qiskit import QuantumCircuit
    import numpy as np
    ## take a simple example circuit
    circuit = QuantumCircuit(2)
    circuit.h(1)
    circuit.u(0.1,0.2,0.3,0)
    circuit.cx(0,1)
    circuit.rx(np.pi/2,1)
    circuit.measure_all()
    ## test code
    print('test code')
    cloud = IBMCloudRun()
    # print(cloud.systemRun(circuit))
