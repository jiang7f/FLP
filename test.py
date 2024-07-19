from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit_ibm_runtime.fake_provider import FakeProvider
import time

ibm_token = '69c20e9ce7c6ee081fcc3551f3b86f9ba140c5403e886fd6d6c8e3ffd0ba0f77c9d2dd1806927e2d01d54d7c405e600f11f0592bf91e8497fba99bf4ea75c17a'
ibm_cloud_api = 'lCgcnWBDUtjJvAfcgAOuuxA10t-bCMoHk46w-Vl86VYe'
ibm_cloud_crn = 'crn:v1:bluemix:public:quantum-computing:us-east:a/86a15c49d1e344948963063dc3912d24:cadd5163-70d1-4435-a7cb-ae18b370f839::'

# service = QiskitRuntimeService(channel='ibm_quantum', token=ibm_token)
service = QiskitRuntimeService(channel='ibm_cloud', token=ibm_cloud_api, instance=ibm_cloud_crn)
backend = service.backend("ibm_osaka")
# backend = FakeProvider()

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.circuit import ParameterVector

qc = QuantumCircuit(1)
qc.x(0)
qc.measure_all()

pm = generate_preset_pass_manager(backend=backend, optimization_level=0, initial_layout=[0])
isa_circuit = pm.run(qc)

print(isa_circuit.draw('text', idle_wires=False))

sampler = Sampler(backend=backend)
# sampler.options.default_shots = 100
job = sampler.run(isa_circuit, shots = 100)
print(f">>> Job ID: {job.job_id()}")

while not job.done():
    print(f">>> Job Status: {job.status()}")
    time.sleep(5)

result = job.result()
print(f">>> {result}")
print(f"  > Quasi-probability distribution: {result.quasi_dists[0]}")
print(f"  > Metadata: {result.metadata[0]}")
