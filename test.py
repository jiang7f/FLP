from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit_ibm_runtime.fake_provider import FakeProvider
import time

ibm_token = ''
ibm_cloud_api = ''
ibm_cloud_crn = ''

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
