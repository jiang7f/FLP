from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler
import time

ibm_token = '666e7d3d4fedd2ecfa60f1ec24658ad7217c5a4ad19999b6e21f17cedd5cfb5034256d52b6c6f5771a5a2337b9ee44db1c6043e4b7e3271149ffd2abd7193106'
ibm_cloud_api = ''
ibm_cloud_crn = ''

service = QiskitRuntimeService(channel='ibm_quantum', token=ibm_token)
# service = QiskitRuntimeService(channel='ibm_cloud', token=ibm_cloud_api, instance=ibm_cloud_crn)
print("service created successfully!")

backend = service.backend("ibm_kyiv")

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.circuit import ParameterVector

qc = QuantumCircuit(1)
qc.x(0)
qc.measure_all()

pm = generate_preset_pass_manager(backend=backend, optimization_level=0, initial_layout=[0])
isa_circuit = pm.run(qc)

print(isa_circuit.draw('text', idle_wires=False))

sampler = Sampler(mode=backend)
# sampler.options.default_shots = 100
job = sampler.run([isa_circuit], shots = 100)
print(f">>> Job ID: {job.job_id()}")

while not job.done():
    print(f">>> Job Status: {job.status()}")
    time.sleep(5)

result = job.result()
dist = result[0].data.meas.get_counts()
# meas: measure_all
# c: classical register
print(f">>> {result}")
print(f"  > Distribution: {dist}")
