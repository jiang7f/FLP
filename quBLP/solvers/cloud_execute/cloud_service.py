from qiskit_ibm_runtime import QiskitRuntimeService
import os
current_dir = os.path.dirname(__file__)

def get_IBM_service():
    with open(os.path.join(current_dir, 'IBM.key'), 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith('ibm_token'):
                ibm_token = line.split('=')[1].strip().strip("'")
            elif line.startswith('ibm_cloud_api'):
                ibm_cloud_api = line.split('=')[1].strip().strip("'")
            elif line.startswith('ibm_cloud_crn'):
                ibm_cloud_crn = line.split('=')[1].strip().strip("'")
    service = QiskitRuntimeService(channel='ibm_cloud', token=ibm_cloud_api, instance=ibm_cloud_crn)
    print(f'IBM service created successfully')
    return service
