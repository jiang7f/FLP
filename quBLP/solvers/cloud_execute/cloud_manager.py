from quBLP.solvers.cloud_execute import cloud_service
from qiskit_ibm_runtime import SamplerV2 as Sampler
import time

class CloudManager:
    def __init__(self, job_dic, result_dic, job_id_dic, one_job_lens, sleep_interval=5) -> None:
        self.one_job_lens = one_job_lens
        self.sleep_interval = sleep_interval
        self.job_dic = job_dic
        self.result_dic = result_dic
        self.job_id_dic = job_id_dic

    # backend_shots 为 backend 和 shots 的元组
    def append_circuit(self, backend_shots, circuit):
        if backend_shots not in self.job_dic.keys():
            self.job_dic[backend_shots] = []
        self.job_dic[backend_shots].append(circuit)
        circuit_id = len(self.job_dic[backend_shots])
        return circuit_id

    def get_counts(self, backend_shots, circuit_id):
        self.run(backend_shots)
        return self.result_dic[backend_shots][circuit_id].data.c.get_counts()
        

    def run(self, backend_shots):
        num_circuit = len(self.job_dic[backend_shots])
        # 如果不该run：
        if num_circuit < self.one_job_lens:
            # 还没出结果, 阻塞
            print(f'{backend_shots} add one circuit, {num_circuit} / {self.one_job_lens}')
            while backend_shots not in self.result_dic.keys():
                time.sleep(self.sleep_interval)
        else:
            backend_name, shots = backend_shots
            service = cloud_service.get_IBM_service()
            backend = service.backend(backend_name)
            sampler = Sampler(backend=backend)
            # while True:
                # print("OuO!!!!!")
                # time.sleep(1)
            job = sampler.run(self.job_dic[backend_shots], shots=shots)
            job_id = job.job_id()
            self.job_id_dic[backend_shots] = job_id
            # 
            while not job.done:
                print(f'{job_id} status: {job.status()}')
                time.sleep(self.sleep_interval)
            self.result_dic[backend_shots] = job.result()
