from quBLP.solvers.cloud_execute import cloud_service
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime.fake_provider import FakeKyiv, FakeTorino, FakeBrisbane
import time
import random
from multiprocessing import Process, Queue, current_process, Manager

class CloudManager:
    def __init__(self, job_dic, results, one_job_lens, sleep_interval=5, use_free=True) -> None:
        self.job_dic = job_dic
        self.results = results
        self.one_job_lens = one_job_lens
        self.use_free = use_free
        self.sleep_interval = sleep_interval
        self.lock_result = Manager().Lock()
        self.lock_IBM_run = Manager().Lock()
        self.lock_job_lens = Manager().Lock()
        self.lock_job_dic = Manager().Lock()

    def submit_task(self, backend_shots, circuit):
        task_id = id((backend_shots, circuit))
        # 会有相同参数的电路应该直接返回，只有无记录(return None)的电路才算
        if self.get_counts(task_id) is None:
            print(f"{backend_shots} submitted")
            self.job_dic[backend_shots].put((task_id,circuit))
        else:
            print(f"{task_id} circuit has runed")
        return task_id

    def one_optimization_finished(self):
        with self.lock_job_lens:
            self.one_job_lens.value -= 1



    def process_task(self, key):
        time.sleep(self.sleep_interval)  # 等待电路线程创建
        while True:
            with self.lock_job_lens:
                one_job_lens = self.one_job_lens.value
            # all optimization finished to break
            if one_job_lens == 0:
                break
            tasks = self.job_dic[key]
            print(f'{key} manager task size: {tasks.qsize()} / {one_job_lens}')
            if tasks.qsize() >= one_job_lens:
                try:
                    print(f"{key}, start to submit to IBM")
                    backend_name, shots= key
                    tasks_to_process = []
                    for _ in range(one_job_lens):
                        tasks_to_process.append(tasks.get())
                    # 同时提交似乎有问题
                    with self.lock_IBM_run:
                        if self.use_free is not None:
                            service = cloud_service.get_IBM_service(use_free=self.use_free, message = f"{key} manager IBM service created successful")
                            backend = service.backend(backend_name)
                        else:
                            backend = FakeKyiv()
                        sampler = Sampler(backend=backend)
                        task_ids = [task_id for task_id, _ in tasks_to_process]
                        circuits = [circuit for _, circuit in tasks_to_process]
                        job = sampler.run(circuits, shots=shots)
                    # while True:
                    #     print(self.circuit_id)
                    #     time.sleep(self.sleep_interval)
                    job_id = job.job_id()
                    while not job.done():
                        print(f'{job_id} status: {job.status()}')
                        time.sleep(self.sleep_interval)
                    # 已得到结果, 清空电路
                    print(f'{key, job_id} status: {job.status()}')
                    counts = [job.result()[i].data.c.get_counts() for i in range(one_job_lens)]
                    # print(counts)
                    # counts = [{'100011': 2} for _ in range(one_job_lens)]
                    with self.lock_result:
                        for i, task_id in enumerate(task_ids):
                            self.results[task_id] = counts[i]
                except Exception as e:
                    print('IBM submit error', e)
            time.sleep(self.sleep_interval)  # 避免忙碌等待
            
            
    def get_counts(self, task_id):
        # return {'101000': 92}
        with self.lock_result:
            counts = self.results.get(task_id, None)
        return counts