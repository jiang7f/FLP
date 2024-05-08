# wait to be done
# This file encapsulates the our method based on QAOA+
import matplotlib.pyplot as plt
from typing import List
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
from scipy.optimize import minimize
from scipy.linalg import expm
from quBLP.utils import linear_system as ls
import numpy as np

# dij需求i到设施j的成本
d = [[1, 2], [1, 2]]
n = 2   # 两个设施点
m = 2   # 两个需求点
# d = [[1, 2], [3, 4], [5, 6]]
# n = 2   # 两个设施点
# m = 3   # 三个需求点
num_qubits = n + 2 * n * m
# gi设施i的建设成本
g = [2, 1]

GateX = np.array([[0, 1],[1, 0]])
GateY = np.array([[0, -1j],[1j, 0]])
GateZ = np.array([[1, 0],[0, -1]])

# 定义σ+和σ-矩阵
sigma_plus = np.array([[0, 1], [0, 0]])
sigma_minus = np.array([[0, 0], [1, 0]])

iteration_count = 0

# def add_in_target(num_qubits, target_qubit, gate=np.array([[1, 0],[0, -1]])):
# 	H = np.eye(2 ** (target_qubit))
# 	H = np.kron(H, gate)
# 	H = np.kron(H, np.eye(2 ** (num_qubits - 1 - target_qubit)))
# 	return H

# def generate_Hp(n, m, d, g):
# 	# 初始化 Hp 矩阵为零矩阵
# 	Hp = np.zeros((2**num_qubits, 2**num_qubits))
# 	for i in range(m):
# 		for j in range(n):
# 			Hp += d[i][j] * (add_in_target(num_qubits, n * (1 + i) + j) - np.eye(2**num_qubits)) / 2
  
# 	for j in range(n):
# 		Hp +=  g[j] * (add_in_target(num_qubits, j)- np.eye(2**num_qubits)) / 2
# 	return Hp

# Hp = generate_Hp(n, m, d, g)

# def gnrt_gate_hdi(u):
# 	result = []
# 	nonzero_indices = ls.find_nonzero_indices(u)
# 	for urow, nrow in zip(u, nonzero_indices):
# 		# 把非0元素映射成01
# 		filtered_arr = [0 if x == -1 else 1 for x in urow if x != 0]
# 		binary_str = ''.join(map(str, filtered_arr))
# 		# 取到二进制表示的数
# 		map_num = int(binary_str, 2)
# 		# print(map_num)
# 		length = len(nrow)
# 		scale = 2**length
# 		matrix = np.zeros((scale, scale))

# 		matrix[map_num, scale - 1 - map_num] = 1
# 		map_num = 2**length - 1 - map_num
# 		# print(map_num)
# 		matrix[map_num, scale - 1 - map_num] = 1
# 		result.append(matrix)
# 	return result

# u = ls.gnrt_u(n, m)
# print(u)
# nonzero_indices = ls.find_nonzero_indices(u)
# gate_hds = gnrt_gate_hdi(u)
# print(nonzero_indices)

# def build_circ(params):
#   depth = len(params) // 2
#   qc = QuantumCircuit(num_qubits)
#   beta = params[:depth]
#   gamma = params[depth:]
#   for i in [1,3,4,6,8,9]:
#     qc.x(i)
#   for dp in range(depth):
#     qc.unitary(expm(-1j * beta[dp] * Hp), range(num_qubits))
#     for gate_hdi, ind in zip(gate_hds, nonzero_indices):
#       qc.unitary(expm(-1j * gamma[dp] * gate_hdi), (num_qubits - 1 - i for i in ind))
#   qc.measure_all()
#   return qc

# def cost_function(x):
#   num = [int(char) for char in x]
#   C = 0
#   for i in range(m):
#     for j in range(n):
#       C += d[i][j] * num[n * (1 + i) + j]
      
#   for j in range(n):
#     C += g[j] * num[j]
#   return C

# def compute_expectation(counts):
#   EV = 0
#   total_count = 0
#   for x, count in counts.items():
#     C = cost_function(x)
#     EV += C*count
#     total_count += count

#   return EV/total_count


# def expectation_from_sample(shots = 2000):
#   backend = Aer.get_backend('qasm_simulator')
#   backend.shots = shots

#   def execute_circ(theta):
#     qc = build_circ(theta)
#     counts = backend.run(qc, seed_simulator=10, shots=shots).result().get_counts()
#     return compute_expectation(counts)
  
#   return execute_circ

# # 初始化迭代计数器
# iteration_count = 0
# def test(par):
#   global iteration_count
#   iteration_count = 0
#   expectation = expectation_from_sample()
#   def callback(x):
#       global iteration_count
#       iteration_count += 1
#       if iteration_count % 10 == 0:
#           print(f"Iteration {iteration_count}, Result: {expectation(x)}")
#   # 设定最大迭代次数
#   max_iterations = 1000

#   # 使用 COBYLA 方法进行最小化，并设置 callback 函数
#   res = minimize(expectation, par, method='COBYLA', options={'maxiter': max_iterations}, callback=callback)
#   # 输出最终结果
#   print("Final Result:", res)
#   backend = Aer.get_backend('aer_simulator')
#   backend.shots = 100000

#   shots=100000
#   qc_res = build_circ(params=res.x)

#   counts = backend.run(qc_res, seed_simulator=10, shots = shots).result().get_counts()
#   # plot_histogram(counts)
#   sorted_counts = sorted(counts, key=counts.get, reverse=True)
#   print("\n----------------- Full result ---------------------")
#   print("selection\t\tprobability\tvalue")
#   print("---------------------------------------------------")
#   for x in sorted_counts[:20]:
#     print(x, "{:.2f}%".format(counts[x] / shots * 100), cost_function(x))

# def QAOAplus(par):
# 	global iteration_count
# 	# 初始化迭代计数器
# 	iteration_count = 0
# 	expectation = expectation_from_sample()
# 	def callback(x):
# 			global iteration_count
# 			iteration_count += 1
# 			if iteration_count % 10 == 0:
# 					print(f"Iteration {iteration_count}, Result: {expectation(x)}")
# 	# 设定最大迭代次数
# 	max_iterations = 1000

# 	# 使用 COBYLA 方法进行最小化，并设置 callback 函数
# 	res = minimize(expectation, par, method='COBYLA', options={'maxiter': max_iterations}, callback=callback)
# 	# 输出最终结果
# 	print("Final Result:", res)
# 	backend = Aer.get_backend('aer_simulator')
# 	backend.shots = 100000

# 	shots=100000
# 	qc_res = build_circ(params=res.x)

# 	counts = backend.run(qc_res, seed_simulator=10, shots = shots).result().get_counts()
# 	# plot_histogram(counts)
# 	sorted_counts = sorted(counts, key=counts.get, reverse=True)
# 	print("\n----------------- Full result ---------------------")
# 	print("selection\t\tprobability\tvalue")
# 	print("---------------------------------------------------")
# 	for x in sorted_counts[:20]:
# 		print(x, "{:.2f}%".format(counts[x] / shots * 100), cost_function(x))
# QAOAplus(np.full(6, np.pi/3))
if __name__ == '__main__':
  # dij需求i到设施j的成本
  d = [[1, 2], [1, 2]]
  n = 2   # 两个设施点
  m = 2   # 两个需求点
  # d = [[1, 2], [3, 4], [5, 6]]
  # n = 2   # 两个设施点
  # m = 3   # 三个需求点
  num_qubits = n + 2 * n * m
  # gi设施i的建设成本
  g = [2, 1]