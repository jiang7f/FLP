from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
from functools import reduce

import numpy as np
from scipy.optimize import minimize

import sys
sys.path.append('../../')
import z_library.linear_system as ls
import z_library.extension as extn
extn.output_to_file_init("start")

# dij需求i到设施j的成本
d = [[1, 2], [3, 4], [5, 6]]
n = 2   # 两个设施点
m = 3   # 三个需求点
num_qubits = n + 2 * n * m

# gi设施i的建设成本
g = [2, 1]

GateX = np.array([[0, 1],[1, 0]])
GateY = np.array([[0, -1j],[1j, 0]])
GateZ = np.array([[1, 0],[0, -1]])

# 定义σ+和σ-矩阵
sigma_plus = np.array([[0, 1], [0, 0]])
sigma_minus = np.array([[0, 0], [1, 0]])

def add_in_target(num_qubits, target_qubit, gate=np.array([[1, 0],[0, -1]])):
    H = np.eye(2 ** (target_qubit))
    H = np.kron(H, gate)
    H = np.kron(H, np.eye(2 ** (num_qubits - 1 - target_qubit)))
    return H

def calculate_hamiltonian(v, w):
    n = len(v[0])
    m = len(v)
    hamiltonian = np.zeros((2**n, 2**n))

    for i in range(m):
        term1 = reduce(np.kron, [np.linalg.matrix_power(sigma_plus, v[i][j]) for j in range(n)])
        term2 = reduce(np.kron, [np.linalg.matrix_power(sigma_minus, w[i][j]) for j in range(n)])
        term3 = reduce(np.kron, [np.linalg.matrix_power(sigma_plus, w[i][j]) for j in range(n)])
        term4 = reduce(np.kron, [np.linalg.matrix_power(sigma_minus, v[i][j]) for j in range(n)])

        hamiltonian += term1 @ term2 + term3 @ term4

    return hamiltonian

def first_nonzero_index(arr, total_bits=3):
    for i, num in enumerate(arr):
        if num != 0:
            binary_repr = format(i, '0' + str(total_bits) + 'b')
            return binary_repr

def generate_Hp(n, m, d, g):
    # 初始化 Hp 矩阵为零矩阵
    # print(num_qubits)
    Hp = np.zeros((2**num_qubits, 2**num_qubits))
    for i in range(m):
        for j in range(n):
            Hp += d[i][j] * (add_in_target(num_qubits, n * (1 + i) + j) - np.eye(2**num_qubits)) / 2
    
    for j in range(n):
        Hp +=  g[j] * (add_in_target(num_qubits, j)- np.eye(2**num_qubits)) / 2
    return Hp
        
Hp = generate_Hp(n, m, d, g)

import sys
sys.path.append('../../')
import z_library.linear_system as ls
Cons = np.array([[-1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
              [-1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
              [0, -1, 0, 1, 0, 0, 0, 1, 0, 0],
              [0, -1, 0, 0, 0, 1, 0, 0, 0, 1],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 1, 0, 0, 0, 0]])
u = ls.find_basic_solution(Cons)
v = np.where(u == 1, 1, 0)
w = np.where(u == -1, 1, 0)
Hd = calculate_hamiltonian(v, w)

from scipy.linalg import expm
def build_circ(params):
  depth = len(params) // 2
  qc = QuantumCircuit(num_qubits)
  beta = params[:depth]
  gamma = params[depth:]
  for i in [1,3,4,6,8,9]:
    qc.x(i)
  for dp in range(depth):
    qc.unitary(expm(-1j * beta[dp] * Hp), range(num_qubits)) # transpile
    qc.unitary(expm(-1j * gamma[dp] * Hd), range(num_qubits))
  qc.measure_all()
  return qc

def cost_function(x):
  num = [int(char) for char in x]
  C = 0
  for i in range(m):
    for j in range(n):
      C += d[i][j] * num[n * (1 + i) + j]
      
  for j in range(n):
    C += g[j] * num[j]
  return C

def compute_expectation(counts):
  EV = 0
  total_count = 0
  for x, count in counts.items():
    C = cost_function(x)
    EV += C*count
    total_count += count

  return EV/total_count


def expectation_from_sample(shots = 2000):
  backend = Aer.get_backend('qasm_simulator')
  backend.shots = shots

  def execute_circ(theta):
    qc = build_circ(theta)
    counts = backend.run(qc, seed_simulator=10, shots=shots).result().get_counts()
    return compute_expectation(counts)
  
  return execute_circ

from numpy.lib.utils import source
from scipy.optimize import minimize
import numpy as np
# 初始化迭代计数器
iteration_count = 0
def test(par):
  global iteration_count
  iteration_count = 0
  expectation = expectation_from_sample()
  def callback(x):
      global iteration_count
      iteration_count += 1
      if iteration_count % 10 == 0:
          print(f"Iteration {iteration_count}, Result: {expectation(x)}")
  # 设定最大迭代次数
  max_iterations = 1000

  # 使用 COBYLA 方法进行最小化，并设置 callback 函数
  res = minimize(expectation, par, method='COBYLA', options={'maxiter': max_iterations}, callback=callback)
  # 输出最终结果
  print("Final Result:", res)
  backend = Aer.get_backend('aer_simulator')
  backend.shots = 100000

  shots=100000
  qc_res = build_circ(params=res.x)

  counts = backend.run(qc_res, seed_simulator=10, shots = shots).result().get_counts()
  # plot_histogram(counts)
  sorted_counts = sorted(counts, key=counts.get, reverse=True)
  print("\n----------------- Full result ---------------------")
  print("selection\t\tprobability\tvalue")
  print("---------------------------------------------------")
  for x in sorted_counts[:20]:
    print(x, "{:.2f}%".format(counts[x] / shots * 100), cost_function(x))

for dep in range(1,10):
  print(f'dep=={dep}')
  test(np.full(dep * 2, np.pi/3, dtype=np.float128))
  print()

extn.output_to_file_reset("flp23-worst finished")