import numpy as np

# 张量积逆序
def reorder_tensor_product(matrix):
  dim = matrix.shape[0]
  n = int(np.log2(dim))
  m = matrix.reshape([2]*(2 * n))
  m = np.transpose(m, axes=(list(range(n - 1, -1, -1)) + list(range(2 * n - 1, n - 1, -1))))
  m = m.reshape(2**n, 2**n)
  return m

# 输入qc, 返回电路酉矩阵 (张量积逆序)
def get_circ_unitary(quantum_circuit):
    from qiskit_aer import Aer
    from qiskit import transpile
    backend = Aer.get_backend('unitary_simulator' )
    new_circuit = transpile(quantum_circuit, backend)
    job = backend.run(new_circuit)
    result = job.result()
    unitary = result.get_unitary()
    reoder_unitary = reorder_tensor_product(np.array(unitary))
    return reoder_unitary

# 把矩阵转换成行阶梯矩阵
def to_row_echelon_form(orimatrix: np.array):
    """Convert a matrix to row echelon form."""
    matrix = orimatrix.copy()
    if len(matrix) == 0:
        return matrix
    num_rows, num_cols = matrix.shape
    lead = 0
    for r in range(num_rows):
        if lead >= num_cols:
            return matrix

        i = r
        while matrix[i, lead] == 0:
            i += 1
            if i == num_rows:
                i = r
                lead += 1
                if num_cols == lead:
                    return matrix

        matrix[[i, r]] = matrix[[r, i]]

        lv = matrix[r, lead]
        matrix[r] = matrix[r] / lv

        for i in range(num_rows):
            if i != r:
                lv = matrix[i, lead]
                matrix[i] = matrix[i] - lv * matrix[r]
        lead += 1
    return matrix
# 去除底部全0行
def remove_zero_rows(matrix):
    non_zero_rows = np.any(matrix, axis=1)
    return matrix[non_zero_rows]
# 返回主元和自由变量索引
def find_free_variables(matrix):
    num_rows, num_cols = matrix.shape
    pivot_columns = set()  # 主元列的集合
    free_columns = []      # 自由列的列表
    # 找到主元列
    for row in matrix:
        for j, elem in enumerate(row):
            if elem != 0:
                pivot_columns.add(j)
                break
    # 找到自由列
    for j in range(num_cols):
        if j not in pivot_columns:
            free_columns.append(j)
    return list(pivot_columns), free_columns
# 求基础解析
def find_basic_solution(matrix):
    matrix = to_row_echelon_form(matrix)
    matrix = remove_zero_rows(matrix)
    pivot_columns, free_variables = find_free_variables(matrix)
    # 计算基础解系
    basic_solutions = []
    for index in free_variables:
        # 使用 np.linalg.solve 求解特解
        x = np.linalg.solve(matrix[:, pivot_columns], -matrix[:, index])
        solution = np.zeros(matrix.shape[1])
        solution[index] = 1
        solution[pivot_columns] = x
        basic_solutions.append(solution)
    ## 通过加或者减 实现基础解系的每个解在（-1，0，1）中  
    return np.array(basic_solutions)
def find_nonzero_indices(u):
    nonzero_indices = []
    for urow in u:
        nonzero_indices.append(np.nonzero(urow)[0])
    return nonzero_indices