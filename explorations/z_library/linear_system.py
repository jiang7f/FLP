import numpy as np

# 设置numpy输出格式
def set_print_form(suppress=True, precision=4, linewidth=300):
    # 不要截断 是否使用科学计数法 输出浮点数位数 宽度
    np.set_printoptions(threshold=np.inf, suppress=suppress, precision=precision,  linewidth=linewidth)

# 把矩阵转换成行阶梯矩阵
def to_row_echelon_form(orimatrix: np.array):
    """Convert a matrix to row echelon form."""
    matrix = orimatrix.copy()
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
    return np.array(basic_solutions)


# 生成FLP问题约束矩阵
def gnrt_cstt(n, m):
    total_columns = n + 2 * (m * n)
    matrix = np.zeros((n * m + m, total_columns))
    for j in range(n):
        for i in range(m):
            matrix[j * m + i, j] = -1
            matrix[j * m + i, n + i * n + j] = 1
            matrix[j * m + i, n + n * m + i * n + j] = 1
            matrix[n * m + i, n + n * i + j] = 1
    return matrix


def gnrt_u(n, m):
  # 自由变量个数
  row = n * m - m + n
  column = 2 * m * n + n
  matrix = np.zeros((row, column))
  # p1
  for i in range(n - 1):
    matrix[i, 0] = -1
    matrix[i, i + 1] = 1
    for j in range(m):
      matrix[i, n + j * n] = -1
      matrix[i, n + j * n + i + 1] = 1
  # p2
  for j in range(m - 1):
    for i in range(n - 1):
      matrix[n - 1 + j * (n - 1) + i, n + j * n] = 1
      matrix[n - 1 + j * (n - 1) + i, n + j * n + i + 1] = -1
      matrix[n - 1 + j * (n - 1) + i, n + m * n + j * n] = -1
      matrix[n - 1 + j * (n - 1) + i, n + m * n + j * n + i + 1] = 1
  # p3
  matrix[n - 1 + (m - 1) * (n - 1), 0]= 1
  for j in range(m):
    matrix[n - 1 + (m - 1) * (n - 1), n + n * m  + n * j]= 1
  # p4
  for i in range(n - 1):
    matrix[n - 1 + (m - 1) * (n - 1) + i + 1, i + 1] = 1
    matrix[n - 1 + (m - 1) * (n - 1) + i + 1, n + n * m + n * (m - 1) + i + 1] = 1
    for j in range(m - 1):
      matrix[n - 1 + (m - 1) * (n - 1) + i + 1, n + j * n] = -1
      matrix[n - 1 + (m - 1) * (n - 1) + i + 1, n + j * n + i + 1] = 1
      matrix[n - 1 + (m - 1) * (n - 1) + i + 1, n  + n * m + j * n] = 1
  return matrix


# 输出非零元素索引
def find_nonzero_indices(matrix):
    nonzero_indices = []
    for row in matrix:
        row_indices = [i for i, x in enumerate(row) if x != 0]
        nonzero_indices.append(row_indices)
    return nonzero_indices


if __name__ == '__main__':
    # 设置输出格式 单行最大长度200
    set_print_form()
    # generate constraint(n = 设施数量, m = 需求数量)
    cstt = gnrt_cstt(2, 2)
    print(to_row_echelon_form(cstt))
    # 求解基础解系
    print(find_basic_solution(cstt))