import numpy as np


def set_print_form(type):
    if type == 1:
        np.set_printoptions(formatter={'float': lambda x: f'{x:2.0f},'})
    elif type == 2:
        np.set_printoptions(formatter={'float': lambda x: f'{x:4.1f},'})


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


def remove_zero_rows(matrix):
    non_zero_rows = np.any(matrix, axis=1)
    return matrix[non_zero_rows]


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

if __name__ == '__main__':
    constraint = np.array([[-1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                     [-1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                     [0, -1, 0, 1, 0, 0, 0, 1, 0, 0],
                     [0, -1, 0, 0, 0, 1, 0, 0, 0, 1],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1, 1, 0, 0, 0, 0]])
    basic_solution = find_basic_solution(constraint)
    print(basic_solution)
