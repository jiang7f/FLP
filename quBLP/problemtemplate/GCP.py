# pending
import numpy as np
from typing import Iterable, List, Tuple
from ..models import ConstrainedBinaryOptimization
# 生成TSP问题约束矩阵
class GraphColoringProblem(ConstrainedBinaryOptimization):
    """ a `graph coloring problem`, is defined as
        
    """
    def __init__(self, num_graphs: int, adjacent_pairs: List[Tuple[int, int]], fastsolve=False) -> None:
        """ a facility location problem

        Args:
            num_graphs (int): number of graphs
            adjacent_pairs (List[Tuple[int, int]]): c[i] : the list     for demand i to facility j
        """
        super().__init__(fastsolve)
        ## 图个数
        self.num_graphs = num_graphs
        ## 相邻图对
        self.adjacent_pairs = adjacent_pairs
        # 最坏情况每个图一个颜色
        num_colors = num_graphs
        # x_i_k = 1 如果顶点 i 被分配颜色 k, 否则为0
        self.X = self.add_binary_variable('x', [num_graphs, num_colors])
        self.Y = self.add_binary_variable('y', [num_graphs, num_graphs, num_colors])
        self.objective = self.objectivefunc()
        self.feasible_solution = self.get_feasible_solution()
        pass
    @property
    def linear_constraints(self):
        n = self.n
        m = self.m
        total_columns = n + 2 * (m * n)
        matrix = np.zeros((n * m + m, total_columns))

        return matrix
    
    def fast_solve_driver_bitstr(self):
        n, m  = self.n,self.m
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
    
    def get_feasible_solution(self):
        """ 根据约束寻找到一个可行解
        """
        # graph_i color color_i
        for i in self.num_graphs:
            self.X[i][i].set_value(1)
        for a, b in self.adjacent_pairs:
            self.Y[a][b][a].set_value(1)
            self.Y[a][b][b].set_value(1)
        return np.nonzero([var.x for var in self.variables])[0]

    def objectivefunc(self):
        def objective(variables:Iterable):
            """ the objective function of the facility location problem

            Args:
                variables (Iterable):  the list of value of variables

            Return:
                cost
            """
            cost = 0
            for j in range(self.n):
                if 1 in [self.X[i][j].x for i in range(self.m)]:
                    cost += 1
            return cost
        return objective

        