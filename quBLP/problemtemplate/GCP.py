# wait to be done
import numpy as np
from typing import Iterable, List, Tuple
from ..models import ConstrainedBinaryOptimization
# 生成TSP问题约束矩阵
class GraphColoringProblem(ConstrainedBinaryOptimization):
    """ a `graph coloring problem`, is defined as
        
    """
    def __init__(self, num_graphs: int, edge_pairs: List[Tuple[int, int]]) -> None:
        """ a facility location problem

        Args:
            num_graphs (int): number of graphs
            edge_pairs (List[Tuple[int, int]]): c[i] : the list     for demand i to facility j
        """
        super().__init__(fastsolve=True)
        ## 图个数
        self.num_graphs = num_graphs
        ## 相邻图对
        self.edge_pairs = edge_pairs

        self.X = [[self.add_binary_variable('x'+str(i)+str(j)) for j in range(num_graphs)] for i in range(num_graphs)]

        self.objective = self.objectivefunc()
        self.feasible_solution = self.get_feasible_solution()
        pass
    @property
    def linear_constraints(self):
        n = self.n
        m = self.m
        total_columns = n + 2 * (m * n)
        matrix = np.zeros((n * m + m, total_columns))
        for j in range(n):
            for i in range(m):
                matrix[j * m + i, j] = -1
                matrix[j * m + i, n + i * n + j] = 1
                matrix[j * m + i, n + n * m + i * n + j] = 1
                matrix[n * m + i, n + n * i + j] = 1
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
        for var in self.X:
            var.set_value(1)
        for i in range(self.m):
            for var in self.Y[i]:
                var.set_value(0)
            self.Y[i][0].set_value(1)
        for i in range(self.m):
            for j in range(self.n):
                self.Z[i][j].set_value(-self.Y[i][j].x + self.X[j].x)
        
        return np.nonzero([x.x for x in self.variables])[0]

    def objectivefunc(self):
        def objective(variables:Iterable):
            """ the objective function of the facility location problem

            Args:
                variables (Iterable):  the list of value of variables

            Returns:
                scaler
            """
            cost = 0
            for i in range(self.m):
                for j in range(self.n):
                    cost += self.c[i][j] * variables[self.n * (1 + i) + j]
            for j in range(self.n):
                cost += self.f[j] * variables[j]
            return cost
        return objective

        