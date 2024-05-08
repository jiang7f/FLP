# pending
import numpy as np
from typing import Iterable
from ..models import ConstrainedBinaryOptimization
# 生成TSP问题约束矩阵
class MaximumCliqueProblem(ConstrainedBinaryOptimization):
    """ a `facility location problem`, is defined as
    .. math::
        min \sum_{i=1}^m \sum_{j=1}^n c_{i j} y_{i j}+\sum_{j=1}^n f_j x_j 
    
    .. math::
        s.t. \quad \sum_{j=1}^n y_{i j}=1, \quad i=1,2, \cdots, m 
    .. math::
        y_{i j}+z_{i j}-x_j=0, \quad i=1, \cdots, m, j=1, \cdots, n
    .. math::
        z_{i j}, y_{i j}, x_j \in\{0,1\} 

    """
    def __init__(self, n: int, m: int, c: Iterable[Iterable],f: Iterable) -> None:
        """ a facility location problem

        Args:
            n (int): number of facility
            m (int): number of demand point
            c (Matrix[m,n]): c_{i,j} : the cost for demand i to facility j
            f (Vector[n]): f_j: the building cost for facility j
        """
        super().__init__(fastsolve=True)
        ## 设施点个数
        self.n = n
        ## 需求点个数
        self.m = m
        # cij需求i到设施j的成本
        self.c = c 
        # fj设施j的建设成本
        self.f = f
        self.num_variables = n + 2 * n * m

        self.X = [self.add_binary_variable('x'+str(i)) for i in range(n)]
        self.Y = [[self.add_binary_variable('y'+str(i)+str(j)) for j in range(n)] for i in range(m)]
        self.Z = [[self.add_binary_variable('z'+str(i)+str(j)) for j in range(n)] for i in range(m)]

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

        