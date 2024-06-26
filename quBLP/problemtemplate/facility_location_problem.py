import numpy as np
import itertools
from typing import Iterable
from ..models import ConstrainedBinaryOptimization
class FacilityLocationProblem(ConstrainedBinaryOptimization):
    """ a facility location problem is defined as
    .. math::
        min \sum_{i=1}^m \sum_{j=1}^n c_{i j} y_{i j}+\sum_{j=1}^n f_j x_j 
    
    .. math::
        s.t. \quad \sum_{j=1}^n y_{i j}=1, \quad i=1,2, \cdots, m 
    .. math::
        y_{i j}+z_{i j}-x_j=0, \quad i=1, \cdots, m, j=1, \cdots, n
    .. math::
        z_{i j}, y_{i j}, x_j \in\{0,1\} 

    """
    def __init__(self, n: int, m: int, c: Iterable[Iterable],f: Iterable, fastsolve=True) -> None:
        """ a facility location problem

        Args:
            n (int): number of facilities
            m (int): number of demand point
            c (Matrix[m,n]): c_{i,j} : the cost for demand i to facility j
            f (Vector[n]): f_j: the building cost for facility j
        """
        super().__init__(fastsolve)
        #* 设定优化方向
        self.set_optimization_direction('min')
        # 设施点个数
        self.n = n
        # 需求点个数
        self.m = m
        # cij需求i到设施j的成本
        self.c = c 
        # fj设施j的建设成本
        self.f = f
        self.num_variables = n + 2 * n * m
        self.X = self.add_binary_variables('x', [n])
        self.Y = self.add_binary_variables('y', [m, n])
        self.Z = self.add_binary_variables('z', [m, n])
        #* 添加支持优化方法
        self.objective_penalty = self.objective_func('penalty')
        self.objective_cyclic = self.objective_func('cyclic')
        self.objective_commute = self.objective_func('commute')
        self.feasible_solution = self.get_feasible_solution()
        #* 直接加目标函数表达式
        self._add_linear_objective(np.multiply(list(itertools.chain(f, *c)), self.cost_dir))
        #* 约束放到 self.constraints 里
        for cstrt in self.linear_constraints:
            self._add_linear_constraint(cstrt)
        pass

    @property
    def Ho_gate_list(self):
        theta_list = self.linear_objective_vector
        gate_list = [[[i], theta_list[i]] for i in range(len(theta_list))]
        return gate_list
    
    @property
    def linear_constraints(self):
        n = self.n
        m = self.m
        total_rows = n * m + m
        total_columns = self.num_variables + 1
        matrix = np.zeros((total_rows, total_columns))
        for i in range(m):
            for j in range(n):
                matrix[n * m + i, self.var_to_idex(self.Y[i][j])] = 1
                # y_ij + z_ij - x_j = 1
                matrix[j * m + i, self.var_to_idex(self.Y[i][j])] = 1
                matrix[j * m + i, self.var_to_idex(self.Z[i][j])] = 1
                matrix[j * m + i, self.var_to_idex(self.X[j])] = -1
            matrix[n * m + i, total_columns - 1] = 1
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
        # for j in range(self.n):
        self.X[0].set_value(1)
        for i in range(self.m):
            self.Y[i][0].set_value(1)
        # for i in range(self.m):
        #     for j in range(self.n):
        #         self.Z[i][j].set_value(-self.Y[i][j].x + self.X[j].x)
        return [x.x for x in self.variables]

    def objective_func(self, algorithm_optimization_method):
        def objective(variables:Iterable):
            cost = 0
            for i in range(self.m):
                for j in range(self.n):
                    cost += self.c[i][j] * variables[self.var_to_idex(self.Y[i][j])]
            for j in range(self.n):
                cost += self.f[j] * variables[self.var_to_idex(self.X[j])]
            # commute 只需要目标函数两项
            if algorithm_optimization_method == 'commute':
                return self.cost_dir * cost
            for i in range(self.m):
                for j in range(self.n):
                    cost += self.penalty_lambda * (variables[self.var_to_idex(self.Y[i][j])] + variables[self.var_to_idex(self.Z[i][j])] - variables[self.var_to_idex(self.X[j])])**2
            # cyclic 多包含一项非∑=x约束
            if algorithm_optimization_method == 'cyclic':
                return self.cost_dir * cost
            for i in range(self.m):
                t = 0
                for j in range(self.n):
                    t += variables[self.var_to_idex(self.Y[i][j])]
                cost += self.penalty_lambda * (t - 1)**2
            # penalty 所有约束都惩罚施加
            if algorithm_optimization_method == 'penalty':
                return self.cost_dir * cost
        return objective
    
    

        