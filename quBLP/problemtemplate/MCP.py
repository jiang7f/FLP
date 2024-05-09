import numpy as np
from typing import Iterable, List, Tuple
from ..models import ConstrainedBinaryOptimization
# 生成TSP问题约束矩阵
class MaximumCliqueProblem(ConstrainedBinaryOptimization):
    """ a maximum clique problem is defined as
        **变量**:
        - 假设有 m 个顶点, p个不相连点对
        - $x_i \text{ for } i = 1, \cdots, m$, 当顶点 $i$ 在团中时 $x_i = 1$, 否则为 $0$  

        **目标函数**:
        - $\max \sum_{i=1}^m x_i$  

        **约束**:
        - $x_{a_k} + x_{b_k} \leq 1$ for all pair of unconnected vertices $(a_k, b_k), k = 1, \cdots, p$
        - $x_{i} \in\{0,1\} $

        **等式约束**:
        - $x_{a_k} + x_{b_k} - y_{k} = 0$ for all pair of unconnected vertices $(a_k, b_k), k = 1, \cdots, p$
        - $x_{i}, y_{k} \in\{0,1\} $  
    """
    def __init__(self, num_points: int, pairs_unconnected: List[Tuple[int, int]], fastsolve=False) -> None:
        """ 
        Args:
            num_points (int): number of points
            pairs_unconnected (List[Tuple[int, int]]): c[i] : the pair of unconnected points like (a, b)
        """
        super().__init__(fastsolve)
        self.num_points = num_points
        self.pairs_unconnected = pairs_unconnected
        self.num_pairs = len(pairs_unconnected)
        self.num_variables = num_points + self.num_pairs

        self.X = self.add_binary_variables('x', [self.num_points])
        self.Y = self.add_binary_variables('y', [self.num_pairs])

        self.objective = self.objectivefunc()
        self.feasible_solution = self.get_feasible_solution()
        pass
    @property
    def linear_constraints(self):
        total_rows = self.num_pairs
        total_columns = self.num_variables + 1
        matrix = np.zeros((total_rows, total_columns))
        for k, (a, b) in enumerate(self.pairs_unconnected):
            matrix[k, a] = 1
            matrix[k, b] = 1
            matrix[k, self.num_points + k] = 1
        return matrix
    
    # def fast_solve_driver_bitstr(self):
    
    def get_feasible_solution(self):
        """ 根据约束寻找到一个可行解
        """
        # 全 0 就是可行解
        return [x.x for x in self.variables]

    def objectivefunc(self):
        def objective(variables:Iterable):
            """ the objective function of the maximum clique problem

            Args:
                variables (Iterable):  the list of value of variables

            Returns:
                cost
            """
            cost = 0
            for i in range(self.num_points):
                cost += variables[self.var_to_idex(self.X[i])]
            return -cost
        return objective

        