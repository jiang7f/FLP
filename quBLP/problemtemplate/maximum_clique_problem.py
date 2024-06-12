import numpy as np
from typing import Iterable, List, Tuple
from ..models import ConstrainedBinaryOptimization
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
            pairs_unconnected (List[Tuple[int, int]]): the pair of unconnected points like (a, b)
        """
        super().__init__(fastsolve)
        # 设定优化方向
        self.set_optimization_direction('max')
        self.num_points = num_points
        self.pairs_unconnected = pairs_unconnected
        self.num_pairs = len(pairs_unconnected)
        self.num_variables = num_points + self.num_pairs

        self.X = self.add_binary_variables('x', [self.num_points])
        self.Y = self.add_binary_variables('y', [self.num_pairs])

        self.objective_penalty = self.objective_func('penalty')
        self.objective_commute = self.objective_func('commute')
        self.feasible_solution = self.get_feasible_solution()
        # 直接加目标函数表达式
        self._add_linear_objective(np.concatenate((self.cost_dir * np.array([1] * self.num_points), np.array([0] * self.num_pairs))))
        # 约束放到 self.constraints 里
        for cstrt in self.linear_constraints:
            self._add_linear_constraint(cstrt)
        pass
    @property
    def linear_constraints(self):
        total_rows = self.num_pairs
        total_columns = self.num_variables + 1
        matrix = np.zeros((total_rows, total_columns))
        for k, (a, b) in enumerate(self.pairs_unconnected):
            matrix[k, self.var_to_idex(self.X[a])] = 1
            matrix[k, self.var_to_idex(self.X[b])] = 1
            matrix[k, self.var_to_idex(self.Y[k])] = -1
        return matrix
    
    # def fast_solve_driver_bitstr(self):
    
    def get_feasible_solution(self):
        """ 根据约束寻找到一个可行解
        """
        # 全 0 就是可行解
        return [x.x for x in self.variables]

    def objective_func(self, optimization_method):
        def objective(variables:Iterable):
            """ the objective function of the maximum clique problem

            Args:
                variables (Iterable):  the list of value of variables

            Returns:
                cost
            """
            m = self.num_points
            cost = 0
            for i in range(m):
                cost += variables[self.var_to_idex(self.X[i])]
            if optimization_method == 'commute':
                return self.cost_dir * cost
            for k, (a, b) in enumerate(self.pairs_unconnected):
                cost += -self.penalty_lambda * (variables[self.var_to_idex(self.X[a])] + variables[self.var_to_idex(self.X[b])] - variables[self.var_to_idex(self.Y[k])])**2
            if optimization_method == 'penalty':
                return self.cost_dir * cost
        return objective

        