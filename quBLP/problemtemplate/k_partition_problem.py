import numpy as np
from typing import Iterable, List, Tuple
from ..models import ConstrainedBinaryOptimization
from quBLP.utils.quantum_lib import *

class KPartitionProblem(ConstrainedBinaryOptimization):
    def __init__(self, num_points: int, block_allot: List[int], pairs_connected: List[Tuple[Tuple[int, int], int]], fastsolve=False) -> None:
        super().__init__(fastsolve)
        self.set_optimization_direction('max')
        self.num_points = num_points
        self.num_block = len(block_allot)
        self.block_allot = block_allot
        self.pairs_connected = pairs_connected
        self.num_pairs = len(pairs_connected)
        self.num_variables = num_points

        self.num_variables = self.num_points * self.num_block
        self.X = self.add_binary_variables('x', [self.num_points, self.num_block])
        self.objective_penalty = self.get_objective_func('penalty')
        self.objective_cyclic = self.get_objective_func('cyclic')
        self.objective_commute = self.get_objective_func('commute')
        self.feasible_solution = self.get_feasible_solution()
        self.add_kpp_objective()
        pass
    
    def add_kpp_objective(self):
        k = self.num_block
        for pair, w in self.pairs_connected:
            u = pair[0]
            v = pair[1]
            theta = -w
            for j in range(k):
                self.add_nonlinear_objective([self.var_to_idex(self.X[u][j]), self.var_to_idex(self.X[v][j])], self.cost_dir * theta)
        pass

    @property
    def linear_constraints(self):
        if self._linear_constraints is None:
            n = self.num_points
            k = self.num_block
            total_rows = n + k
            total_columns = self.num_variables + 1
            matrix = np.zeros((total_rows, total_columns))
            for i in range(n):
                for j in range(k):
                    matrix[i, self.var_to_idex(self.X[i][j])] = 1
                matrix[i, total_columns - 1] = 1
            for j in range(k):
                for i in range(n):
                    matrix[n + j, self.var_to_idex(self.X[i][j])] = 1
                matrix[n + j, total_columns - 1] = self.block_allot[j]
            self._linear_constraints = matrix
            print(self._linear_constraints)
        return self._linear_constraints

    def get_feasible_solution(self):
        """ 根据约束寻找到一个可行解
        """
        t = 0
        for b, p in enumerate(self.block_allot):
            for _ in range(p):
                self.X[t][b].set_value(1)
                t += 1
        # n = self.num_points
        # k = self.num_block
        # p = n // k
        # t = 0
        # for i in range(n):
        #     self.X[i][t // p].set_value(1)
        #     t += 1
        return [x.x for x in self.variables]
    
    def get_objective_func(self, algorithm_optimization_method):
        n = self.num_points
        k = self.num_block
        def objective(variables:Iterable):
            cost = 0
            for pair, w in self.pairs_connected:
                u = pair[0]
                v = pair[1]
                t = 0
                for j in range(k):
                    x_uj = variables[self.var_to_idex(self.X[u][j])]
                    x_vj = variables[self.var_to_idex(self.X[v][j])]
                    t += x_uj * x_vj
                cost += w * (1 - t)
            if algorithm_optimization_method == 'commute':
                return self.cost_dir * cost
            for j in range(k):
                t = 0
                for i in range(n):
                    t += variables[self.var_to_idex(self.X[i][j])]
                cost += self.cost_dir * self.penalty_lambda * (t - self.block_allot[j])**2
            if algorithm_optimization_method == 'cyclic':
                return self.cost_dir * cost
            for i in range(n):
                t = 0
                for j in range(k):
                    t += variables[self.var_to_idex(self.X[i][j])]
                cost += self.cost_dir * self.penalty_lambda * (t - 1)**2
            if algorithm_optimization_method == 'penalty':
                return self.cost_dir * cost
        return objective
