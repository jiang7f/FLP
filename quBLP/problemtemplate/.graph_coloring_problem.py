import numpy as np
from typing import Iterable, List, Tuple
from ..models import ConstrainedBinaryOptimization
from quBLP.utils.quantum_lib import *
class GraphColoringProblem(ConstrainedBinaryOptimization):
    def __init__(self, num_graphs: int, weight_list:List[int], pairs_adjacent: List[Tuple[int, int]], num_colors=None, fastsolve=False) -> None:
        """ 
        Args:
            num_graphs (int): number of graphs
            pairs_adjacent (List[Tuple[int, int]]): the pair of adjacent graphs like (a, b)
        """
        super().__init__(fastsolve)
        # 设定优化方向
        self.set_optimization_direction('min')
        ## 图个数
        self.num_graphs = num_graphs
        ## 相邻图对
        self.pairs_adjacent = pairs_adjacent
        self.num_adjacent = len(pairs_adjacent)
        # 如果不给定最大颜色数量, 最坏情况每个图一个颜色, 但是可行解需要改（待补）
        if num_colors is None:
            self.num_colors = num_graphs
        else:
            self.num_colors = num_colors
        self.num_variables = 2 * self.num_graphs * self.num_colors + self.num_colors + self.num_adjacent * self.num_colors
        # x_i_k = 1 如果顶点 i 被分配颜色 k, 否则为0
        self.X = self.add_binary_variables('x', [self.num_graphs, self.num_colors])
        self.Y = self.add_binary_variables('y', [self.num_colors])
        self.Z = self.add_binary_variables('z', [self.num_adjacent, self.num_colors])
        self.H = self.add_binary_variables('h', [self.num_graphs, self.num_colors])
        self.objective_penalty = self.get_objective_func('penalty')
        self.objective_cyclic = self.get_objective_func('cyclic')
        self.objective_commute = self.get_objective_func('commute')
        self.feasible_solution = self.get_feasible_solution()
        self.add_linear_objective([0] * self.num_graphs * self.num_colors + weight_list + [0] * self.num_adjacent * self.num_colors + [0] * self.num_graphs * self.num_colors)
        pass
    
    @property
    def linear_constraints(self):
        if self._linear_constraints is None:
            V = self.num_graphs
            C = self.num_colors
            E = self.num_adjacent
            total_rows = V + E * C + V * C
            total_columns = 2 * V * C + E * C + C + 1
            matrix = np.zeros((total_rows, total_columns))
            for v in range(V):
                for i in range(C):
                    matrix[v, self.var_to_idex(self.X[v][i])] = 1
                matrix[v, total_columns - 1] = 1
            for i in range(C):
                for k, (u, w) in enumerate(self.pairs_adjacent):
                    matrix[V + i * E + k, self.var_to_idex(self.X[u][i])] = 1
                    matrix[V + i * E + k, self.var_to_idex(self.X[w][i])] = 1
                    matrix[V + i * E + k, self.var_to_idex(self.Z[k][i])] = 1
                    matrix[V + i * E + k, self.var_to_idex(self.Y[i])] = -1
            for i in range(C):
                for v in range(V):
                    matrix[V + C * E + i * V + v, self.var_to_idex(self.X[v][i])] = 1
                    matrix[V + C * E + i * V + v, self.var_to_idex(self.H[v][i])] = 1
                    matrix[V + C * E + i * V + v, self.var_to_idex(self.Y[i])] = -1
            self._linear_constraints = matrix
        return self._linear_constraints
    
    # def fast_solve_driver_bitstr(self):

    
    def get_feasible_solution(self):
        """ 根据约束寻找到一个可行解
        """
        # graph_i color color_i
        V = self.num_graphs
        C = self.num_colors
        for i in range(C):
            self.X[i][i].set_value(1)
            self.Y[i].set_value(1)
        for i in range(C):
            for k, (u, w) in enumerate(self.pairs_adjacent):
                self.Z[k][i].set_value(self.Y[i].x - self.X[u][i].x - self.X[w][i].x)
        for i in range(C):
            for v in range(V):
                self.H[v][i].set_value(self.Y[i].x - self.X[v][i].x)
        return [x.x for x in self.variables]

    def get_objective_func(self, algorithm_optimization_method):
        def objective(variables:Iterable):
            """ the objective function of the graph coloring problem

            Args:
                variables (Iterable):  the list of value of variables

            Return:
                cost
            """
            V = self.num_graphs
            C = self.num_colors
            # costfun_1
            cost = 0
            for c in range(C):
                cost += variables[self.var_to_idex(self.Y[c])]
            # commute 只需要目标函数一项
            if algorithm_optimization_method == 'commute':
                return self.cost_dir * cost
            for i in range(C):
                for v in range(V):
                    cost += self.penalty_lambda * (variables[self.var_to_idex(self.X[v][i])] + variables[self.var_to_idex(self.H[v][i])] - variables[self.var_to_idex(self.Y[i])])**2
            for i in range(C):
                for k, (u, w) in enumerate(self.pairs_adjacent):
                    cost += self.penalty_lambda * (variables[self.var_to_idex(self.X[u][i])] + variables[self.var_to_idex(self.X[w][i])] + variables[self.var_to_idex(self.Z[k][i])] - variables[self.var_to_idex(self.Y[i])])**2
            if algorithm_optimization_method == 'cyclic':
                return self.cost_dir * cost
            for v in range(V):
                t = 0
                for i in range(C):
                    t += variables[self.var_to_idex(self.X[v][i])]
                cost += self.penalty_lambda * (t - 1)**2
            # penalty 所有约束都惩罚施加
            if algorithm_optimization_method == 'penalty':
                return self.cost_dir * cost
        return objective

        