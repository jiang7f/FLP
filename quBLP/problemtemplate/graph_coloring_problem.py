import numpy as np
from typing import Iterable, List, Tuple
from ..models import ConstrainedBinaryOptimization
from quBLP.utils.quantum_lib import *
class GraphColoringProblem(ConstrainedBinaryOptimization):
    """ a graph coloring problem is defined as
        **变量**:
        - 假设有 m 块图, n 种颜色, p个相邻图对
        - $x_{ij} \text{ for } i = 1, \cdots, m \text{ and } j = 1, \cdots, n$, 当图 $i$ 被分配颜色 $j$ 时 $x_{ij} = 1$, 否则为 $0$

        **目标函数**:
        - $\min n $

        **约束**:
        - $\sum_{j = 1}^n x_{ij} = 1 \text{ for all } i = 1, \cdots, m$
        - $x_{a_kj} + x_{b_kj} \leq 1$ for all pair of adjacent graphs $(a_k, b_k), k = 1, \cdots, p$ and $j = 1, \cdots, n$
        - $x_{ij} \in\{0,1\} $

        **等式约束**:
        - $\sum_{j = 1}^n x_{ij} = 1 \text{ for all } i = 1, \cdots, m$
        - $x_{a_kj} + x_{b_kj} - y_{jk} = 0$ for all pair of adjacent graphs $(a_k, b_k), k = 1, \cdots, p$ and $j = 1, \cdots, n$
        - $x_{ij}, y_{jk} \in\{0,1\} $
    """
    def __init__(self, num_graphs: int, pairs_adjacent: List[Tuple[int, int]], fastsolve=False) -> None:
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
        self.objective_func_term_list = [[] for _ in range(self.num_graphs)]
        ## 相邻图对
        self.pairs_adjacent = pairs_adjacent
        self.num_adjacent = len(pairs_adjacent)
        # 最坏情况每个图一个颜色
        self.num_colors = num_graphs
        self.num_variables = self.num_graphs * self.num_colors + self.num_colors * self.num_adjacent
        # x_i_k = 1 如果顶点 i 被分配颜色 k, 否则为0
        self.X = self.add_binary_variables('x', [self.num_graphs, self.num_colors])
        self.Y = self.add_binary_variables('y', [self.num_adjacent, self.num_colors])
        self.objective_penalty = self.get_objective_func('penalty')
        self.objective_cyclic = self.get_objective_func('cyclic')
        self.objective_commute = self.get_objective_func('commute')
        self.feasible_solution = self.get_feasible_solution()
        # 加目标函数
        self.add_gcp_objective()
        pass


    def add_gcp_objective(self):
        m = self.num_graphs
        n = self.num_colors
        from itertools import combinations
        for j in range(n):
            for k in range(1, m + 1):
                for combo in combinations(range(m), k):
                    theta = (-1) ** k
                    term_list = []
                    for index in combo:
                        term_list.append(self.var_to_idex(self.X[index][j]))
                    self.add_nonlinear_objective(term_list, self.cost_dir * -theta)
    
    @property
    def linear_constraints(self):
        from quBLP.utils import linear_system as ls
        ls.set_print_form()
        m = self.num_graphs
        n = self.num_colors # 颜色的最大数量
        p = self.num_adjacent
        total_rows = m + p * n
        total_columns = m * n + p * n + 1
        matrix = np.zeros((total_rows, total_columns))
        for i in range(m):
            for j in range(n):
                matrix[i, self.var_to_idex(self.X[i][j])] = 1
            matrix[i, total_columns - 1] = 1
        for j in range(n):
            for k, (u, v) in enumerate(self.pairs_adjacent):
                matrix[m + j * p + k, self.var_to_idex(self.X[u][j])] = 1
                matrix[m + j * p + k, self.var_to_idex(self.X[v][j])] = 1
                matrix[m + j * p + k, self.var_to_idex(self.Y[k][j])] = -1
        return matrix
    
    # def fast_solve_driver_bitstr(self):

    
    def get_feasible_solution(self):
        """ 根据约束寻找到一个可行解
        """
        # graph_i color color_i
        for i in range(self.num_graphs):
            self.X[i][i].set_value(1)
        for k, (u, v) in enumerate(self.pairs_adjacent):
            for j in range(self.num_colors):
                self.Y[k][j].set_value(self.X[u][j].x + self.X[v][j].x)
            # self.Y[k][a].set_value(1)
            # self.Y[k][b].set_value(1)
        return [var.x for var in self.variables]

    def get_objective_func(self, algorithm_optimization_method):
        def objective(variables:Iterable):
            """ the objective function of the graph coloring problem

            Args:
                variables (Iterable):  the list of value of variables

            Return:
                cost
            """
            m = self.num_graphs
            n = self.num_colors # 颜色的最大数量
            # costfun_1
            cost = 0
            for j in range(n):
                t = 1
                for i in range(m):
                    t *= 1 - variables[self.var_to_idex(self.X[i][j])]
                cost += 1 - t
            # commute 只需要目标函数一项
            if algorithm_optimization_method == 'commute':
                return self.cost_dir * cost
            for j in range(n):
                for k, (u, v) in enumerate(self.pairs_adjacent):
                    cost += self.penalty_lambda * (variables[self.var_to_idex(self.X[u][j])] + variables[self.var_to_idex(self.X[v][j])] - variables[self.var_to_idex(self.Y[k][j])])**2
            # cyclic 多包含一项∑=x
            if algorithm_optimization_method == 'cyclic':
                return self.cost_dir * cost
            for i in range(m):
                t = 0
                for j in range(n):
                    t += variables[self.var_to_idex(self.X[i][j])]
                cost += self.penalty_lambda * (t - 1)**2
            # penalty 所有约束都惩罚施加
            if algorithm_optimization_method == 'penalty':
                return self.cost_dir * cost
            
            # costfun_2 构造Hp可以考虑这个
            # from functools import reduce
            # cost = 0
            # odd_even = -1 if self.num_graphs % 2 else 1
            # for j in range(self.num_colors): 
            #     cost += 1 - odd_even * reduce(lambda x, y: x * y, [variables[self.var_to_idex(self.X[i][j])] - 1 for i in range(self.num_graphs)])
            # if algorithm_optimization_method == 'commute':
            #     return self.cost_dir * cost
        return objective

        