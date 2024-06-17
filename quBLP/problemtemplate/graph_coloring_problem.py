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
        ## 相邻图对
        self.pairs_adjacent = pairs_adjacent
        self.num_adjacent = len(pairs_adjacent)
        # 最坏情况每个图一个颜色
        self.num_colors = num_graphs
        self.num_variables = self.num_graphs * self.num_colors + self.num_colors * self.num_adjacent
        # x_i_k = 1 如果顶点 i 被分配颜色 k, 否则为0
        self.X = self.add_binary_variables('x', [self.num_graphs, self.num_colors])
        self.Y = self.add_binary_variables('y', [self.num_colors, self.num_adjacent])
        self.objective_penalty = self.objective_func('penalty')
        self.objective_cyclic = self.objective_func('cyclic')
        self.objective_commute = self.objective_func('commute')
        self.feasible_solution = self.get_feasible_solution()
        # 加目标函数
        self.nolinear_objective_matrix = [self.generate_Hp]
        # 约束放到 self.constraints 里
        for cstrt in self.linear_constraints:
            self._add_linear_constraint(cstrt)
        pass

    @property
    def generate_Hp(self):
        def add_in_target(num_qubits, target_qubit, gate=np.array([[1, 0],[0, -1]])):
            H = np.eye(2 ** (target_qubit))
            H = np.kron(H, gate)
            H = np.kron(H, np.eye(2 ** (num_qubits - 1 - target_qubit)))
            return H
        odd_even = -1 if self.num_graphs % 2 else 1
        num_qubits = self.num_variables
        Hp = np.zeros((2**num_qubits, 2**num_qubits))
        for j in range(self.num_colors):
            Ht = np.eye(2**num_qubits)
            for i in range(self.num_graphs):
                Ht = Ht @ add_in_target(num_qubits, self.var_to_idex(self.X[i][j]), (gate_z + gate_I)/2)
            Hp += np.eye(2**num_qubits) - odd_even * Ht
        return np.multiply(Hp, self.cost_dir)
    
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
            for k, (a, b) in enumerate(self.pairs_adjacent):
                matrix[m + j * p + k, self.var_to_idex(self.X[a][j])] = 1
                matrix[m + j * p + k, self.var_to_idex(self.X[b][j])] = 1
                matrix[m + j * p + k, self.var_to_idex(self.Y[j][k])] = -1
        return matrix
    
    # def fast_solve_driver_bitstr(self):

    
    def get_feasible_solution(self):
        """ 根据约束寻找到一个可行解
        """
        # graph_i color color_i
        for i in range(self.num_graphs):
            self.X[i][i].set_value(1)
        for k, (a, b) in enumerate(self.pairs_adjacent):
            self.Y[a][k].set_value(1)
            self.Y[b][k].set_value(1)
        return [var.x for var in self.variables]

    def objective_func(self, algorithm_optimization_method):
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
                if 1 in [variables[self.var_to_idex(self.X[i][j])] for i in range(self.num_graphs)]:
                    cost += 1
            # commute 只需要目标函数一项
            if algorithm_optimization_method == 'commute':
                return self.cost_dir * cost
            for j in range(n):
                for k, (a, b) in enumerate(self.pairs_adjacent):
                    cost += self.penalty_lambda * (variables[self.var_to_idex(self.X[a][j])] + variables[self.var_to_idex(self.X[b][j])] - variables[self.var_to_idex(self.Y[j][k])])**2
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

        