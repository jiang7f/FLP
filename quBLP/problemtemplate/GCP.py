import numpy as np
from typing import Iterable, List, Tuple
from ..models import ConstrainedBinaryOptimization
# 生成TSP问题约束矩阵
class GraphColoringProblem(ConstrainedBinaryOptimization):
    """ a `graph coloring problem`, is defined as
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
        - $x_{a_kj} + x_{b_kj} - y_{kj} = 0$ for all pair of adjacent graphs $(a_k, b_k), k = 1, \cdots, p$ and $j = 1, \cdots, n$
        - $x_{ij}, y_{kj} \in\{0,1\} $
    """
    def __init__(self, num_graphs: int, pairs_adjacent: List[Tuple[int, int]], fastsolve=False) -> None:
        """ 
        Args:
            num_graphs (int): number of graphs
            pairs_adjacent (List[Tuple[int, int]]): c[i] : the pair of adjacent graphs like (a, b)
        """
        super().__init__(fastsolve)
        ## 图个数
        self.num_graphs = num_graphs
        ## 相邻图对
        self.pairs_adjacent = pairs_adjacent
        self.num_adjacent = len(pairs_adjacent)
        # 最坏情况每个图一个颜色
        self.num_colors = num_graphs
        # x_i_k = 1 如果顶点 i 被分配颜色 k, 否则为0
        self.X = self.add_binary_variables('x', [self.num_graphs, self.num_colors])
        self.Y = self.add_binary_variables('y', [self.num_adjacent, self.num_colors])
        self.objective = self.objectivefunc()
        self.feasible_solution = self.get_feasible_solution()
        pass
    @property
    def linear_constraints(self):
        from quBLP.utils import linear_system as ls
        ls.set_print_form()
        m = self.num_graphs
        n = self.num_graphs # 颜色的最大数量
        p = self.num_adjacent
        total_rows = m + p * n
        total_columns = m * n + p * n + 1
        matrix = np.zeros((total_rows, total_columns))
        for i in range(m):
            for j in range(n):
                matrix[i, i * n + j] = 1
            matrix[i, total_columns - 1] = 1
        for j in range(n):
            for k, (a, b) in enumerate(self.pairs_adjacent):
                matrix[m + j * p + k, a * n + j] = 1
                matrix[m + j * p + k, b * n + j] = 1
                matrix[m + j * p + k, m * n + k * n + j] = -1
        return matrix
    
    # def fast_solve_driver_bitstr(self):

    
    def get_feasible_solution(self):
        """ 根据约束寻找到一个可行解
        """
        # graph_i color color_i
        for i in range(self.num_graphs):
            self.X[i][i].set_value(1)
        for k, (a, b) in enumerate(self.pairs_adjacent):
            self.Y[k][a].set_value(1)
            self.Y[k][b].set_value(1)
        return [var.x for var in self.variables]

    def objectivefunc(self):
        def objective(variables:Iterable):
            """ the objective function of the graph coloring problem

            Args:
                variables (Iterable):  the list of value of variables

            Return:
                cost
            """
            # costfun_1
            cost = 0
            for j in range(self.num_colors):
                if 1 in [variables[self.var_to_idex(self.X[i][j])] for i in range(self.num_graphs)]:
                    cost += 1
            return cost

            # costfun_2 构造Hp可以考虑这个
            # from functools import reduce
            # cost = self.num_colors
            # maintain = -1 if self.num_graphs % 2 else 1
            # for j in range(self.num_colors): 
            #     cost -= maintain * reduce(lambda x, y: x * y, [variables[self.var_to_idex(self.X[i][j])] - 1 for i in range(self.num_graphs)])
            # return cost
        return objective

        