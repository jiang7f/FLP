# pending
import numpy as np
from typing import Iterable, List, Tuple
from ..models import ConstrainedBinaryOptimization
# 生成TSP问题约束矩阵
class GraphColoringProblem(ConstrainedBinaryOptimization):
    """ a `graph coloring problem`, is defined as
        
    """
    def __init__(self, num_graphs: int, pairs_adjacent: List[Tuple[int, int]], fastsolve=False) -> None:
        """ a facility location problem

        Args:
            num_graphs (int): number of graphs
            pairs_adjacent (List[Tuple[int, int]]): c[i] : the list     for demand i to facility j
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
            for i, (a, b) in enumerate(self.pairs_adjacent):
                matrix[m + j * p + i, a * n + j] = 1
                matrix[m + j * p + i, b * n + j] = 1
                matrix[m + j * p + i, m * n + i * n + j] = -1
        return matrix
    
    # def fast_solve_driver_bitstr(self):

    
    def get_feasible_solution(self):
        """ 根据约束寻找到一个可行解
        """
        # graph_i color color_i
        for i in range(self.num_graphs):
            self.X[i][i].set_value(1)
        for i, (a, b) in enumerate(self.pairs_adjacent):
            self.Y[i][a].set_value(1)
            self.Y[i][b].set_value(1)
        return np.nonzero([var.x for var in self.variables])[0]

    def objectivefunc(self):
        def objective(variables:Iterable):
            """ the objective function of the graph coloring problem

            Args:
                variables (Iterable):  the list of value of variables

            Return:
                cost
            """
            #  for j in range(self.num_colors):
            #     if 1 in [self.X[i][j].x for i in range(self.num_graphs)]:
            #         cost += 1
            # return cost
            from functools import reduce
            cost = self.num_colors
            maintain = -1 if self.num_graphs % 2 else 1
            for j in range(self.num_colors):
                cost -= maintain * reduce(lambda x, y: x * y, [self.X[i][j].x - 1 for i in range(self.num_graphs)])
            return cost
        return objective

        