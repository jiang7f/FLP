# pending
import numpy as np
from typing import Iterable, List, Tuple
from ..models import ConstrainedBinaryOptimization
# 生成TSP问题约束矩阵
class GraphColoringProblem(ConstrainedBinaryOptimization):
    """ a `graph coloring problem`, is defined as
        
    """
    def __init__(self, num_graphs: int, adjacent_pairs: List[Tuple[int, int]], fastsolve=False) -> None:
        """ a facility location problem

        Args:
            num_graphs (int): number of graphs
            adjacent_pairs (List[Tuple[int, int]]): c[i] : the list     for demand i to facility j
        """
        super().__init__(fastsolve)
        ## 图个数
        self.num_graphs = num_graphs
        ## 相邻图对
        self.adjacent_pairs = adjacent_pairs
        # 最坏情况每个图一个颜色
        num_colors = num_graphs
        self.num_colors = num_graphs
        # x_i_k = 1 如果顶点 i 被分配颜色 k, 否则为0
        self.X = self.add_binary_variables('x', [num_graphs, num_colors])
        self.Y = self.add_binary_variables('y', [num_graphs, num_graphs, num_colors])
        self.objective = self.objectivefunc()
        self.feasible_solution = self.get_feasible_solution()
        pass
    @property
    def linear_constraints(self):
        from quBLP.utils import linear_system as ls
        ls.set_print_form()
        m = self.num_graphs
        n = self.num_graphs # 颜色的最大数量
        p = len(self.adjacent_pairs)
        total_rows = m + p * n
        total_columns = m * n + m * m * n + 1
        matrix = np.zeros((total_rows, total_columns))
        for i in range(m):
            for j in range(n):
                matrix[i, i * n + j] = 1
            matrix[i, total_columns - 1] = 1
        for j in range(n):
            for i, (a, b) in enumerate(self.adjacent_pairs):
                matrix[m + j * p + i, a * n + j] = 1
                matrix[m + j * p + i, b * n + j] = 1
                matrix[m + j * p + i, m * n + a * m * n  + b * n + j] = -1
        return matrix
    
    # def fast_solve_driver_bitstr(self):

    
    def get_feasible_solution(self):
        """ 根据约束寻找到一个可行解
        """
        # graph_i color color_i
        for i in range(self.num_graphs):
            self.X[i][i].set_value(1)
        for a, b in self.adjacent_pairs:
            self.Y[a][b][a].set_value(1)
            self.Y[a][b][b].set_value(1)
        return np.nonzero([var.x for var in self.variables])[0]

    def objectivefunc(self):
        def objective(variables:Iterable):
            """ the objective function of the facility location problem

            Args:
                variables (Iterable):  the list of value of variables

            Return:
                cost
            """
            cost = 0
            for j in range(self.num_colors):
                if 1 in [self.X[i][j].x for i in range(self.num_graphs)]:
                    cost += 1
            return cost
        return objective

        