# pending
import numpy as np
from typing import Iterable
from ..models import ConstrainedBinaryOptimization
class SetCoverProblem(ConstrainedBinaryOptimization):
    """ a set cover problem is defined as
        **变量**:
        - 假设有 m 个集合, n 个元素
        - $x_i \text{ for } i = 1, \cdots, m$, 当选择集合 $i$ 时 $x_i = 1$, 否则为 $0$

        **目标函数**:
        - $\min \sum_{i=1}^m x_i$

        **约束**:
        - $\sum_{i: e_j \in S_i} x_i \geq 1$ for all $j = 1, \cdots, n$.  
        即每个元素 $e_j$ 至少被一个集合覆盖
        - $x_{i} \in\{0,1\} $  

        **等式约束**:
        - 没想好怎么表述元素属于多少个集合, 写了len
        - $\sum_{i: e_j \in S_i} x_i - \sum_{k=1}^{len(e_j \in) - 1}y_{jk}= 1$ for all $j = 1, \cdots, n$. 
        - $x_{i}, y_{jk} \in\{0,1\} $   
    """
    def __init__(self, num_sets: int, list_element_belongs: Iterable[Iterable], fastsolve=False) -> None:
        """ 
        Args:
            num_sets (int): number of sets
            list_element_belongs (Iterable[Iterable]):  list_element_belong[i]: element i belongs to list of sets
        """
        super().__init__(fastsolve)
        # 如果只属于 1 个, 那个集合直接选中, 只考虑 >= 2 的情况
        for element_belongs in list_element_belongs:
            assert len(element_belongs) >= 2
        self.num_sets = num_sets
        self.list_element_belongs = list_element_belongs
        self.num_elements = len(self.list_element_belongs)
        self.num_variables = num_sets + sum([len(element_belongs) - 1 for element_belongs in self.list_element_belongs])

        self.X = self.add_binary_variables('x', [self.num_sets])
        # 注意每一维度变量个数不同
        self.Y = [[self.add_binary_variable('y_'+str(j) + '_' + str(k)) for k in range(len(self.list_element_belongs[j]) - 1)] for j in range(len(self.list_element_belongs))]

        self.objective = self.objectivefunc()
        self.feasible_solution = self.get_feasible_solution()
        pass
    @property
    def linear_constraints(self):
        m = self.num_sets
        n = self.num_elements
        total_rows = n
        total_columns = self.num_variables + 1
        matrix = np.zeros((total_rows, total_columns))
        for j in range(n):
            for i in self.list_element_belongs[j]:
                matrix[j, self.var_to_idex(self.X[i])] = 1
            for k in range(len(self.list_element_belongs[j]) - 1):
                matrix[j, self.var_to_idex(self.Y[j][k])] = -1
            matrix[j, total_columns - 1] = 1
        return matrix
    
    # def fast_solve_driver_bitstr(self):
    
    def get_feasible_solution(self):
        """ 根据约束寻找到一个可行解
        """
        for var in self.variables:
            var.set_value(1)
        return [x.x for x in self.variables]

    def objectivefunc(self):
        def objective(variables:Iterable):
            """ 
            Args:
                variables (Iterable):  the list of value of variables
            Returns:
                scaler
            """
            m = self.num_sets
            cost = 0
            for i in range(m):
                cost += variables[self.var_to_idex(self.X[i])]
            return cost
        return objective

        