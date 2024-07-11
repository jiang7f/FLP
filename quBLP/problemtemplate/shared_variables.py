from quBLP.utils import iprint
import numpy as np
from typing import Iterable
from ..models import ConstrainedBinaryOptimization
class SharedVariables(ConstrainedBinaryOptimization):
    def __init__(self, num_qubits: int, num_shared_variales: int, fastsolve=False) -> None:
        super().__init__(fastsolve)
        #* 设定优化方向
        self.set_optimization_direction('min')
        # 需求点个数
        self.num_qubits = num_qubits
        # 设施点个数
        self.num_shared_variales = num_shared_variales
        self.num_variables = num_qubits
        self.X = self.add_binary_variables('x', [num_qubits])
        self.objective_penalty = self.objective
        self.objective_cyclic = self.objective
        self.objective_commute = self.objective
        self.feasible_solution = self.get_feasible_solution()
        #* 直接加目标函数表达式
        self.add_linear_objective([0] * num_qubits)
        pass
    
    @property
    def linear_constraints(self):
        if self._linear_constraints is None:
            n = self.num_qubits
            m = self.num_shared_variales
            total_rows = n // 2
            total_columns = self.num_variables + 1
            matrix = np.zeros((total_rows, total_columns))
            for i in range(m):
                matrix[i, i] = 1
                matrix[i, self.num_qubits - 1] = 1
            for i in range(m, total_rows):
                matrix[i, i] = 1
                matrix[i, self.num_qubits - 1 - i] = 1
            self._linear_constraints = matrix
            iprint(self._linear_constraints)
        return self._linear_constraints
    
    def get_feasible_solution(self):
        return [x.x for x in self.variables]

    def objective(variables:Iterable):
        cost = 0
        return cost