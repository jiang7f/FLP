from quBLP.utils import iprint
import numpy as np
from typing import Iterable
from ..models import ConstrainedBinaryOptimization
class One_Hdi(ConstrainedBinaryOptimization):
    def __init__(self, num_qubits: int, fastsolve=False) -> None:
        super().__init__(fastsolve)
        self.set_optimization_direction('min')
        self.num_qubits = num_qubits
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
            total_rows = 1
            total_columns = n + 1
            matrix = np.zeros((total_rows, total_columns))
            for i in range(n):
                matrix[0, i] = 1 - (i % 2) * 2
            self._linear_constraints = matrix
            iprint(self._linear_constraints)
        return self._linear_constraints

    @linear_constraints.setter
    def linear_constraints(self, constraints):
        self._linear_constraints = constraints

    def get_feasible_solution(self):
        return [x.x for x in self.variables]

    def objective(variables:Iterable):
        cost = 0
        return cost