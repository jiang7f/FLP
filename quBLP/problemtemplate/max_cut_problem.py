import numpy as np
from typing import Iterable, List, Tuple
from ..models import ConstrainedBinaryOptimization
from quBLP.utils.quantum_lib import *

class MaxCutProblem(ConstrainedBinaryOptimization):
    def __init__(self, num_points: int, pairs_connected: List[Tuple[int, int]], fastsolve=False) -> None:
        super().__init__(fastsolve)
        self.set_optimization_direction('max')
        self.num_points = num_points
        self.pairs_connected = pairs_connected
        self.num_pairs = len(pairs_connected)
        self.num_variables = num_points

        self.X = self.add_binary_variables('x', [self.num_points])

        self.objective_penalty = self.objectivefunc()
        self.feasible_solution = self.get_feasible_solution()
        self.add_mct_objective()
        pass

    def add_mct_objective(self):
        for pair in self.pairs_connected:
            i = pair[0]
            j = pair[1]
            self.add_nonlinear_objective([i], 1)
            self.add_nonlinear_objective([j], 1)
            self.add_nonlinear_objective([i, j], -2)

    # @property
    # def generate_Hp(self):
    #     def add_in_target(num_qubits, target_qubit, gate):
    #         H = np.eye(2 ** (target_qubit))
    #         H = np.kron(H, gate)
    #         H = np.kron(H, np.eye(2 ** (num_qubits - 1 - target_qubit)))
    #         return H
    #     num_qubits = self.num_variables
    #     Hp = np.zeros((2**num_qubits, 2**num_qubits))
    #     for pair in self.pairs_connected:
    #         i = pair[0]
    #         j = pair[1]
    #         Hp += 1/2*(add_in_target(num_qubits, i, gate_z) @ add_in_target(num_qubits, j, gate_z) + np.eye(2**num_qubits))
    #     return Hp
    
    # def fast_solve_driver_bitstr(self):
    
    def get_feasible_solution(self):
        """ 根据约束寻找到一个可行解
        """
        # 全 0 就是可行解
        return [x.x for x in self.variables]

    def objectivefunc(self):
        def objective(variables:Iterable):
            cost = 0
            for pair in self.pairs_connected:
                i = pair[0]
                j = pair[1]
                xi = variables[self.var_to_idex(self.X[i])]
                xj = variables[self.var_to_idex(self.X[j])]
                cost += xi * (1 - xj) + xj * (1 - xi)
            # cost -= self.penalty_lambda*(variables[self.var_to_idex(self.X[0])] + variables[self.var_to_idex(self.X[2])]-2)**2
            return self.cost_dir * cost
        return objective

        