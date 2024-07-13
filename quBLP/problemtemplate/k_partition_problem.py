from quBLP.utils import iprint
import numpy as np
from typing import Iterable, List, Tuple
from ..models import ConstrainedBinaryOptimization
from quBLP.utils.quantum_lib import *
import gurobipy as gp

class KPartitionProblem(ConstrainedBinaryOptimization):
    def __init__(self, num_points: int, block_allot: List[int], pairs_connected: List[Tuple[Tuple[int, int], int]], fastsolve=False) -> None:
        super().__init__(fastsolve)
        self.set_optimization_direction('max')
        self.num_points = num_points
        self.num_block = len(block_allot)
        self.block_allot = block_allot
        self.pairs_connected = pairs_connected
        self.num_pairs = len(pairs_connected)
        self.num_variables = self.num_points * self.num_block
        self.X = self.add_binary_variables('x', [self.num_points, self.num_block])
        self.objective_penalty = self.get_objective_func_penalty
        self.objective_cyclic = self.get_objective_func_cyclic
        self.objective_commute = self.get_objective_func_commute
        self.feasible_solution = self.get_feasible_solution()
        self.add_kpp_objective()
        pass
    
    def add_kpp_objective(self):
        k = self.num_block
        for pair, w in self.pairs_connected:
            u = pair[0]
            v = pair[1]
            theta = -w
            for j in range(k):
                self.add_nonlinear_objective([self.var_to_idex(self.X[u][j]), self.var_to_idex(self.X[v][j])], self.cost_dir * theta)
        pass

    @property
    def linear_constraints(self):
        if self._linear_constraints is None:
            n = self.num_points
            k = self.num_block
            total_rows = n + k
            total_columns = self.num_variables + 1
            matrix = np.zeros((total_rows, total_columns))
            for i in range(n):
                for j in range(k):
                    matrix[i, self.var_to_idex(self.X[i][j])] = 1
                matrix[i, total_columns - 1] = 1
            for j in range(k):
                for i in range(n):
                    matrix[n + j, self.var_to_idex(self.X[i][j])] = 1
                matrix[n + j, total_columns - 1] = self.block_allot[j]
            self._linear_constraints = matrix
            iprint(self._linear_constraints)
        return self._linear_constraints

    def get_feasible_solution(self):
        """ 根据约束寻找到一个可行解
        """
        t = 0
        for b, p in enumerate(self.block_allot):
            for _ in range(p):
                self.X[t][b].set_value(1)
                t += 1
        # n = self.num_points
        # k = self.num_block
        # p = n // k
        # t = 0
        # for i in range(n):
        #     self.X[i][t // p].set_value(1)
        #     t += 1
        return [x.x for x in self.variables]
    
    def get_objective_func_commute(self, variables:Iterable):
        cost = 0
        for pair, w in self.pairs_connected:
            u = pair[0]
            v = pair[1]
            t = 0
            for j in range(self.num_block):
                x_uj = variables[self.var_to_idex(self.X[u][j])]
                x_vj = variables[self.var_to_idex(self.X[v][j])]
                t += x_uj * x_vj
            cost += w * (1 - t)
        return self.cost_dir * cost
    
    def get_objective_func_cyclic(self, variables:Iterable):
        cost = 0
        for j in range(self.num_block):
            t = 0
            for i in range(self.num_points):
                t += variables[self.var_to_idex(self.X[i][j])]
            cost += self.cost_dir * self.penalty_lambda * (t - self.block_allot[j])**2
        return self.get_objective_func_commute(variables) + self.cost_dir * cost

    def get_objective_func_penalty(self, variables:Iterable):
        cost = 0
        for i in range(self.num_points):
            t = 0
            for j in range(self.num_block):
                t += variables[self.var_to_idex(self.X[i][j])]
            cost += self.cost_dir * self.penalty_lambda * (t - 1)**2
        return self.get_objective_func_cyclic(variables) + self.cost_dir * cost

    def get_best_cost(self):
        """ Solve the KPartitionProblem using Gurobi """
        try:
            model = gp.Model()
            model.setParam('OutputFlag', 0)
            
            # Add variables
            X = model.addVars(self.num_points, self.num_block, vtype=gp.GRB.BINARY, name="X")

            # Objective function
            model.setObjective(gp.quicksum(w - w * X[pair[0], j] for j in range(self.num_block) for pair, w in self.pairs_connected), gp.GRB.MAXIMIZE)

            # Constraints
            for i in range(self.num_points):
                model.addConstr(gp.quicksum(X[i, j] for j in range(self.num_block)) == 1, f"PointAssignment_{i}")
            
            for j in range(self.num_block):
                model.addConstr(gp.quicksum(X[i, j] for i in range(self.num_points)) == self.block_allot[j], f"BlockAllotment_{j}")

            # Optimize the model
            model.optimize()

            if model.status == gp.GRB.OPTIMAL:
                # solution = {var.varName: var.x for var in model.getVars()}.values()
                optimal_value = model.objVal
                return optimal_value
            else:
                return None
        except gp.GurobiError as e:
            print(f'Error code {e.errno}: {e}')
            return None
        