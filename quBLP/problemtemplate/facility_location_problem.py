import numpy as np
import itertools
from typing import Iterable
from ..models import ConstrainedBinaryOptimization
import gurobipy as gp

class FacilityLocationProblem(ConstrainedBinaryOptimization):
    """ a facility location problem is defined as
    .. math::
        min \sum_{i=1}^m \sum_{j=1}^n c_{i j} y_{i j}+\sum_{j=1}^n f_j x_j 
    
    .. math::
        s.t. \quad \sum_{j=1}^n y_{i j}=1, \quad i=1,2, \cdots, m 
    .. math::
        y_{i j}+z_{i j}-x_j=0, \quad i=1, \cdots, m, j=1, \cdots, n
    .. math::
        z_{i j}, y_{i j}, x_j \in\{0,1\} 

    """
    def __init__(self, num_demands: int, num_facilities: int, c: Iterable[Iterable],f: Iterable, fastsolve=True) -> None:
        """ a facility location problem

        Args:
            n (int): number of facilities
            m (int): number of demand point
            c (Matrix[m,n]): c_{i,j} : the cost for demand i to facility j
            f (Vector[n]): f_j: the building cost for facility j
        """
        super().__init__(fastsolve)
        #* 设定优化方向
        self.set_optimization_direction('min')
        # 需求点个数
        self.num_demands = num_demands
        # 设施点个数
        self.num_facilities = num_facilities
        # cij需求i到设施j的成本
        self.c = c 
        # fj设施j的建设成本
        self.f = f
        self.num_variables = num_facilities + 2 * num_facilities * num_demands
        self.X = self.add_binary_variables('x', [num_facilities])
        self.Y = self.add_binary_variables('y', [num_demands, num_facilities])
        self.Z = self.add_binary_variables('z', [num_demands, num_facilities])

        self.objective_penalty = self.get_objective_func_penalty
        self.objective_cyclic = self.get_objective_func_cyclic
        self.objective_commute = self.get_objective_func_commute
        self.feasible_solution = self.get_feasible_solution()
        #* 直接加目标函数表达式
        self.add_linear_objective(np.multiply(list(itertools.chain(f, *c)), self.cost_dir))
        pass
    
    @property
    def linear_constraints(self):
        if self._linear_constraints is None:
            n = self.num_facilities
            m = self.num_demands
            total_rows = n * m + m
            total_columns = self.num_variables + 1
            matrix = np.zeros((total_rows, total_columns))
            for i in range(m):
                for j in range(n):
                    matrix[n * m + i, self.var_to_idex(self.Y[i][j])] = 1
                    # y_ij + z_ij - x_j = 1
                    matrix[j * m + i, self.var_to_idex(self.Y[i][j])] = 1
                    matrix[j * m + i, self.var_to_idex(self.Z[i][j])] = 1
                    matrix[j * m + i, self.var_to_idex(self.X[j])] = -1
                matrix[n * m + i, total_columns - 1] = 1
            self._linear_constraints = matrix
        return self._linear_constraints

    def fast_solve_driver_bitstr(self):
        n, m  = self.num_facilities,self.num_demands
        # 自由变量个数
        row = n * m - m + n
        column = 2 * m * n + n
        matrix = np.zeros((row, column))
        # p1
        for i in range(n - 1):
            matrix[i, 0] = -1
            matrix[i, i + 1] = 1
            for j in range(m):
                matrix[i, n + j * n] = -1
                matrix[i, n + j * n + i + 1] = 1
        # p2
        for j in range(m - 1):
            for i in range(n - 1):
                matrix[n - 1 + j * (n - 1) + i, n + j * n] = 1
                matrix[n - 1 + j * (n - 1) + i, n + j * n + i + 1] = -1
                matrix[n - 1 + j * (n - 1) + i, n + m * n + j * n] = -1
                matrix[n - 1 + j * (n - 1) + i, n + m * n + j * n + i + 1] = 1
        # p3
        matrix[n - 1 + (m - 1) * (n - 1), 0]= 1
        for j in range(m):
            matrix[n - 1 + (m - 1) * (n - 1), n + n * m  + n * j]= 1
        # p4
        for i in range(n - 1):
            matrix[n - 1 + (m - 1) * (n - 1) + i + 1, i + 1] = 1
            matrix[n - 1 + (m - 1) * (n - 1) + i + 1, n + n * m + n * (m - 1) + i + 1] = 1
            for j in range(m - 1):
                matrix[n - 1 + (m - 1) * (n - 1) + i + 1, n + j * n] = -1
                matrix[n - 1 + (m - 1) * (n - 1) + i + 1, n + j * n + i + 1] = 1
                matrix[n - 1 + (m - 1) * (n - 1) + i + 1, n  + n * m + j * n] = 1
        return matrix
    
    def get_feasible_solution(self):
        """ 根据约束寻找到一个可行解
        """
        # for j in range(self.n):
        self.X[0].set_value(1)
        for i in range(self.num_demands):
            self.Y[i][0].set_value(1)
        # for i in range(self.m):
        #     for j in range(self.n):
        #         self.Z[i][j].set_value(-self.Y[i][j].x + self.X[j].x)
        return [x.x for x in self.variables]

    def get_objective_func_commute(self, variables:Iterable):
        cost = 0
        for i in range(self.num_demands):
            for j in range(self.num_facilities):
                cost += self.c[i][j] * variables[self.var_to_idex(self.Y[i][j])]
        for j in range(self.num_facilities):
            cost += self.f[j] * variables[self.var_to_idex(self.X[j])]
        # commute 只需要目标函数两项
        return self.cost_dir * cost
    
    def get_objective_func_cyclic(self, variables:Iterable):
        cost = 0
        for i in range(self.num_demands):
            for j in range(self.num_facilities):
                cost += self.cost_dir * self.penalty_lambda * (variables[self.var_to_idex(self.Y[i][j])] + variables[self.var_to_idex(self.Z[i][j])] - variables[self.var_to_idex(self.X[j])])**2
        # cyclic 多包含一项非∑=x约束
        return self.get_objective_func_commute(variables) + self.cost_dir * cost

    def get_objective_func_penalty(self, variables:Iterable):
        cost = 0
        for i in range(self.num_demands):
            t = 0
            for j in range(self.num_facilities):
                t += variables[self.var_to_idex(self.Y[i][j])]
            cost += self.cost_dir * self.penalty_lambda * (t - 1)**2
        # penalty 所有约束都惩罚施加
        return self.get_objective_func_cyclic(variables) + self.cost_dir * cost
    
    def get_best_cost(self):
        """ Solve the FLP using Gurobi """
        try:
            model = gp.Model()
            model.setParam('OutputFlag', 0)
            # Add variables
            x = model.addVars(self.num_facilities, vtype=gp.GRB.BINARY, name="x")
            y = model.addVars(self.num_demands, self.num_facilities, vtype=gp.GRB.BINARY, name="y")

            # Objective function
            model.setObjective(gp.quicksum(self.c[i][j] * y[i, j] for i in range(self.num_demands) for j in range(self.num_facilities)) + 
                               gp.quicksum(self.f[j] * x[j] for j in range(self.num_facilities)), gp.GRB.MINIMIZE)

            # Constraints
            model.addConstrs((gp.quicksum(y[i, j] for j in range(self.num_facilities)) == 1 for i in range(self.num_demands)), "DemandAssignment")
            model.addConstrs((y[i, j] <= x[j] for i in range(self.num_demands) for j in range(self.num_facilities)), "FacilityOpen")

            # Optimize the model
            model.optimize()

            if model.status == gp.GRB.OPTIMAL:
                # solution = {var.varName: var.x for var in model.getVars()}
                optimal_value = model.objVal
                # return solution, optimal_value
                return optimal_value
            else:
                return None
        except gp.GurobiError as e:
            print(f'Error code {e.errno}: {e}')
            return None

    # def get_solution_bitstr(self):
    #     """ Convert the optimal solution to a bit string """
    #     solution, optimal_value = self.solve_with_gurobi()
    #     if solution is None:
    #         return None

    #     bitstr = ""
    #     for var_name, value in solution.items():
    #         bitstr += str(int(value))
    #     return bitstr, optimal_value
        