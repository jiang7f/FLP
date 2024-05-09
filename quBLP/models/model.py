## This file defines a problem of binary constraint optimization 
import numpy as np
from typing import Iterable, List
from ..utils.linear_system import find_basic_solution
from ..utils.parseexpr import split_expr
from ..solvers import solve
class Model:
    def __init__(self) -> None:
        pass

class Expression:
    def __init__(self,expr:str) -> None:
        self.expr = expr
        pass
    def __add__(self,other):
        if isinstance(other,Expression):
            return self.expr + other.expr
        
        return self.expr + other
    def __sub__(self,other):
        return self.expr - other
    def __eq__(self,other):
        return self.expr == other
    
   
class Variable:
    def __init__(self,name:str) -> None:
        self.name = name
        self.x = 0
        pass
    def set_value(self, value):
        self.x = value
    def __eq__(self,other):
        return self.x == other
    def __add__(self,other):
        if isinstance(other,Variable):
            return self.x + other.x
        return Expression(self) + Expression(other)
    def __sub__(self,other):
        if isinstance(other,Variable):
            return self.x - other.x
        return self.x - other
    def __mul__(self,other):
        if isinstance(other,Variable):
            return self.x * other.x
        return self.x * other

    
    
    

## Define the problem
class ConstrainedBinaryOptimization(Model):
    def __init__(self,fastsolve=False):
        """ a constrainted binary optimization problem

        Args:
            fastsolve (bool, optional): whether to use fast solve method to get the bitstring for driver hamiltonian. Defaults to False.
        """
        self.fastsolve = fastsolve
        self.variables = []
        self.constraints = []
        self.linear_objective = []
        self.objective = None
        self.variables_idx = {}
        self.current_var_idx = 0
        self.variable_name_set = set()
        pass
    def add_binary_variables(self, name:str, shape:List[int]):
        """ Add `num` 0-1 variables. 
        
        If the variable is already in the dictionary, provide feedback and then skip this addition.

        Args:
            name (str): the label of varibles. 
            shape (int): the shape of the variables array. 

        Example:
            add_binary_variables('x', [5]): ['x_0', 'x_1', 'x_2', 'x_3', 'x_4']
            add_binary_variables('x', [2, 2]): [['x_0_0', 'x_0_1'], ['x_1_0', 'x_1_1']]
        """
        if name in self.variable_name_set:
            print(f'varible {name} already exist')
            return None
        self.variable_name_set.add(name)
        def gnrt_variables(prefix, shape):
            if len(shape) == 1:
                var_name_list = [f"{prefix}_{i}" for i in range(shape[0])]
                var_list = [Variable(var) for var in var_name_list]
                for var_name in var_name_list:
                    self.variables_idx[var_name] = self.current_var_idx
                    self.current_var_idx += 1
                self.variables.extend(var_list)
                return var_list
            else:
                variables = []
                for i in range(shape[0]):
                    sub_indices = gnrt_variables(prefix=f"{prefix}_{i}", shape=shape[1:])
                    variables.append(sub_indices)
                return variables
        BVars = gnrt_variables(name, shape)
        return BVars
    def add_binary_variable(self, name:str):
        """ Add one 0-1 variable

        Args:
            name (str): the name to label variable
        """
        var = Variable(name)
        if name in self.variables_idx:
            print(f'varible {name} already exist')
            return None
        self.variables.append(var)
        self.variables_idx[name] = self.current_var_idx
        self.current_var_idx += 1
        return var
    # 输入变量对象, 输出变量在问题对象变量列表中的下标, 算cost用
    def var_to_idex(self, var: Variable):
        return self.variables_idx[var.name]
    def add_constraint(self, expr):
        """
        Add a constraint to the optimization problem.

        Args:
            expr : A string representing a linear expression with variables and coefficients, followed by an equality or inequality sign.
                   e.g., '2*x_0 + 3*x_1 + 3*x_2 == 1'.
        """
        ## extract the coefficients and the variables from the expression like 2*x_0 + 3*x_1 + 3*x_2 == 1
        print(expr)
        coefficients = np.zeros(len(self.variables) + 1)
        ## parse the expression:2*x_0 + 3*x_1 + 3*x_2 == 1
        for sign,term in split_expr(expr.split('==')[0]):
            sign = 1 if sign == '+' else -1
            if '*' in term:
                coefficient,variable = term.split('*')
                coefficients[self.variables_idx[variable]] = sign*int(coefficient)
            else:
                variable = term
                coefficients[self.variables_idx[variable]] = sign
        coefficients[-1]= int(expr.split('==')[1])
        print(coefficients)
        self._add_linear_constraint(coefficients)
        ## get the coefficients and the variables

    def _add_linear_constraint(self, coefficients: Iterable):
        """add one linear constraint to the problem

        Args:
            coefficients (Iterable): [a_0, a_1,..., a_n-1, b] represents a_0*x_0 + a_1*x_1 + ... + a_n-1*x_n-1 == b
        """
        assert len(coefficients) == 1 + len(self.variables)
        self.constraints.append(coefficients)
    
    def _add_linear_objective(self, coefficients: Iterable):
        """add the objective function to the problem

        Args:
            coefficients (Iterable): [c_0, c_1,..., c_n-1] represents c_0*x_0 + c_1*x_1 + ... + c_n-1*x_n-1
        """
        self.linear_objective = coefficients

    def add_eq_constraint(self, coefficients: Iterable, variable):
        """_summary_

        Args:
            coefficients (Iterable): _description_
        """
        pass
    @property
    def linear_constraints(self):
        return []
    def get_driver_bitstr(self):
        if self.fastsolve:
            return self.fast_solve_driver_bitstr()
        # 如果不使用解析的解系, 则输入约束, 高斯消元求解
        for cstrt in self.linear_constraints:
            self._add_linear_constraint(cstrt)
        basic_vector = find_basic_solution(np.array(self.constraints)[:,:-1])
        return basic_vector
    def add_objective(self,objectivefunc):
        self.objective = objectivefunc
    def set_objective(self, expr):
        pass
    def get_feasible_solution(self):
        ## find a feasible solution for the constraints
        for i in range(1 << len(self.variables)):
            bitstr = [int(j) for j in list(bin(i)[2:].zfill(len(self.variables)))]
            if all([np.dot(bitstr,constr[:-1]) == constr[-1] for constr in self.constraints]):
                return np.nonzero(bitstr)[0]
        return
    def optimize(self, max_iter=30,learning_rate=0.1,num_layers=2):
        self.driver_bitstrs = self.get_driver_bitstr()
        # print(f'driverstr:\n {self.driver_bitstrs}') #-
        self.feasiable_state = self.get_feasible_solution()
        # print(f'fsb_state:{self.feasiable_state}') #-
        best_solution,cost = solve(self.variables, self.objective, self.driver_bitstrs, self.feasiable_state, max_iter, learning_rate, num_layers)
        self.solution = best_solution
        self.objVal = cost
        return best_solution

